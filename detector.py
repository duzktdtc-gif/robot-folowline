# -*- coding: utf-8 -*-
"""
detector.py
Detector process using multiprocessing + Look-ahead polynomial fit.

Replace previous threading-based Detector with this file.
"""

import time
import os
import numpy as np
import cv2
import multiprocessing as mp
from queue import Full, Empty  # used with mp.Queue for exceptions

from ultralytics import YOLO  # will be imported in child process when run() executes

# helper to put into queue but drop oldest if full
def _offer(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            # if still failing, give up silently
            pass

def resolve_device(pref):
    """
    Resolve device similar to original logic.
    Return int (cuda index), "xpu", "cpu", or "auto" style str fallback.
    """
    # if user supplied explicit int or non-"auto" string, keep it
    if isinstance(pref, int):
        return pref
    if isinstance(pref, str) and pref not in ("auto",):
        return pref
    # auto-detect
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass
    except Exception:
        pass
    return "cpu"

class Detector(mp.Process):
    """
    Multiprocessing Detector that loads YOLO model in its own process.
    Produces:
      - images with overlays to vis_queue
      - {"error": float_or_None, "timestamp": ...} to error_queue
    """
    def __init__(self, model_path, device, imgsz, conf, iou, roi_bottom_frac,
                 frame_queue, error_queue, vis_queue,
                 prefer_segmentation=True, verbose=True):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.roi_bottom_frac = float(roi_bottom_frac)
        self.frame_queue = frame_queue
        self.error_queue = error_queue
        self.vis_queue = vis_queue
        self.prefer_segmentation = prefer_segmentation
        self.verbose = verbose

        # stop event shared between parent and child (works with mp.Event)
        self._stop_event = mp.Event()

        # model and fps stats (created in child)
        self.model = None
        self._last_frame_time = None
        self._ema_fps = None

    def stop(self):
        # parent can call this to request stop
        try:
            self._stop_event.set()
        except Exception:
            pass

    def run(self):
        # Import config inside the process to ensure child sees same settings
        import config as C

        # Load model inside child process (avoid copying CUDA contexts)
        try:
            self.model = YOLO(self.model_path)
            if self.verbose:
                print(f"[DETECTOR] (PID={os.getpid()}) Model loaded: {self.model_path}  device_resolved={self.device}")
        except Exception as e:
            print(f"[DETECTOR] Failed to load model: {e}")
            # If model fails to load, we should not loop predict; just exit
            return

        # look-ahead parameters (tuneable)
        DEFAULT_NUM_SLICES = 5     # how many horizontal slices inside ROI
        LOOKAHEAD_Y_FACTOR = 0.5   # fraction of image height for lookahead (0..1). 0.5 ~ mid image

        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                continue

            t0 = time.time()
            img = frame.copy()
            h, w = img.shape[:2]

            # YOLO inference
            try:
                results = self.model.predict(
                    frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                    device=self.device, verbose=False
                )
            except Exception as e:
                # inference error — report None and continue
                if self.verbose:
                    print(f"[DETECTOR] Inference error: {e}")
                _offer(self.error_queue, {"error": None, "timestamp": time.time()})
                continue

            if not results or len(results) == 0:
                _offer(self.error_queue, {"error": None, "timestamp": time.time()})
                continue

            r = results[0]

            # Prepare binary mask (0/255) of detected line region
            mask = None
            used_seg = False
            try:
                if self.prefer_segmentation and hasattr(r, "masks") and r.masks is not None:
                    # r.masks.data can be (N, h, w)
                    try:
                        masks = r.masks.data
                        if masks is not None and len(masks) > 0:
                            # pick largest mask by area (within mask array)
                            best_idx = 0
                            best_area = -1
                            for i in range(len(masks)):
                                mi = masks[i].cpu().numpy().astype(np.uint8)
                                area = int(mi.sum())
                                if area > best_area:
                                    best_area = area
                                    best_idx = i
                            m0 = masks[best_idx].cpu().numpy().astype(np.uint8)
                            # m0 is 0/1 or 0/255 depending on version; normalize to 0/255
                            if m0.max() <= 1:
                                m0 = (m0 * 255).astype(np.uint8)
                            else:
                                m0 = m0.astype(np.uint8)
                            # resize mask to frame size (YOLO predict may change shape)
                            m0 = cv2.resize(m0, (w, h), interpolation=cv2.INTER_NEAREST)
                            mask = m0
                            used_seg = True
                    except Exception:
                        # fallback to boxes below
                        mask = None
            except Exception:
                mask = None

            if mask is None:
                # fallback: build mask from the first bbox if available
                try:
                    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                        # use boxes.xyxy[0]
                        xy = r.boxes.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xy.tolist()
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        used_seg = False
                except Exception:
                    mask = None

            found = False
            error = None
            x_predicted = None
            y_lookahead = None

            if mask is not None:
                # Define ROI at bottom of image
                roi_h = int(h * self.roi_bottom_frac)
                roi_y = h - roi_h
                roi_rect = (0, roi_y, w, roi_h)

                mask_roi = mask[roi_y:h, :]

                # Look-ahead: sample several horizontal slices inside ROI
                num_slices = DEFAULT_NUM_SLICES
                if num_slices < 2:
                    num_slices = 2
                slice_y_indices = np.linspace(0, max(0, mask_roi.shape[0] - 1), num_slices, dtype=int)

                x_points = []
                y_points = []
                for y_idx in slice_y_indices:
                    row = mask_roi[y_idx, :]
                    x_coords = np.where(row > 0)[0]
                    if x_coords.size > 0:
                        cx = float(np.mean(x_coords))
                        x_points.append(cx)
                        y_points.append(float(y_idx + roi_y))

                        # visualize measured centroids
                        if C.SHOW_WINDOW:
                            cv2.circle(img, (int(cx), int(y_idx + roi_y)), 3, (0, 255, 255), -1)

                if len(x_points) >= 3:
                    # fit polynomial x = a*y^2 + b*y + c
                    try:
                        coeffs = np.polyfit(y_points, x_points, 2)
                        poly_fn = np.poly1d(coeffs)
                        found = True

                        # choose lookahead y (in full image coordinates)
                        y_lookahead = int(h * LOOKAHEAD_Y_FACTOR)
                        x_predicted = float(poly_fn(y_lookahead))

                        # compute normalized error in [-1, 1]
                        error = (x_predicted - (w / 2.0)) / (w / 2.0)
                        error = float(np.clip(error, -1.0, 1.0))

                        # draw fitted curve for visualization
                        if C.SHOW_WINDOW:
                            vis_y = np.arange(roi_y, h)
                            vis_x = poly_fn(vis_y).astype(int)
                            valid = (vis_x >= 0) & (vis_x < w)
                            vis_xv = vis_x[valid]
                            vis_yv = vis_y[valid]
                            if vis_xv.size > 0:
                                pts = np.vstack((vis_xv, vis_yv)).T.reshape((-1, 1, 2))
                                try:
                                    cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
                                except Exception:
                                    # fallback to point-draw if polylines fails
                                    for (xx, yy) in zip(vis_xv, vis_yv):
                                        cv2.circle(img, (int(xx), int(yy)), 1, (0, 255, 0), -1)
                            # lookahead marker
                            cv2.circle(img, (int(x_predicted), int(y_lookahead)), 6, (0, 165, 255), -1)

                    except np.linalg.LinAlgError:
                        found = False
                        error = None
                    except Exception as e:
                        if self.verbose:
                            print(f"[DETECTOR] polyfit error: {e}")
                        found = False
                        error = None
                else:
                    # Not enough points -> fallback to centroid of the whole ROI
                    # compute centroid of mask_roi if any
                    M = cv2.moments((mask_roi > 0).astype(np.uint8))
                    if M.get("m00", 0) > 0:
                        cx_roi = (M["m10"] / M["m00"])
                        cy_roi = (M["m01"] / M["m00"]) + roi_y
                        error = (cx_roi - (w / 2.0)) / (w / 2.0)
                        error = float(np.clip(error, -1.0, 1.0))
                        found = True
                        x_predicted = cx_roi
                        y_lookahead = int(cy_roi)

            # overlay ROI rectangle and center-line, mode text, error text
            try:
                # draw ROI rect
                if mask is not None:
                    cv2.rectangle(img, (0, int(h - int(h * self.roi_bottom_frac))),
                                  (w - 1, h - 1), (255, 200, 0), 2)
                # center vertical
                cv2.line(img, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
                # optional horizontal at lookahead
                if y_lookahead is not None:
                    cv2.line(img, (0, y_lookahead), (w, y_lookahead), (200, 200, 200), 1)

                latency_ms = (time.time() - t0) * 1000.0
                now2 = time.time()
                if self._last_frame_time is not None:
                    dt = now2 - self._last_frame_time
                    inst_fps = (1.0 / dt) if dt > 0 else 0.0
                    self._ema_fps = inst_fps if self._ema_fps is None else 0.9 * self._ema_fps + 0.1 * inst_fps
                self._last_frame_time = now2

                cv2.putText(img, f"latency: {latency_ms:.1f} ms  mode: {'SEG' if used_seg else 'BOX'}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                if self._ema_fps is not None:
                    cv2.putText(img, f"FPS: {self._ema_fps:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if error is not None:
                    cv2.putText(img, f"Error: {error:.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            except Exception:
                pass

            # push results
            _offer(self.vis_queue, img)
            _offer(self.error_queue, {"error": error, "timestamp": time.time()})
