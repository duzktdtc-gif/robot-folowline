import time
#import threading
import numpy as np
import cv2
#from queue import Queue, Full, Empty

import multiprocessing
from queue import Full, Empty
from multiprocessing import Queue

from ultralytics import YOLO

def _offer(q: Queue, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

def resolve_device(pref):
    if isinstance(pref, int):
        return pref
    if isinstance(pref, str) and pref not in ("auto",):
        return pref
    # auto
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

def get_mask_centroid(mask_slice, x_offset = 0):
    if mask_slice.sum()==0:
        return None, None

    if mask_slice.ndim == 2:
        M =
class Detector(multiprocessing.Process):
    def __init__(self, model_path, device, imgsz, conf, iou, roi_bottom_frac,
                 frame_queue: Queue, error_queue: Queue, vis_queue: Queue,
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
        self._stop_event = multiprocessing.Event()
        self.model = None
        # FPS stats
        self._last_frame_time = None
        self._ema_fps = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        self.model = YOLO(self.model_path)
        if self.verbose:
            print(f"[DETECTOR] Model loaded: {self.model_path}, device={self.device}")
        while not self._stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except Empty:
                continue
            t0 = time.time()
            # YOLO predict
            results = self.model.predict(
                frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False
            )
            r = results[0]
            img = frame.copy()
            h, w = img.shape[:2]
            cx, cy, found = None, None, False

            y0 = int(h * (1.0 - self.roi_bottom_frac))
            roi_rect = (0, y0, w, h - y0)

            # Try segmentation first (if preferred and masks exist)
            used_seg = False
            if self.prefer_segmentation and hasattr(r, "masks") and (r.masks is not None):
                try:
                    masks = r.masks.data  # (N, h, w)
                    if masks is not None and len(masks) > 0:
                        best_area = -1
                        best_centroid = None
                        for i in range(len(masks)):
                            mask_i = masks[i].cpu().numpy().astype(np.uint8) * 255  # 0/255
                            mask_i = cv2.resize(mask_i, (w, h), interpolation=cv2.INTER_NEAREST)
                            roi_mask = mask_i[y0:, :]
                            area = int(roi_mask.sum() // 255)
                            if area <= 0:
                                continue
                            M = cv2.moments(roi_mask, binaryImage=True)
                            if M["m00"] > 0:
                                cx_i = (M["m10"] / M["m00"])
                                cy_i = (M["m01"] / M["m00"]) + y0
                                if area > best_area:
                                    best_area = area
                                    best_centroid = (cx_i, cy_i)
                        if best_centroid is not None:
                            cx, cy = best_centroid
                            found = True
                            used_seg = True
                            # visualize mask overlay
                            overlay = img.copy()
                            overlay[y0:, :] = cv2.addWeighted(
                                overlay[y0:, :],
                                0.6,
                                cv2.merge([np.zeros_like(mask_i[y0:, :]), mask_i[y0:, :], np.zeros_like(mask_i[y0:, :])]),
                                0.4, 0
                            )
                            img = overlay
                except Exception as e:
                    if self.verbose:
                        print(f"[DETECTOR] Segmentation branch error: {e}")

            # Fallback to detection (boxes)
            if not found:
                boxes = []
                try:
                    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        for i, bb in enumerate(xyxy):
                            x1, y1, x2, y2 = bb.astype(int).tolist()
                            conf_i = float(confs[i]) if i < len(confs) else 0.0
                            boxes.append((x1, y1, x2, y2, conf_i))
                except Exception:
                    pass

                if boxes:
                    # chọn box giao với ROI có diện tích lớn nhất
                    best_score = -1
                    best_box = None
                    for (x1, y1, x2, y2, conf_i) in boxes:
                        # area of intersection with ROI
                        ix1 = max(x1, 0)
                        iy1 = max(y1, y0)
                        ix2 = min(x2, w-1)
                        iy2 = min(y2, h-1)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        score = inter if inter > 0 else (x2-x1)*(y2-y1)  # ưu tiên giao ROI
                        if score > best_score:
                            best_score = score
                            best_box = (x1, y1, x2, y2, conf_i)
                    if best_box is not None:
                        x1, y1, x2, y2, conf_i = best_box
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        found = True
                        # draw bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Visuals common
            cv2.rectangle(img, (roi_rect[0], roi_rect[1]), (roi_rect[0] + roi_rect[2], roi_rect[1] + roi_rect[3]), (255, 200, 0), 2)
            cv2.line(img, (w//2, 0), (w//2, h), (255, 0, 0), 1)

            if found and (cx is not None):
                cv2.circle(img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                error = (cx - (w / 2.0)) / (w / 2.0)  # normalize to [-1, 1]
                error = max(-1.0, min(1.0, error))
            else:
                error = None

            # timings
            latency_ms = (time.time() - t0) * 1000.0

            # FPS (processed loop rate)
            now2 = time.time()
            if self._last_frame_time is not None:
                dt = now2 - self._last_frame_time
                inst_fps = (1.0 / dt) if dt > 0 else 0.0
                if self._ema_fps is None:
                    self._ema_fps = inst_fps
                else:
                    self._ema_fps = 0.9 * self._ema_fps + 0.1 * inst_fps
            self._last_frame_time = now2

            # overlay text
            cv2.putText(img, f"latency: {latency_ms:.1f} ms  mode: {'SEG' if used_seg else 'BOX'}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            if self._ema_fps is not None:
                cv2.putText(img, f"FPS: {self._ema_fps:.1f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            _offer(self.vis_queue, img)
            _offer(self.error_queue, {"error": error, "timestamp": time.time()})
