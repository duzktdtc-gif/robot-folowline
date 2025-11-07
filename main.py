import time
import cv2
#from queue import Queue
from multiprocessing import Queue
from threading import Event

import config as C
from receiver import FrameReceiver
from detector import Detector
from controller import Controller
from sender import UDPSender

def main():
    stop_event = Event()

    q_frames = Queue(maxsize=C.FRAME_QUEUE_MAX)
    q_errors = Queue(maxsize=C.ERROR_QUEUE_MAX)
    q_angles = Queue(maxsize=C.ANGLE_QUEUE_MAX)
    q_vis = Queue(maxsize=C.VIS_QUEUE_MAX)

    # Threads
    recv = FrameReceiver(C.HOST, C.TCP_PORT, q_frames, resize=C.RESIZE_INPUT, verbose=C.VERBOSE)
    det = Detector(C.YOLO_MODEL_PATH, C.DEVICE, C.IMG_SIZE, C.CONF_THRES, C.IOU_THRES,
                   C.ROI_BOTTOM_FRACTION, q_frames, q_errors, q_vis,
                   prefer_segmentation=C.PREFER_SEGMENTATION, verbose=C.VERBOSE)
    ctrl = Controller(C.KP, C.KI, C.KD, C.ANGLE_LIMIT, C.DEFAULT_ANGLE_ON_LOST,
                      q_errors, q_angles, max_lost_count=C.MAX_LOST_COUNT, verbose=C.VERBOSE, ema_alpha=0.25)
    snd = UDPSender(C.ESP_IP, C.ESP_PORT, q_angles, msg_format=C.UDP_MESSAGE_FORMAT, send_hz=C.SEND_HZ, verbose=C.VERBOSE)

    print("[MAIN] starting threads ...")
    recv.start()
    det.start()
    ctrl.start()
    snd.start()

    last_show = time.time()
    try:
        while True:
            if C.SHOW_WINDOW:
                try:
                    img = q_vis.get(timeout=0.02)
                    cv2.imshow("Line Tracking (YOLO + PID)", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
            else:
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user")
    finally:
        print("[MAIN] stopping ...")
        recv.stop()
        det.stop()
        ctrl.stop()
        snd.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
