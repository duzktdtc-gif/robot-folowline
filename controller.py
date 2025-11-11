import multiprocessing
import time
import threading
#from queue import Queue, Full, Empty
import numpy as np
from multiprocessing import Queue
from queue import Full, Empty

from utils.pid import PIDController

def _offer(q: Queue, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

class Controller(multiprocessing.Process):
    def __init__(self, kp, ki, kd, angle_limit, default_angle_on_lost,
                 error_queue: Queue, angle_queue: Queue,
                 max_lost_count=6, verbose=True, ema_alpha=0.25):
        super().__init__(daemon=True)
        self._pid = PIDController(kp, ki, kd, windup=1000.0)
        self.angle_limit = float(angle_limit)
        self.default_angle_on_lost = float(default_angle_on_lost)
        self.error_queue = error_queue
        self.angle_queue = angle_queue
        self.max_lost_count = int(max_lost_count)
        self.verbose = verbose
        self._stop_event = multiprocessing.Event()
        self._lost_count = 0
        self._ema_alpha = float(ema_alpha)
        self._ema_angle = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                pkt = self.error_queue.get(timeout=0.2)
            except Empty:
                continue
            err = pkt.get("error", None)
            now = time.time()
            if err is None:
                self._lost_count += 1
                if self._lost_count > self.max_lost_count:
                    self._pid.reset()
                    angle = self.default_angle_on_lost
                else:
                    angle = self.default_angle_on_lost
            else:
                self._lost_count = 0
                pid_out = self._pid.update(err, now=now)
                # scale PID directly to angle range assuming |pid_out|<=1 with tuned Kp
                # clamp after scaling
                angle = pid_out * self.angle_limit
                if angle > self.angle_limit:
                    angle = self.angle_limit
                elif angle < -self.angle_limit:
                    angle = -self.angle_limit

            # EMA smoothing
            if self._ema_angle is None:
                self._ema_angle = angle
            else:
                self._ema_angle = (1.0 - self._ema_alpha) * self._ema_angle + self._ema_alpha * angle

            _offer(self.angle_queue, int(round(self._ema_angle)))