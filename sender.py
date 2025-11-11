import multiprocessing
import time
import socket
import threading

from multiprocessing import Queue
from queue import Full, Empty

def _recv_latest(q: Queue, default=None):
    """Drain queue to get the latest value immediately."""
    val = default
    while True:
        try:
            val = q.get_nowait()
        except Exception:
            break
    return val

class UDPSender(multiprocessing.Process):
    def __init__(self, esp_ip, esp_port, angle_queue: Queue, msg_format="ANG:{angle}\n", send_hz=20, verbose=True):
        super().__init__(daemon=True)
        self.esp_ip = esp_ip
        self.esp_port = esp_port
        self.angle_queue = angle_queue
        self.msg_format = msg_format
        self.period = 1.0 / float(send_hz)
        self.verbose = verbose
        self._stop_event = multiprocessing.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def stop(self):
        self._stop_event.set()
        try:
            self.sock.close()
        except Exception:
            pass

    def run(self):
        last_angle = 0
        last_send = 0.0
        while not self._stop_event.is_set():
            # get the most recent angle (if any)
            latest = _recv_latest(self.angle_queue, default=None)
            if latest is not None:
                last_angle = int(latest)

            now = time.time()
            if now - last_send >= self.period:
                msg = self.msg_format.format(angle=last_angle).encode("utf-8")
                try:
                    self.sock.sendto(msg, (self.esp_ip, self.esp_port))
                    if self.verbose:
                        print(f"[UDP] send -> {self.esp_ip}:{self.esp_port}  {msg.decode().strip()}")
                except Exception as e:
                    if self.verbose:
                        print(f"[UDP] ERROR: {e}")
                last_send = now
            # short sleep to avoid busy loop
            time.sleep(0.001)