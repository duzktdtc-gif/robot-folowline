import cv2
import time
import socket
import struct
import numpy as np
import threading

from multiprocessing import Queue
from queue import Full, Empty

#from queue import Queue, Full, Empty
from typing import Tuple

def _offer(q: Queue, item):
    """Put item into queue, discarding the oldest if full (low-latency)."""
    try:
        q.put_nowait(item)
    except Full:
        try:
            _ = q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

class FrameReceiver(threading.Thread):
    def __init__(self, host: str, port: int, frame_queue: Queue, resize: Tuple[int, int] | None, verbose=True):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.frame_queue = frame_queue
        self.resize = resize
        self.verbose = verbose
        self._stop_event = threading.Event()
        self.server = None
        self.conn = None
        self.addr = None

    def stop(self):
        self._stop_event.set()
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.server:
                self.server.close()
        except Exception:
            pass

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._serve_once()
            except Exception as e:
                if self.verbose:
                    print(f"[RECEIVER] Error: {e}. Re-listen in 2s...")
                time.sleep(2.0)

    def _serve_once(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        if self.verbose:
            print(f"[RECEIVER] Listening on {self.host}:{self.port} ...")
        self.conn, self.addr = self.server.accept()
        if self.verbose:
            print(f"[RECEIVER] Connected: {self.addr}")

        data = b""
        header_size = 4  # little-endian uint32 length

        while not self._stop_event.is_set():
            # read header
            while len(data) < header_size:
                packet = self.conn.recv(4096)
                if not packet:
                    raise ConnectionError("socket closed while reading header")
                data += packet

            packed_len = data[:header_size]
            data = data[header_size:]
            msg_len = struct.unpack("<L", packed_len)[0]
            # read payload
            while len(data) < msg_len:
                packet = self.conn.recv(4096)
                if not packet:
                    raise ConnectionError("socket closed while reading frame")
                data += packet
            frame_buf = data[:msg_len]
            data = data[msg_len:]

            # decode
            img = cv2.imdecode(np.frombuffer(frame_buf, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                if self.verbose:
                    print("[RECEIVER] WARN: failed to decode frame")
                continue

            if self.resize is not None:
                img = cv2.resize(img, self.resize, interpolation=cv2.INTER_LINEAR)

            _offer(self.frame_queue, img)
        # done loop
