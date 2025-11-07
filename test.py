# PC-side: nhận frame từ ESP32-CAM (TCP), YOLO xử lý line, PID tính góc, gửi góc về ESP32 (UDP)
import socket
import struct
import numpy as np
import cv2
import time
from collections import deque

from torch.xpu import device
# --- Thư viện YOLO (ultralytics) ---
# pip install ultralytics
from ultralytics import YOLO

# ------------- CẤU HÌNH -------------
# Địa chỉ lắng nghe (nhận ảnh từ ESP32)
HOST = "0.0.0.0"
PORT = 5050

# ESP32 nhận lệnh (UDP)
ESP_IP = "192.168.26.103"   # <-- sửa thành IP ESP32 của bạn
ESP_PORT = 8080

# Đường dẫn model YOLO (best.pt)
YOLO_MODEL_PATH = "best_goc.pt"  # <-- sửa nếu tên khác

# Tham số PID (tune cho robot của bạn)
KP = 0.5
KI = 0.002
KD = 0.12

# Giới hạn góc gửi về (degree)
ANGLE_LIMIT = 45  # -45 ... +45

# Tốc độ gửi tối đa (giảm bão lệnh)
SEND_INTERVAL = 0.05  # giây (20Hz)

# Lựa chọn: nếu model không phát hiện line => hành động mặc định
DEFAULT_ANGLE_ON_LOST = 0  # 0 = thẳng

# Lưu 1 số thông tin log
VERBOSE = True
# ------------------------------------

# --- Tạo socket UDP để gửi lệnh về ESP32 ---
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Load model YOLO ---
model = YOLO(YOLO_MODEL_PATH)
# Nếu model trả về segmentation bạn sẽ cần thay logic để lấy mask centroid.
# Ở đây ta giả sử model detect line và trả về bounding boxes.

# --- PID class ---
class PIDController:
    def __init__(self, kp, ki, kd, windup=1000):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None
        self.windup = windup
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now=None):
        if now is None:
            now = time.time()
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = now - self.prev_time
        # Proportional
        p = self.kp * error
        # Integral
        if dt > 0:
            self.integral += error * dt
            # anti-windup
            self.integral = max(min(self.integral, self.windup), -self.windup)
        i = self.ki * self.integral
        # Derivative
        if self.prev_error is None or dt == 0:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt
        # update state
        self.prev_error = error
        self.prev_time = now
        output = p + i + d
        return output

pid = PIDController(KP, KI, KD)

# --- TCP server (nhận frame) ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"[SERVER] Listening on {HOST}:{PORT} ...")

conn, addr = server.accept()
print(f"[CONNECTED] {addr}")

data = b""
payload_size = 4  # kích thước header 4 byte như bạn đã dùng

last_send_time = 0.0

try:
    while True:
        # --- Nhận kích thước ảnh (4 byte) ---
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < payload_size:
            print("[INFO] Kết nối đóng hoặc không đủ dữ liệu header")
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("<L", packed_msg_size)[0]

        # --- Nhận dữ liệu ảnh ---
        while len(data) < msg_size:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < msg_size:
            print("[INFO] Kết nối đóng hoặc không đủ dữ liệu frame")
            break

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # --- Giải mã ảnh ---
        img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            if VERBOSE:
                print("[WARN] Không thể decode ảnh")
            continue

        h, w = img.shape[:2]
        img_center_x = w / 2.0

        # --- Chạy YOLO (inference) ---
        # Chú ý: ultralytics model(frame) trả về kết quả dạng list. Dùng results[0] để lấy đầu ra.
        results = model(img, device = 0 ,verbose=False)  # mặc định trả về nhiều thứ
        r = results[0]

        # Lấy bounding boxes
        # boxes.xyxy: tensor (N,4), boxes.conf: (N,), boxes.cls: (N,)
        boxes = []
        if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
            # r.boxes.xyxyn? depending on model; we'll use xyxy in pixels if present
            try:
                # ultralytics v8: r.boxes.xyxy.numpy(), r.boxes.conf
                xyxy = r.boxes.xyxy.cpu().numpy()  # shape (N,4)
                confs = r.boxes.conf.cpu().numpy()
                # build list of (x1,y1,x2,y2,conf)
                for i, b in enumerate(xyxy):
                    x1, y1, x2, y2 = b
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
            except Exception as e:
                # fallback: try r.boxes.xyxy (list)
                try:
                    for box in r.boxes:
                        b = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = b
                        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                        boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
                except Exception as e2:
                    boxes = []

        # --- Quyết định điểm tham chiếu của line ---
        # Nếu có nhiều box: chọn box có diện tích lớn nhất (giả sử đó là line chính)
        target_center_x = None
        target_center_y = None
        if len(boxes) > 0:
            best = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))  # x1,y1,x2,y2,conf
            x1, y1, x2, y2, conf = best
            target_center_x = (x1 + x2) / 2.0
            target_center_y = (y1 + y2) / 2.0
            # vẽ khung
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.circle(img, (int(target_center_x), int(target_center_y)), 4, (0,0,255), -1)
        else:
            # Nếu model không detect (lost) -> có thể dùng color processing backup (nếu muốn)
            # Ở đây ta để None và dùng DEFAULT_ANGLE_ON_LOST
            target_center_x = None

        # --- Tính error và PID ---
        if target_center_x is None:
            angle_out = DEFAULT_ANGLE_ON_LOST
            # reset integral maybe
            # pid.reset()
        else:
            # error: positive nếu đường lệch sang phải (robot cần rẽ phải), negative nếu trái
            # Normalized error: -1 ... 1
            pixel_error = target_center_x - img_center_x
            norm_error = pixel_error / (img_center_x)  # divide by half-width => -1..1
            # Optionally clamp
            if norm_error > 1: norm_error = 1
            if norm_error < -1: norm_error = -1
            # PID update
            pid_output = pid.update(norm_error, now=time.time())
            # map pid_output to angle in degrees
            # pid_output is in arbitrary scale because KP based on normalized error; tune KP to yield degrees
            angle_out = pid_output * ANGLE_LIMIT  # this maps -1..1 pid_output to -ANGLE_LIMIT..ANGLE_LIMIT (approx)
            # clamp
            angle_out = max(min(angle_out, ANGLE_LIMIT), -ANGLE_LIMIT)

        # --- Ghi thông tin lên ảnh ---
        cv2.line(img, (int(img_center_x), 0), (int(img_center_x), h), (255,0,0), 1)
        text = f"Angle: {angle_out:.2f}"
        cv2.putText(img, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        # Hiển thị ảnh (annotated)
        cv2.imshow("YOLO Line Tracking with PID", img)

        # --- Gửi lệnh về ESP32 qua UDP (dạng: ANG:{angle}\n) ---
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            # Format: gửi số nguyên góc (ví dụ "A:15\n" hoặc "ANGLE:15\n")
            # Lựa chọn: gửi có tiền tố để ESP32 parse dễ dàng. Ta dùng "ANG:{angle_int}\n"
            angle_int = int(round(angle_out))
            msg = f"ANG:{angle_int}\n".encode('utf-8')
            try:
                udp_sock.sendto(msg, (ESP_IP, ESP_PORT))
                if VERBOSE:
                    print(f"[SEND] {msg.decode().strip()} to {ESP_IP}:{ESP_PORT}")
            except Exception as e:
                print("[ERROR] Gửi UDP thất bại:", e)
            last_send_time = now

        # --- thoát nếu nhấn q ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    print("[CLEANUP] Closing sockets and windows")
    try:
        conn.close()
    except:
        pass
    try:
        server.close()
    except:
        pass
    udp_sock.close()
    cv2.destroyAllWindows()
