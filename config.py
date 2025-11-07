
# -*- coding: utf-8 -*-
"""
Global configuration for the modular line-following system (Windows friendly).
"""

# --- Network (TCP image in, UDP control out) ---
HOST = "0.0.0.0"      # TCP server bind for receiving frames
TCP_PORT = 5050
ESP_IP = "192.168.26.103"  # <-- chỉnh IP ESP32 của bạn
ESP_PORT = 8080

# --- YOLO ---
YOLO_MODEL_PATH = "best_goc.pt"  # để file .pt ở cùng thư mục với main.py
DEVICE = "auto"               # "auto" | 0 (cuda:0) | "cpu" | "xpu"
IMG_SIZE = 320                # imgsz cho YOLO (320 là nhanh và đủ tốt)
CONF_THRES = 0.15
IOU_THRES = 0.45
PREFER_SEGMENTATION = True    # Nếu model có mask → dùng mask cho chính xác hơn
RESIZE_INPUT = (320, 240)     # Resize frame trước khi inference (W, H) - None để giữ nguyên

# ROI: chỉ lấy vùng đáy ảnh (gần robot) để centroid ổn định
ROI_BOTTOM_FRACTION = 0.45    # 45% đáy ảnh

# --- PID & điều khiển ---
KP = 0.55
KI = 0.002
KD = 0.12
ANGLE_LIMIT = 45              # độ
DEFAULT_ANGLE_ON_LOST = 0     # khi mất line nhiều khung

# --- Gửi UDP ---
SEND_HZ = 20                  # tần số gửi lệnh (Hz)
UDP_MESSAGE_FORMAT = "ANG:{angle}\n"

# --- Hàng đợi & hiển thị ---
FRAME_QUEUE_MAX = 1           # đặt 1 để luôn xử lý frame mới nhất (giảm độ trễ)
ERROR_QUEUE_MAX = 2
ANGLE_QUEUE_MAX = 2
VIS_QUEUE_MAX = 1
SHOW_WINDOW = True            # tắt nếu chạy headless (không GUI)

# --- Failsafe ---
MAX_LOST_COUNT = 6            # số khung liên tiếp mất line thì reset PID
VERBOSE = True
