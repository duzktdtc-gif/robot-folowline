# YOLO-based Line Following Robot
Hệ thống robot dò line sử dụng AI YOLO (detection/segmentation) kết hợp PID controller,
nhận ảnh từ ESP32 qua TCP và gửi góc lái điều khiển qua UDP theo thời gian thực.

## Cấu trúc thư mục
.
├── main.py          # Khởi chạy hệ thống <br>
├── config.py        # Cấu hình mạng, YOLO, PID <br>
├── detector.py      # YOLO detection / segmentation <br>
├── receiver.py      # Nhận ảnh TCP từ ESP32 <br>
├── controller.py    # PID controller <br>
├── sender.py        # Gửi lệnh điều khiển UDP <br>
├── best.pt          # YOLO trained model <br>
└── README.md
