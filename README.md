# YOLO-based Line Following Robot
Hệ thống robot dò line sử dụng AI YOLO (detection/segmentation) kết hợp PID controller,
nhận ảnh từ ESP32 qua TCP và gửi góc lái điều khiển qua UDP theo thời gian thực.

## Cấu trúc thư mục
.
 main.py          # Khởi chạy hệ thống
 config.py        # Cấu hình mạng, YOLO, PID
 detector.py      # YOLO detection / segmentation
 receiver.py      # Nhận ảnh TCP từ ESP32
 controller.py    # PID controller
sender.py        # Gửi lệnh điều khiển UDP
 best.pt          # YOLO trained model
README.md
