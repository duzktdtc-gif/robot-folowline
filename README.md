
# Modular YOLO Line-Following (Windows Ready)

Cấu trúc mô-đun, đa luồng: **receiver (TCP)** → **detector (YOLO)** → **controller (PID)** → **sender (UDP)**.

## Cài đặt (Windows)
1. Cài Python 3.10/3.11 (64-bit).
2. Tạo venv (khuyến nghị) và cài:
   ```bash
   pip install -r requirements.txt
   ```
   > Nếu có GPU NVIDIA, cài PyTorch CUDA theo hướng dẫn chính thức trước khi cài `ultralytics`.

3. Đặt `best.pt` cạnh `main.py` (hoặc chỉnh `YOLO_MODEL_PATH` trong `config.py`).

## Chạy
```bash
python main.py
```

## Cấu hình nhanh
Xem `config.py`:
- `PREFER_SEGMENTATION = True`: nếu model có mask → dùng mask (chính xác hơn). Nếu không có mask, code tự động fallback sang bbox.
- `RESIZE_INPUT = (320, 240)`: giảm độ trễ.
- `SEND_HZ = 20`: tần số gửi lệnh UDP.
- `KP/KI/KD`: PID tuning.
- `ROI_BOTTOM_FRACTION = 0.45`: chỉ xét 45% đáy ảnh.

## Ghi chú
- Hàng đợi `maxsize=1` giúp **giảm độ trễ**: luôn xử lý khung hình mới nhất, bỏ qua khung cũ.
- Nhấn `q` để thoát.
