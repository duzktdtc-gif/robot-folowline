#include <WiFi.h>
#include <WiFiUdp.h>

// ================= CẤU HÌNH WIFI =================
const char* ssid = "s23";      // <--- SỬA WIFI CỦA BẠN
const char* password = "duc123abcc"; // <--- SỬA PASSWORD

// ================= CẤU HÌNH ĐIỀU KHIỂN =================
// Tốc độ cơ bản khi robot chạy thẳng (0-255)
#define BASE_SPEED 150 

// Hệ số "lái". 
// Ví dụ: Góc lệch 10 độ * K_GAIN 4.0 = Chênh lệch tốc độ 40
#define K_GAIN 4.0    

// Định nghĩa chân Motor
#define PIN_MOTOR_L1 12
#define PIN_MOTOR_L2 13
#define PIN_MOTOR_R1 14
#define PIN_MOTOR_R2 15

// Cấu hình PWM
const int FREQ = 1000;
const int RES = 8; // 8 bit: 0-255

WiFiUDP udp;
const int UDP_PORT = 8080; // Port nhận lệnh (phải khớp với config.py trên PC)

// ================= HÀM HỖ TRỢ =================

// Hàm kẹp giá trị (để tốc độ không vượt quá 255 hoặc dưới 0)
int clamp(int v, int min_v, int max_v) {
  if (v < min_v) return min_v;
  if (v > max_v) return max_v;
  return v;
}

// Hàm điều khiển 1 Motor (Hỗ trợ chạy tới và lùi)
// speed: -255 đến 255
void setMotor(int pin1, int pin2, int speed) {
  // Giới hạn an toàn
  if (speed > 255) speed = 255;
  if (speed < -255) speed = -255;

  if (speed > 0) {
    // Chạy tới
    ledcWrite(pin1, speed);
    ledcWrite(pin2, 0);
  } else if (speed < 0) {
    // Chạy lùi
    ledcWrite(pin1, 0);
    ledcWrite(pin2, -speed); // Lấy giá trị dương
  } else {
    // Dừng (Thả trôi hoặc phanh)
    ledcWrite(pin1, 255); 
    ledcWrite(pin2, 255);
  }
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  
  // 1. Cấu hình PWM cho Motor (Cú pháp ESP32 v3.0 mới nhất)
  ledcAttach(PIN_MOTOR_L1, FREQ, RES);
  ledcAttach(PIN_MOTOR_L2, FREQ, RES);
  ledcAttach(PIN_MOTOR_R1, FREQ, RES);
  ledcAttach(PIN_MOTOR_R2, FREQ, RES);
  
  // Dừng động cơ khi khởi động
  setMotor(PIN_MOTOR_L1, PIN_MOTOR_L2, 0);
  setMotor(PIN_MOTOR_R1, PIN_MOTOR_R2, 0);

  // 2. Kết nối WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nWiFi OK!");
  Serial.print("IP Address: "); Serial.println(WiFi.localIP());
  Serial.printf("UDP Listening on port: %d\n", UDP_PORT);

  // 3. Bắt đầu lắng nghe UDP
  udp.begin(UDP_PORT);
}

// ================= LOOP =================
void loop() {
  // Kiểm tra xem có gói tin UDP nào đến không
  int packetSize = udp.parsePacket();
  
  if (packetSize > 0) {
    char buf[64];
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len > 0) {
      buf[len] = 0; // Kết thúc chuỗi ký tự
      
      int angle = 0;
      // Parse tin nhắn dạng "ANG:20" hoặc "ANG:-15"
      if (sscanf(buf, "ANG:%d", &angle) == 1) {
        
        // --- LOGIC TÍNH TOÁN TỐC ĐỘ ---
        
        // Tính lượng "bẻ lái"
        int steer = (int)(angle * K_GAIN);
        
        // Tính tốc độ 2 bánh
        // Góc > 0 (Lệch phải) -> Cần rẽ Phải -> Trái tăng, Phải giảm
        int speed_L = BASE_SPEED + steer;
        int speed_R = BASE_SPEED - steer;
        
        // Đảm bảo tốc độ nằm trong khoảng cho phép
        speed_L = clamp(speed_L, 0, 255);
        speed_R = clamp(speed_R, 0, 255);
        
        // Điều khiển phần cứng
        setMotor(PIN_MOTOR_L1, PIN_MOTOR_L2, speed_L);
        setMotor(PIN_MOTOR_R1, PIN_MOTOR_R2, speed_R);
        Serial.printf("CMD: %-8s =>  ANG: %-3d  =>  L: %-3d | R: %-3d\n", buf, angle, speed_L, speed_R);
        
      } else {
         // Nếu nhận được gói tin nhưng không đúng định dạng ANG
         Serial.print("Unknown format: ");
         Serial.println(buf);
      }
        // Debug (Nếu cần kiểm tra thì bỏ comment dòng dưới)
        // Serial.printf("ANG:%d -> L:%d R:%d\n", angle, speed_L, speed_R);
      
    }
  }
  
  // Không cần delay, loop chạy càng nhanh càng tốt để bắt lệnh kịp thời
}