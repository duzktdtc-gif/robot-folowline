#include <SoftwareSerial.h>

// RX, TX nối với ESP32
SoftwareSerial espSerial(2, 3);

// ĐỘNG CƠ TRÁI
#define L_EN 5   // PWM
#define L_IN1 8  // Chiều
#define L_IN2 9  // Chiều

// ĐỘNG CƠ PHẢI
#define R_EN 6   // PWM
#define R_IN3 10 // Chiều
#define R_IN4 11 // Chiều

void setup() {
  Serial.begin(115200);
  espSerial.begin(38400); // Khớp với ESP32

  pinMode(L_EN, OUTPUT); pinMode(L_IN1, OUTPUT); pinMode(L_IN2, OUTPUT);
  pinMode(R_EN, OUTPUT); pinMode(R_IN3, OUTPUT); pinMode(R_IN4, OUTPUT);
}

// Hàm điều khiển 1 động cơ hỗ trợ số ÂM (quay lùi)
void driveMotor(int enPin, int in1Pin, int in2Pin, int speed) {
  if (speed > 0) {
    // Quay TIẾN
    digitalWrite(in1Pin, HIGH);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, speed);
  } else if (speed < 0) {
    // Quay LÙI (Đảo chiều)
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, HIGH);
    analogWrite(enPin, -speed); // Lấy giá trị dương cho PWM
  } else {
    // Dừng
    digitalWrite(in1Pin, LOW);
    digitalWrite(in2Pin, LOW);
    analogWrite(enPin, 0);
  }
}

void loop() {
  if (espSerial.available()) {
    String s = espSerial.readStringUntil('\n');
    int L, R;

    // Nhận chuỗi "L:110 R:-110"
    if (sscanf(s.c_str(), "L:%d R:%d", &L, &R) == 2) {

      // Giới hạn an toàn
      L = constrain(L, -255, 255);
      R = constrain(R, -255, 255);

      // Điều khiển động cơ
      driveMotor(L_EN, L_IN1, L_IN2, L);
      driveMotor(R_EN, R_IN3, R_IN4, R);

      // Serial.print("L:"); Serial.print(L); Serial.print(" R:"); Serial.println(R);
    }
  }
}
