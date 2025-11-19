// ================= MOTOR PINS =================
#define ENA 9     // PWM Left
#define IN1 7
#define IN2 8

#define ENB 10    // PWM Right
#define IN3 5
#define IN4 6

// ================== SETUP =====================
void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();
}

// ================= MOTOR CONTROL ================
void setLeft(int speed) {
  if (speed >= 0) {
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);
  } else {
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
    speed = -speed;
  }
  analogWrite(ENA, constrain(speed, 0, 255));
}

void setRight(int speed) {
  if (speed >= 0) {
    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);
  } else {
    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
    speed = -speed;
  }
  analogWrite(ENB, constrain(speed, 0, 255));
}

void stopMotors() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
}

// ================= MAIN LOOP ====================
// đọc liên tục từng ký tự, ghép thành 1 dòng
String inputLine = "";

void loop() {

  while (Serial.available() > 0) {

    char c = Serial.read();

    // Nếu ký tự kết thúc dòng
    if (c == '\n') {

      int L, R;

      // Parse đúng format: L:100 R:-150
      if (sscanf(inputLine.c_str(), "L:%d R:%d", &L, &R) == 2) {

        Serial.print("→ L=");
        Serial.print(L);
        Serial.print("  R=");
        Serial.println(R);

        setLeft(L);
        setRight(R);
      }

      // Xóa dòng sau khi xử lý
      inputLine = "";
    }
    else {
      // Ghép ký tự vào buffer
      inputLine += c;
    }
  }
}