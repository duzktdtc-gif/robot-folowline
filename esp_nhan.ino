#include <WiFi.h>
#include <WiFiUdp.h>

const char* ssid = "s23";
const char* password = "duc123abcc";

WiFiUDP udp;
const int UDP_PORT = 8080;      // phải trùng PC
const int UART_BAUD = 115200;

const int BASE_SPEED = 150;
const float K = 3.0;

int clamp(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void setup() {
  Serial.begin(UART_BAUD);
  Serial.setTimeout(5);

  WiFi.begin(ssid, password);
  Serial.println("Connecting WiFi...");
  while (WiFi.status() != WL_CONNECTED) delay(200);

  Serial.print("ESP32-CAM #2 IP: ");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
  Serial.println("UDP listener started.");
}

void loop() {
  char buf[64];
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len > 0) buf[len] = 0;

    // Expect message: "ANG:xx"
    int ang = 0;
    if(sscanf(buf, "ANG:%d", &ang) == 1) {
      // Compute wheel speeds
      int left = BASE_SPEED - K * ang;
      int right = BASE_SPEED + K * ang;

      left = clamp(left, 0, 255);
      right = clamp(right, 0, 255);

      // Send to Arduino
      Serial.printf("L:%d R:%d\n", left, right);
    }
  }
}
