/*
 * ESP32-CAM TCP Client → PC Server (Python)
 * - Gửi frame JPEG: [4-byte little-endian length][JPEG bytes]
 * - Tuỳ chọn: nhận UDP "ANG:<int>\n" để debug (in Serial)
 * 
 * Phù hợp với Python receiver: header 4 byte little-endian và imdecode (OpenCV).
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>

// ========== USER CONFIG ==========
#define WIFI_SSID       "TP-LINK_54FC"
#define WIFI_PASS       "12301230"

// PC (chạy main.py lắng nghe TCP ở port 5050)
#define PC_SERVER_IP    "192.168.0.103"   // <-- sửa thành IP của PC
#define PC_SERVER_PORT  5050

// UDP nhận lệnh góc từ PC (tùy chọn, giữ trùng với Python)
#define ENABLE_UDP_CONTROL 1
#define UDP_LISTEN_PORT    8080
// =================================

// Camera model: AI Thinker (phổ biến)
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Stream settings
static const framesize_t FRAME_SIZE = FRAMESIZE_QVGA; // 320x240 (khớp Python)
static const int JPEG_QUALITY = 12;   // 10~20 (nhỏ = chất lượng cao, kích thước lớn)
static const int FB_COUNT     = 2;    // 2 khung: mượt hơn
static const uint8_t TARGET_FPS = 12; // gửi khoảng 12 FPS là hợp lý

WiFiClient client;
#if ENABLE_UDP_CONTROL
WiFiUDP udp;
#endif

void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count = FB_COUNT;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    config.frame_size = FRAME_SIZE;
    config.jpeg_quality = JPEG_QUALITY;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(2000);
    ESP.restart();
  }
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("Connecting WiFi: %s", WIFI_SSID);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++tries > 60) { // ~30s timeout
      Serial.println("\nWiFi timeout, reboot");
      ESP.restart();
    }
  }
  Serial.printf("\nWiFi OK: %s  IP=%s\n", WIFI_SSID, WiFi.localIP().toString().c_str());
}

bool connectServer() {
  Serial.printf("Connecting TCP to %s:%d ... ", PC_SERVER_IP, PC_SERVER_PORT);
  if (client.connect(PC_SERVER_IP, PC_SERVER_PORT)) {
    client.setNoDelay(true);
    Serial.println("OK");
    return true;
  } else {
    Serial.println("FAILED");
    return false;
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);
  initCamera();
  connectWiFi();
#if ENABLE_UDP_CONTROL
  udp.begin(UDP_LISTEN_PORT);
  Serial.printf("UDP listening on %u\n", UDP_LISTEN_PORT);
#endif
}

void loop() {
  static uint32_t lastFpsTick = 0;
  static uint32_t sentFrames = 0;
  static uint32_t lastSentMs = 0;
  const uint32_t frameInterval = 1000 / TARGET_FPS;

  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  if (!client.connected()) {
    if (!connectServer()) {
      delay(1000);
      return;
    }
  }

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(10);
    return;
  }

  // Ensure JPEG buffer
  uint8_t *jpg = fb->buf;
  size_t jpg_len = fb->len;
  if (fb->format != PIXFORMAT_JPEG) {
    uint8_t *new_jpg = nullptr;
    size_t new_len = 0;
    if (!frame2jpg(fb, JPEG_QUALITY, &new_jpg, &new_len)) {
      Serial.println("JPEG encode failed");
      esp_camera_fb_return(fb);
      delay(5);
      return;
    }
    esp_camera_fb_return(fb);
    jpg = new_jpg;
    jpg_len = new_len;
  } else {
    // use original buffer
  }

  // 4-byte little-endian header (length)
  uint8_t header[4];
  header[0] = (uint8_t)(jpg_len & 0xFF);
  header[1] = (uint8_t)((jpg_len >> 8) & 0xFF);
  header[2] = (uint8_t)((jpg_len >> 16) & 0xFF);
  header[3] = (uint8_t)((jpg_len >> 24) & 0xFF);

  // Send header + payload
  size_t w1 = client.write(header, 4);
  size_t w2 = client.write(jpg, jpg_len);
  if (fb->format != PIXFORMAT_JPEG) {
    // buffer allocated by frame2jpg
    free(jpg);
  } else {
    esp_camera_fb_return(fb);
  }

  if (w1 != 4 || w2 != jpg_len) {
    Serial.println("TCP write failed, reconnecting...");
    client.stop();
    delay(100);
    return;
  }

  sentFrames++;

  // Show FPS each 1s
  uint32_t now = millis();
  if (now - lastFpsTick >= 1000) {
    Serial.printf("stream fps=%u  last_size=%u bytes\n", sentFrames, (unsigned)jpg_len);
    sentFrames = 0;
    lastFpsTick = now;
  }

  // pacing
  uint32_t elapsed = now - lastSentMs;
  if (elapsed < frameInterval) {
    delay(frameInterval - elapsed);
  }
  lastSentMs = millis();

#if ENABLE_UDP_CONTROL
  // Non-blocking read UDP angle
  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    char buf[64];
    int len = udp.read(buf, sizeof(buf) - 1);
    if (len > 0) {
      buf[len] = 0;
      int angle = 0;
      if (sscanf(buf, "ANG:%d", &angle) == 1) {
        Serial.printf("UDP ANG received: %d\n", angle);
        // TODO: Điều khiển động cơ/servo của bạn ở đây (nếu ESP32-CAM gánh luôn).
      }
    }
  }
#endif
}
