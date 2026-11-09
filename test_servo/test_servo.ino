#include <Servo.h>

// Chân cắm Servo trên NodeMCU ESP8266: GPIO5 (D1)
#define SERVO_PIN 5 

Servo testServo;

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n======================================");
  Serial.println("   CHUONG TRINH TEST KIEM TRA SERVO   ");
  Serial.println("======================================");
  
  testServo.attach(SERVO_PIN);
}

void loop() {
  // Test 1: Vị trí góc nhỏ (15 độ)
  Serial.println("[1] Servo quay ve 15 do (HOME)...");
  testServo.write(15);
  delay(1500);

  // Test 2: Vị trí chính giữa (90 độ)
  Serial.println("[2] Servo quay ra 90 do (CHINH GIUA)...");
  testServo.write(90);
  delay(1500);

  // Test 3: Vị trí góc gạt an toàn (165 độ - khong ep kich 180 do)
  Serial.println("[3] Servo quay ra 165 do (VI TRI GAT)...");
  testServo.write(165);
  delay(1500);

  // Test 4: Quay về 90 độ
  Serial.println("[4] Servo quay lai 90 do...");
  testServo.write(90);
  delay(1500);
}

