#include <Arduino.h>
#include <Servo.h>
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <stdarg.h>

// ====================== PIN ======================
#define IR_PIN     14
#define PUL_PIN    5
#define SERVO_PIN  4

// ====================== WIFI ======================
const char* WIFI_SSID     = "CHOCO_2.4G";
const char* WIFI_PASSWORD = "chococamon";

WiFiServer server(12345);
WiFiClient tcpClient;
bool wifiConnected = false;
unsigned long wifiRetryTimer = 0;

// ====================== STEPPER ======================
volatile bool isRunning = false;
unsigned long stepInterval = 1000;
unsigned long lastStepTime = 0;
bool pulseState = false;

// ====================== SERVO ======================
Servo servo;

// ====================== TCP/SERIAL ======================
String cmdBuffer = "";
bool cmdReady = false;
String cmdSource = "";

// ====================== STATE MACHINE ======================
enum State { RUNNING, WAITING, GATING };
State state = RUNNING;
int lastIR = HIGH;

int gatingStep = 0;
unsigned long gatingTimer = 0;
unsigned long lastDiagTime = 0;
unsigned long lastWaitNotifyTime = 0;
unsigned long irReleaseTimer = 0;

const char* STATE_STR[] = { "RUNNING", "WAITING", "GATING" };

// ====================== TCP/SERIAL OUTPUT ======================
void sendMsg(const char* fmt, ...) {
  char buf[256];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  Serial.print(buf);
  if (tcpClient && tcpClient.connected()) {
    tcpClient.print(buf);
  }
}

// ====================== MOTOR ======================
void motorOn() {
  if (!isRunning) {
    isRunning = true;
    sendMsg("MOTOR_START\n");
  }
}

void motorOff() {
  if (isRunning) {
    isRunning = false;
    digitalWrite(PUL_PIN, LOW);
    sendMsg("MOTOR_STOP\n");
  }
}

void handleStepper() {
  if (!isRunning) return;
  unsigned long currentMicros = micros();
  if (currentMicros - lastStepTime >= stepInterval) {
    lastStepTime = currentMicros;
    pulseState = !pulseState;
    digitalWrite(PUL_PIN, pulseState);
  }
}

// ====================== COMMAND INPUT ======================
void readTcp() {
  if (!tcpClient || !tcpClient.connected()) {
    if (tcpClient) tcpClient.stop();
    tcpClient = server.available();
    return;
  }
  while (tcpClient.available()) {
    char c = (char)tcpClient.read();
    if (c == '\n') { cmdReady = true; cmdSource = "TCP"; break; }
    if (c != '\r') cmdBuffer += c;
  }
}

void readSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { cmdReady = true; cmdSource = "Serial"; break; }
    if (c != '\r') cmdBuffer += c;
  }
}

// ====================== STATE MACHINE ======================
void setState(State s) {
  state = s;
  switch (s) {
    case RUNNING:
      motorOn();
      sendMsg("STATE=RUNNING\n");
      break;
    case WAITING:
      motorOff();
      irReleaseTimer = 0;
      sendMsg("STATE=WAITING TRIGGER_CAMERA\n");
      break;
    case GATING:
      motorOff();
      gatingStep = 0;
      gatingTimer = 0;
      sendMsg("STATE=GATING\n");
      break;
  }
}

void runGating() {
  switch (gatingStep) {
    case 0:
      servo.write(120);
      gatingTimer = millis();
      gatingStep = 1;
      sendMsg("GAT_ACTIVE\n");
      break;
    case 1:
      if (millis() - gatingTimer >= 200) {
        servo.write(0);
        gatingTimer = millis();
        gatingStep = 2;
      }
      break;
    case 2:
      if (millis() - gatingTimer >= 1500) {
        sendMsg("GAT_DONE\n");
        setState(RUNNING);
      }
      break;
  }
}

void handleCmd(String cmd) {
  sendMsg("[CMD] from=%s cmd=%s state=%s run=%d\n",
          cmdSource.c_str(), cmd.c_str(), STATE_STR[state], isRunning);

  if (state == WAITING) {
    if (cmd == "OK")       setState(RUNNING);
    else if (cmd == "NG")  setState(GATING);
  }
}

void updateState() {
  int ir = digitalRead(IR_PIN);

  switch (state) {
    case RUNNING:
      motorOn();
      if (lastIR == HIGH && ir == LOW) {
        setState(WAITING);
      }
      break;

    case WAITING:
      if (lastIR == LOW && ir == HIGH) {
        irReleaseTimer = millis();
      }
      if (irReleaseTimer && millis() - irReleaseTimer >= 500) {
        irReleaseTimer = 0;
        setState(RUNNING);
      } else if (!irReleaseTimer) {
        motorOff();
        if (millis() - lastWaitNotifyTime >= 1000) {
          lastWaitNotifyTime = millis();
          sendMsg("WAITING_INSPECTION\n");
        }
      }
      break;

    case GATING:
      runGating();
      break;
  }
  lastIR = ir;
}

// ====================== WIFI ======================
void tryConnectWiFi() {
  Serial.print("WiFi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" OK");
    Serial.println(WiFi.localIP());
    wifiConnected = true;
    MDNS.begin("esp8266-conveyor");
  } else {
    Serial.println(" FAIL");
  }
}

// ====================== SETUP ======================
void setup() {
  Serial.begin(115200);

  tryConnectWiFi();

  server.begin();
  Serial.println("TCP :12345");

  pinMode(PUL_PIN, OUTPUT);
  digitalWrite(PUL_PIN, LOW);

  servo.attach(SERVO_PIN);
  servo.write(0);

  pinMode(IR_PIN, INPUT);

  lastIR = digitalRead(IR_PIN);
  state = RUNNING;
  isRunning = true;

  sendMsg("ESP_READY STATE=RUNNING\n");
  sendMsg("WIFI_IP=%s\n", wifiConnected ? WiFi.localIP().toString().c_str() : "DISCONNECTED");
}

bool wifiRetrying = false;
int wifiRetryCount = 0;
unsigned long wifiRetryStepTimer = 0;

// ====================== LOOP ======================
void loop() {
  if (!wifiConnected) {
    if (!wifiRetrying && millis() - wifiRetryTimer >= 30000) {
      wifiRetryTimer = millis();
      wifiRetrying = true;
      wifiRetryCount = 0;
      wifiRetryStepTimer = 0;
      Serial.print("WiFi retry...");
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    }
    if (wifiRetrying) {
      if (wifiRetryCount < 20) {
        if (millis() - wifiRetryStepTimer >= 500) {
          wifiRetryStepTimer = millis();
          if (WiFi.status() == WL_CONNECTED) {
            Serial.println(" OK");
            Serial.println(WiFi.localIP());
            wifiConnected = true;
            wifiRetrying = false;
            MDNS.begin("esp8266-conveyor");
          } else {
            Serial.print(".");
            wifiRetryCount++;
          }
        }
      } else {
        Serial.println(" FAIL");
        wifiRetrying = false;
      }
    }
  }

  handleStepper();
  readTcp();
  readSerial();

  if (cmdReady) {
    cmdBuffer.trim();
    handleCmd(cmdBuffer);
    cmdBuffer = "";
    cmdReady = false;
  }

  updateState();

  if (millis() - lastDiagTime >= 5000) {
    lastDiagTime = millis();
    sendMsg("[DIAG] %s IR=%s MOTOR=%d WIFI=%s\n",
            STATE_STR[state],
            digitalRead(IR_PIN) == HIGH ? "CLEAR" : "BLOCK",
            isRunning,
            wifiConnected ? WiFi.localIP().toString().c_str() : "NO");
  }

  if (wifiConnected) MDNS.update();
}
