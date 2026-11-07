#include <Arduino.h>
#include <Servo.h>
#include <ESP8266WiFi.h>
#include <ESP8266mDNS.h>
#include <stdarg.h>

// ====================== PIN ======================
// Servo thanh chắn         : GPIO5  (D1)
// Cảm biến hồng ngoại      : GPIO14 (D5)
// Driver STEP/PUL-           : GPIO12 (D6)
#define IR_PIN     14
#define PUL_PIN    12
#define SERVO_PIN  5

// Driver PUL- active-LOW: tao xung bang LOW ngan, sau do tha chan ve INPUT.
// Day la cach pulse cua sketch ban dau da chay tot; khong doi sang relay/HIGH.
#define OBSTACLE_STATE LOW
#define STEP_DELAY_US   1500

// ====================== WIFI ======================
const char* WIFI_SSID     = "Sara";
const char* WIFI_PASSWORD = "0382911493";

WiFiServer server(12345);
WiFiClient tcpClient;
bool wifiConnected = false;
unsigned long wifiRetryTimer = 0;

// ====================== DRIVER STEP/PUL ======================
volatile bool isRunning = false;

// ============================================================================
// Y NGHIA QUY TRINH:
// 1. RUNNING  : phat xung PUL tren GPIO12, cho lon moi toi che IR (lastIR!=LOW && ir==LOW).
// 2. Lon cham IR:
//      -> FEEDING : quay thanh chan 0 -> 180 do de gat lon vao vi tri chup.
//      -> WAITING : dung bang chuyen (motorOff) va gui "TRIGGER_CAMERA".
// 3. PC (main.py + DRAEM) tra ket qua:
//      "OK" (Lon tot) :
//             -> OK_PASSING: Mo servo ve 0 (SERVO_HOME), bat bang chuyen (motorOn)
//                            cho lon tot di qua. Bo qua cam bien trong luc lon
//                            dang luot qua IR; khi IR clear lien tuc 600ms
//                            (lon da thoat han) moi quay ve RUNNING!
//      "NG" (Lon loi) :
//             -> NG_HOME: Mo thanh chan ve 0 truoc de lon di toi vi tri gat.
//             -> NG_PASSING: Chay bang chuyen cho lon toi vi tri gat (IR clear 500ms).
//             -> GATING: Servo da nhanh 0 -> 180 -> 0 de loai bo lon loi!
// ============================================================================

#define SLOW_FEED_ENABLE    1      // 1 = quay cham toi 180 do | 0 = chi code cu

#define SERVO_HOME          0      // vi tri mo / cho ban dau (cho lon di qua)
#define SERVO_FEED          180    // vi tri gat lon vao vi tri chup / chan lon
#define SERVO_EJECT         180    // goc da nhanh toi da cua MG90S khi NG

#define SLOW_FEED_TIME_MS   3000   // tong thoi gian quay 180 do cham (ms)
#define SLOW_RETURN_TIME_MS 3000   // hoi servo cham khi ket qua OK (neu dung sweep)
#define SLOW_FEED_REFRESH   20     // cap nhat goc moi 20ms
#define NG_HOME_SETTLE_MS   300    // cho servo dung yen tai 0 truoc khi chay bang chuyen
#define NG_PASS_TIMEOUT_MS  15000  // neu lon khong roi IR thi dung bang chuyen
#define OK_PASS_TIMEOUT_MS  10000  // failsafe neu lon tot bi ket tren duong di
#define STRIKE_HOLD_MS      500    // cho servo toi goc day lon truoc khi hoi ve
#define STRIKE_RETURN_MS    1500   // thoi gian cho truoc khi bang chuyen chay lai
#define INSPECTION_FAILSAFE_MS 30000 // fallback neu PC mat ket noi

Servo servo;

// Goc servo hien tai (dung cho [DIAG])
int servoPos = SERVO_HOME;

void servoWrite(int angle) {
  servoPos = angle;
  servo.write(angle);
}

// -------- Quay cham (slow sweep) --------
bool slowActive = false;
int slowFrom = SERVO_HOME;
int slowTo = SERVO_FEED;
int slowAngle = SERVO_HOME;
unsigned long slowStart = 0;
unsigned long slowDuration = 0;
unsigned long slowRefresh = 0;

void sendMsg(const char* fmt, ...);   // forward declaration

void startSlowSweep(int from, int to, unsigned long duration) {
  slowFrom = from;
  slowTo = to;
  slowAngle = from;
  slowStart = millis();
  slowDuration = duration;
  slowRefresh = millis();
  slowActive = true;
  servoWrite(from);
}

// Tra ve true khi da quay du goc
bool updateSlowSweep() {
  if (!slowActive) return true;

  unsigned long now = millis();

  if (now - slowStart >= slowDuration) {
    servoWrite(slowTo);
    slowAngle = slowTo;
    slowActive = false;
    sendMsg("SLOW_FEED_DONE angle=%d\n", slowAngle);
    return true;
  }

  if (now - slowRefresh >= SLOW_FEED_REFRESH) {
    slowRefresh = now;
    unsigned long elapsed = now - slowStart;
    int a = slowFrom + (int)(((long)(slowTo - slowFrom) * (long)elapsed) / (long)slowDuration);
    if (a != slowAngle) {
      slowAngle = a;
      servoWrite(a);
    }
  }
  return false;
}

// ====================== TCP/SERIAL ======================
String cmdBuffer = "";
bool cmdReady = false;
String cmdSource = "";

// ====================== STATE MACHINE ======================
enum State { RUNNING, FEEDING, WAITING, RETURNING, OK_PASSING, NG_HOME, NG_PASSING, NG_FAULT, GATING };
State state = RUNNING;
int lastIR = HIGH;

int gatingStep = 0;
unsigned long gatingTimer = 0;
unsigned long lastDiagTime = 0;
unsigned long lastWaitNotifyTime = 0;
unsigned long irReleaseTimer = 0;
unsigned long waitingStartTime = 0;
unsigned long ngPassStartTime = 0;
unsigned long okPassStartTime = 0;

const char* STATE_STR[] = {
  "RUNNING", "FEEDING", "WAITING", "RETURNING", "OK_PASSING",
  "NG_HOME", "NG_PASSING", "NG_FAULT", "GATING"
};

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
    pinMode(PUL_PIN, INPUT);  // tha noi, trang thai nghi cua PUL-
    sendMsg("MOTOR_START\n");
  }
}

void motorOff() {
  if (isRunning) {
    isRunning = false;
    pinMode(PUL_PIN, INPUT);  // ngat xung, tha noi optocoupler
    sendMsg("MOTOR_STOP\n");
  }
}

void handleStepper() {
  if (!isRunning) {
    pinMode(PUL_PIN, INPUT);
    return;
  }

  // Giong sketch test ban dau: LOW de bat opto, INPUT de tha/no pulse.
  pinMode(PUL_PIN, OUTPUT);
  digitalWrite(PUL_PIN, LOW);
  delayMicroseconds(STEP_DELAY_US);
  pinMode(PUL_PIN, INPUT);
  delayMicroseconds(STEP_DELAY_US);
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
void setState(int nextState) {
  State s = static_cast<State>(nextState);
  state = s;
  switch (s) {
    case RUNNING:
      slowActive = false;
      servoWrite(SERVO_HOME);
      motorOn();
      sendMsg("STATE=RUNNING\n");
      break;

    case FEEDING:
      // Giu hanh vi code test: dung xung khi IR bao vat can; chi quay servo cham.
      motorOff();
      startSlowSweep(SERVO_HOME, SERVO_FEED, SLOW_FEED_TIME_MS);
      sendMsg("STATE=FEEDING SLOW_180\n");
      break;

    case WAITING:
      slowActive = false;
      motorOff();
      waitingStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=WAITING TRIGGER_CAMERA\n");
      break;

    case RETURNING:
      motorOff();
      startSlowSweep(SERVO_FEED, SERVO_HOME, SLOW_RETURN_TIME_MS);
      sendMsg("STATE=RETURNING SLOW_HOME\n");
      break;

    case OK_PASSING:
      // Lon tot: mo thanh chan ve HOME (0 do) ngay va bat bang chuyen de lon di qua hoan toan.
      slowActive = false;
      servoWrite(SERVO_HOME);
      motorOn();
      okPassStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=OK_PASSING GATE_OPEN_MOTOR_ON\n");
      break;

    case NG_HOME:
      // Ket qua NG: hoi thanh gat cham ve 0 truoc de mo duong cho lon di qua vi tri gat.
      motorOff();
      startSlowSweep(SERVO_FEED, SERVO_HOME, SLOW_RETURN_TIME_MS);
      sendMsg("STATE=NG_HOME SLOW\n");
      break;

    case NG_PASSING:
      // Servo da o 0; cho bang chuyen dua lon toi vi tri gat.
      motorOn();
      ngPassStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=NG_PASSING WAIT_IR_CLEAR_500MS\n");
      break;

    case NG_FAULT:
      motorOff();
      sendMsg("STATE=NG_FAULT IR_DID_NOT_CLEAR\n");
      break;

    case GATING:
      // Lon da roi IR 500ms; thuc hien cu gat nhanh 0 -> 180 -> 0 de loai bo lon.
      slowActive = false;
      motorOff();
      gatingStep = 0;
      gatingTimer = 0;
      sendMsg("STATE=GATING\n");
      break;
  }
}

// Chuc nang NG -> quet nhanh 0 -> 180 -> 0 sau khi IR clear lien tuc 500ms.
void runGating() {
  switch (gatingStep) {
    case 0:
      servoWrite(SERVO_EJECT);
      gatingTimer = millis();
      gatingStep = 1;
      sendMsg("GAT_ACTIVE EJECT=%d\n", SERVO_EJECT);
      break;
    case 1:
      if (millis() - gatingTimer >= STRIKE_HOLD_MS) {
        servoWrite(SERVO_HOME);
        gatingTimer = millis();
        gatingStep = 2;
      }
      break;
    case 2:
      if (millis() - gatingTimer >= STRIKE_RETURN_MS) {
        sendMsg("GAT_DONE\n");
        setState(RUNNING);
      }
      break;
  }
}

void handleCmd(String cmd) {
  sendMsg("[CMD] from=%s cmd=%s state=%s run=%d\n",
          cmdSource.c_str(), cmd.c_str(), STATE_STR[state], isRunning);

  String upper = cmd;
  upper.toUpperCase();

  // Lenh dieu khien tu Web hoac Serial
  if (upper == "START") {
    setState(RUNNING);
    return;
  }
  if (upper == "STOP") {
    motorOff();
    return;
  }

  // Lenh test tay (Serial hoac TCP)
  if (upper == "FEED")   { setState(FEEDING); return; }   // quay cham toi 180 do
  if (upper == "STRIKE") { setState(GATING);  return; }   // danh lon toc do cao
  if (upper == "HOME")   {
    slowActive = false;
    servoWrite(SERVO_HOME);
    sendMsg("SERVO_HOME\n");
    return;
  }

  if (state == WAITING) {
      if (upper == "OK") {
        setState(OK_PASSING);   // Lon tot: mo thanh chan, cho chay qua hoan toan
      } else if (upper == "NG") {
        setState(NG_HOME);      // Lon loi: hoi ve 0 de dua lon toi goc gat
      }
  }

  if (upper == "RESET" && state == NG_FAULT) {
    setState(RUNNING);
    return;
  }
}

void updateState() {
  int ir = digitalRead(IR_PIN);

  switch (state) {
    case RUNNING:
      // Chi kich hoat khi CÓ LON MỚI ĐẾN (chuyen tu khong can sang co can)
      if (lastIR != OBSTACLE_STATE && ir == OBSTACLE_STATE) {
#if SLOW_FEED_ENABLE
        // MOI: lon cham hong ngoai -> quay cham toi 180 do de dua lon vao vi tri
        setState(FEEDING);
#else
        // CODE CU: dung bang chuyen + chup anh ngay, servo khong quay cham
        setState(WAITING);
#endif
      } else {
        motorOn();
      }
      break;

    case FEEDING:
      // IR vua phat hien lon: dung PUL, servo quay cham trong khi bang chuyen dung
      motorOff();
      if (updateSlowSweep()) {
        // Da quay du 180 do -> dung bang chuyen, cho AI
        setState(WAITING);
      }
      break;

    case WAITING:
      if (lastIR == OBSTACLE_STATE && ir != OBSTACLE_STATE) {
        irReleaseTimer = millis();
      } else if (ir == OBSTACLE_STATE) {
        irReleaseTimer = 0;
      }
      // Neu PC mat ket noi, cho IR clear 500ms va qua timeout roi cho lon di qua
      if (irReleaseTimer && millis() - irReleaseTimer >= 500 &&
          millis() - waitingStartTime >= INSPECTION_FAILSAFE_MS) {
        irReleaseTimer = 0;
        sendMsg("INSPECTION_TIMEOUT -> OK_PASSING\n");
        setState(OK_PASSING);
      } else {
        motorOff();
        if (millis() - lastWaitNotifyTime >= 5000) {
          lastWaitNotifyTime = millis();
          sendMsg("WAITING_INSPECTION\n");
        }
      }
      break;

    case RETURNING:
      motorOff();
      if (updateSlowSweep()) {
        sendMsg("SERVO_HOME_DONE\n");
        setState(OK_PASSING);
      }
      break;

    case OK_PASSING:
      // Lon tot: bang chuyen chay lien tuc, bo qua cam bien IR trong luc than lon dang luot qua
      motorOn();
      if (ir == OBSTACLE_STATE) {
        irReleaseTimer = 0;  // Than lon van dang nam tren cam bien
      } else if (!irReleaseTimer) {
        irReleaseTimer = millis();  // Bat dau thay duoi lon thoat khoi mat cam bien
      }

      // Khi mat doc IR da thong thoang lien tuc 800ms -> lon da di qua hoan toan!
      if (irReleaseTimer && millis() - irReleaseTimer >= 800) {
        sendMsg("OK_CAN_PASSED\n");
        setState(RUNNING);
      } else if (millis() - okPassStartTime >= OK_PASS_TIMEOUT_MS) {
        sendMsg("OK_PASS_TIMEOUT\n");
        setState(RUNNING);
      }
      break;

    case NG_HOME:
      motorOff();
      if (updateSlowSweep()) {
        // Cho servo o home mot chut roi moi bat bang chuyen.
        if (millis() - slowStart >= SLOW_RETURN_TIME_MS + NG_HOME_SETTLE_MS) {
          setState(NG_PASSING);
        }
      }
      break;

    case NG_PASSING:
      if (ir == OBSTACLE_STATE) {
        irReleaseTimer = 0;
      } else if (!irReleaseTimer) {
        irReleaseTimer = millis();
      }

      if (irReleaseTimer && millis() - irReleaseTimer >= 500) {
        motorOff();
        sendMsg("IR_CLEAR_500MS -> EJECT\n");
        setState(GATING);
      } else if (millis() - ngPassStartTime >= NG_PASS_TIMEOUT_MS) {
        motorOff();
        setState(NG_FAULT);
      } else {
        motorOn();
      }
      break;

    case NG_FAULT:
      motorOff();
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

  pinMode(PUL_PIN, INPUT);

  servo.attach(SERVO_PIN);
  servoWrite(SERVO_HOME);

  pinMode(IR_PIN, INPUT_PULLUP);

  lastIR = digitalRead(IR_PIN);
  state = RUNNING;
  isRunning = false;
  motorOn();

  sendMsg("ESP_READY STATE=RUNNING\n");
  sendMsg("PIN SERVO=%d IR=%d PUL=%d\n", SERVO_PIN, IR_PIN, PUL_PIN);
  sendMsg("PUL MODE=ACTIVE_LOW STEP_DELAY_US=%d\n", STEP_DELAY_US);
  sendMsg("SERVO HOME=%d FEED=%d EJECT=%d SLOW=%dms\n",
          SERVO_HOME, SERVO_FEED, SERVO_EJECT, SLOW_FEED_TIME_MS);
  sendMsg("SLOW_FEED_ENABLE=%d (1=quay 90 cham, 0=chi code cu)\n", SLOW_FEED_ENABLE);
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

  readTcp();
  readSerial();

  if (cmdReady) {
    cmdBuffer.trim();
    handleCmd(cmdBuffer);
    cmdBuffer = "";
    cmdReady = false;
  }

  updateState();
  // Doc IR va cap nhat state truoc khi phat xung, tranh them xung khi IR vua bi che.
  handleStepper();

  if (millis() - lastDiagTime >= 5000) {
    lastDiagTime = millis();
    sendMsg("[DIAG] %s IR=%s MOTOR=%d SERVO=%d WIFI=%s\n",
            STATE_STR[state],
            digitalRead(IR_PIN) == HIGH ? "CLEAR" : "BLOCK",
            isRunning,
            servoPos,
            wifiConnected ? WiFi.localIP().toString().c_str() : "NO");
  }

  if (wifiConnected) MDNS.update();
}
