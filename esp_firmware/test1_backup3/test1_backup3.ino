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
bool manualStopped = false;
bool aiEnabled = true;

// ============================================================================
// Y NGHIA QUY TRINH:
// 1. RUNNING  : phat xung PUL tren GPIO12, cho lon moi toi che IR.
// 2. Lon cham IR:
//      -> FEEDING : quay thanh chan 2 giai doan (nhanh 140 do dau, cham 40 do cuoi)
//                   de nhe nhang om lon dung tam, khong lam hat lui lon.
//      -> WAITING : dung bang chuyen (motorOff) va gui "TRIGGER_CAMERA".
// 3. PC (main.py + DRAEM) tra ket qua:
//      "OK" (Lon tot) :
//             -> RETURNING : Mo can gat nhanh ve 0 (SERVO_HOME), cho can rut han 250ms (motorOff).
//             -> OK_PASSING: Bat bang chuyen (motorOn) cho lon tot di qua.
//      "NG" (Lon loi) :
//             -> NG_HOME   : CUNG MO CAN GAT NHANH VE 0 (y het OK), cho can rut han 250ms (motorOff).
//             -> NG_PASSING: Bat bang chuyen dua lon NG chay NG_EJECT_DELAY_MS toi vi tri gat.
//             -> GATING    : Can gat vung manh 180 do danh vang lon NG, roi thu ve 0 do!
// ============================================================================

#define SLOW_FEED_ENABLE       1      // 1 = quay cham toi 180 do | 0 = chi code cu

#define SERVO_HOME             0      // vi tri mo / thu can (cho lon di qua)
#define SERVO_FEED             180    // vi tri gat lon vao vi tri chup / chan lon
#define SERVO_EJECT            180    // goc vung manh de danh vang lon NG

// ==================== CẤU HÌNH CHẶN LON CHO MG996R ====================
// MG996R lực rất mạnh (11 kg.cm) nên rút ngắn thời gian để servo gạt nhanh & dứt khoát:
// Giai đoạn 1: Vụt nhanh 0 -> 140 độ qua khoảng trống không chạm lon (200ms)
// Giai đoạn 2: Đi 40 độ cuối (140 -> 180 độ) để ôm lon vừa tới (250ms thay vì 900ms)
// Tổng thời gian chặn lon = 200ms + 250ms = 450ms (nhanh gấp 3 lần code cũ!)
#define FEED_FAST_ANGLE        140    // Goc quet nhanh qua khoang trong (do)
#define FEED_FAST_TIME_MS      200    // Thoi gian quet nhanh 140 do dau (ms)
#define FEED_SLOW_TIME_MS      250    // Thoi gian om 40 do cuoi (ms)
#define SLOW_FEED_TIME_MS      (FEED_FAST_TIME_MS + FEED_SLOW_TIME_MS) // 450ms

#define GATE_OPEN_SETTLE_MS    200    // thoi gian cho can gat rut han ve 0 truoc khi bat bang chuyen
#define SLOW_FEED_REFRESH      20     // cap nhat goc moi 20ms

// ==================== ĐỊNH THỜI GẠT LON NG (CHO MG996R) ====================
// Thoi gian bang chuyen dua lon NG chay tiep sau khi duoi lon roi IR de toi dung tam can gat (ms)
#define NG_EJECT_DELAY_MS      900  

#define NG_PASS_TIMEOUT_MS     15000  // neu lon khong roi IR thi dung bang chuyen
#define OK_PASS_TIMEOUT_MS     10000  // failsafe neu lon tot bi ket tren duong di
// MG996R cần ~400-450ms để vung hết hành trình 180 độ va chạm lon NG:
#define STRIKE_HOLD_MS         450    // cho servo vung toi goc danh lon (ms)
#define STRIKE_RETURN_MS       400    // thoi gian thu servo ve vi tri san sang (ms)
#define INSPECTION_FAILSAFE_MS 30000  // fallback neu PC mat ket noi

Servo servo;

// Goc servo hien tai (dung cho [DIAG])
int servoPos = SERVO_HOME;

void servoWrite(int angle) {
  if (!servo.attached()) {
    servo.attach(SERVO_PIN);
  }
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
    int a;

    // Chế độ chặn lon 2 giai đoạn: vụt nhanh qua khoảng trống, ôm cực êm 40 độ cuối
    if (slowFrom == SERVO_HOME && slowTo == SERVO_FEED) {
      if (elapsed < FEED_FAST_TIME_MS) {
        a = (int)(((long)FEED_FAST_ANGLE * (long)elapsed) / (long)FEED_FAST_TIME_MS);
      } else {
        unsigned long el2 = elapsed - FEED_FAST_TIME_MS;
        a = FEED_FAST_ANGLE + (int)(((long)(SERVO_FEED - FEED_FAST_ANGLE) * (long)el2) / (long)FEED_SLOW_TIME_MS);
      }
    } else {
      a = slowFrom + (int)(((long)(slowTo - slowFrom) * (long)elapsed) / (long)slowDuration);
    }

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
unsigned long irBlockStartTime = 0;
bool irMustClearFirst = false;
#define IR_DEBOUNCE_MS  150   // Chống nhiễu quang học & mối nối đai: phải chắn liên tục >= 150ms mới kích hoạt
unsigned long waitingStartTime = 0;
unsigned long ngPassStartTime = 0;
unsigned long okPassStartTime = 0;
unsigned long returningStartTime = 0;
unsigned long ngHomeStartTime = 0;

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
      irMustClearFirst = true;
      motorOn();
      sendMsg("STATE=RUNNING\n");
      break;

    case FEEDING:
      // IR bao vat can: dung xung PUL, servo quay 2 giai doan de nhe nhang om lon
      motorOff();
      startSlowSweep(SERVO_HOME, SERVO_FEED, SLOW_FEED_TIME_MS);
      sendMsg("STATE=FEEDING 2_STAGE\n");
      break;

    case WAITING:
      slowActive = false;
      motorOff();
      waitingStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=WAITING TRIGGER_CAMERA\n");
      break;

    case RETURNING:
      // Lon tot: mo can gat nhanh ve 0 do, cho can rut han 250ms truoc khi bat motor
      motorOff();
      slowActive = false;
      servoWrite(SERVO_HOME);
      returningStartTime = millis();
      sendMsg("STATE=RETURNING QUICK_OPEN\n");
      break;

    case OK_PASSING:
      // Can gat da o 0 do: bat bang chuyen cho lon tot di qua
      motorOn();
      okPassStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=OK_PASSING GATE_OPEN_MOTOR_ON\n");
      break;

    case NG_HOME:
      // Lon loi: CUNG MO CAN GAT NHANH VE 0 DO (y het lon OK), cho can rut han 250ms
      motorOff();
      slowActive = false;
      servoWrite(SERVO_HOME);
      ngHomeStartTime = millis();
      sendMsg("STATE=NG_HOME QUICK_OPEN\n");
      break;

    case NG_PASSING:
      // Can gat da o 0 do: bat bang chuyen dua lon toi vi tri gat
      motorOn();
      ngPassStartTime = millis();
      irReleaseTimer = digitalRead(IR_PIN) == OBSTACLE_STATE ? 0 : millis();
      sendMsg("STATE=NG_PASSING WAIT_IR_CLEAR\n");
      break;

    case NG_FAULT:
      motorOff();
      sendMsg("STATE=NG_FAULT IR_DID_NOT_CLEAR\n");
      break;

    case GATING:
      // Lon da toi dung vi tri gat: vung manh 180 do de danh vang lon loi
      slowActive = false;
      motorOff();
      gatingStep = 0;
      gatingTimer = 0;
      sendMsg("STATE=GATING\n");
      break;
  }
}

// Chuc nang NG -> quet nhanh 0 -> 180 -> 0 sau khi lon toi vi tri gat
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
    manualStopped = false;
    setState(RUNNING);
    return;
  }
  if (upper == "STOP") {
    manualStopped = true;
    motorOff();
    sendMsg("STATE=STOPPED MANUAL_STOP\n");
    return;
  }
  if (upper == "AI_ON") {
    aiEnabled = true;
    sendMsg("STATE=AI_ON\n");
    return;
  }
  if (upper == "AI_OFF") {
    aiEnabled = false;
    slowActive = false;
    servoWrite(SERVO_HOME);
    if (state == WAITING || state == FEEDING) {
      setState(RUNNING);
    }
    sendMsg("STATE=AI_OFF\n");
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
  if (upper.startsWith("SERVO ")) {
    int targetA = upper.substring(6).toInt();
    targetA = constrain(targetA, 0, 180);
    slowActive = false;
    servoWrite(targetA);
    sendMsg("SERVO_SET angle=%d\n", targetA);
    return;
  }
  if (upper == "DETACH" || upper == "RELAX") {
    slowActive = false;
    servo.detach();
    sendMsg("SERVO_DETACHED (RELAXED)\n");
    return;
  }

  if (state == WAITING) {
      if (upper == "OK") {
        setState(RETURNING);    // Lon tot: mo can gat nhanh ve 0, cho 250ms roi moi chay bang chuyen
      } else if (upper == "NG") {
        setState(NG_HOME);      // Lon loi: CUNG mo can gat nhanh ve 0, cho 250ms roi chay toi cho danh lon
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
      if (manualStopped) {
        motorOff();
        irBlockStartTime = 0;
        break;
      }
      // Neu tat AI: giu servo o HOME, cho lon di qua tu do, khong chan khong gat
      if (!aiEnabled) {
        motorOn();
        irBlockStartTime = 0;
        break;
      }
      // Bat buoc cam bien phai CLEAR (khong vat can) it nhat 1 lan truoc khi chap nhan don lon moi
      // (chong triet de tinh trang ket cam bien gay tu dong chan lien tuc trong vo han)
      if (irMustClearFirst) {
        if (ir != OBSTACLE_STATE) {
          irMustClearFirst = false; // Mat doc da thong thoang, san sang don lon moi
        }
        motorOn();
        irBlockStartTime = 0;
        break;
      }
      // Chống nhiễu quang học (Debounce):
      // Cảm biến phải bị chắn liên tục >= IR_DEBOUNCE_MS (150ms) mới kích hoạt,
      // triệt tiêu 100% xung nhiễu điện từ cuộn dây động cơ và vết xước đai băng tải.
      if (ir == OBSTACLE_STATE) {
        if (irBlockStartTime == 0) {
          irBlockStartTime = millis();
        } else if (millis() - irBlockStartTime >= IR_DEBOUNCE_MS) {
          irBlockStartTime = 0;
          irMustClearFirst = true; // Kich hoat xong phai cho mat doc thong thoang moi nhan tiep
#if SLOW_FEED_ENABLE
          setState(FEEDING);
#else
          setState(WAITING);
#endif
        }
      } else {
        irBlockStartTime = 0;
        motorOn();
      }
      break;

    case FEEDING:
      motorOff();
      if (updateSlowSweep()) {
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
        sendMsg("INSPECTION_TIMEOUT -> RETURNING\n");
        setState(RETURNING);
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
      // Cho 250ms cho can gat rut han ve 0 do (khong con cham lon)
      if (millis() - returningStartTime >= GATE_OPEN_SETTLE_MS) {
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

      // Khi mat doc IR da thong thoang lien tuc 600ms -> lon da di qua hoan toan!
      if (irReleaseTimer && millis() - irReleaseTimer >= 600) {
        sendMsg("OK_CAN_PASSED\n");
        setState(RUNNING);
      } else if (millis() - okPassStartTime >= OK_PASS_TIMEOUT_MS) {
        sendMsg("OK_PASS_TIMEOUT\n");
        setState(RUNNING);
      }
      break;

    case NG_HOME:
      motorOff();
      // Cho 250ms cho can gat rut han ve 0 do y het lon OK
      if (millis() - ngHomeStartTime >= GATE_OPEN_SETTLE_MS) {
        sendMsg("NG_GATE_OPEN_DONE\n");
        setState(NG_PASSING);
      }
      break;

    case NG_PASSING:
      if (ir == OBSTACLE_STATE) {
        irReleaseTimer = 0;
      } else if (!irReleaseTimer) {
        irReleaseTimer = millis();
      }

      // Khi duoi lon vua thoat khoi IR lien tuc NG_EJECT_DELAY_MS (580ms) -> lon da toi dung tam can gat
      if (irReleaseTimer && millis() - irReleaseTimer >= NG_EJECT_DELAY_MS) {
        motorOff();
        sendMsg("IR_CLEAR_REACHED -> EJECT\n");
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
  sendMsg("SERVO HOME=%d FEED=%d EJECT=%d SLOW=%dms NG_DELAY=%dms\n",
          SERVO_HOME, SERVO_FEED, SERVO_EJECT, SLOW_FEED_TIME_MS, NG_EJECT_DELAY_MS);
  sendMsg("SLOW_FEED_ENABLE=%d\n", SLOW_FEED_ENABLE);
  sendMsg("WIFI_IP=%s\n", wifiConnected ? WiFi.localIP().toString().c_str() : "DISCONNECTED");
}

bool wifiRetrying = false;
int wifiRetryCount = 0;
unsigned long wifiRetryStepTimer = 0;

// ====================== LOOP ======================
void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnected = false;
  }

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
