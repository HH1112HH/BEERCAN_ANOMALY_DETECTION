#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

const int RELAY_PIN = 2;

WebServer server(80);

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("ESP IP: ");
  Serial.println(WiFi.localIP());

  server.on("/detect", handleDetect);
  server.on("/health", handleHealth);

  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
}

void handleDetect() {
  if (!server.hasArg("status")) {
    server.send(400, "text/plain", "Missing status parameter");
    return;
  }

  String status = server.arg("status");

  if (status == "good") {
    digitalWrite(RELAY_PIN, HIGH);
    server.send(200, "text/plain", "OK - Conveyor RUNNING");
    Serial.println("GOOD -> Conveyor RUNNING");
  } else if (status == "ng") {
    digitalWrite(RELAY_PIN, LOW);
    server.send(200, "text/plain", "OK - Conveyor STOPPED");
    Serial.println("NG -> Conveyor STOPPED");
  } else {
    server.send(400, "text/plain", "Invalid status. Use 'good' or 'ng'");
  }
}

void handleHealth() {
  String state = digitalRead(RELAY_PIN) == HIGH ? "running" : "stopped";
  String json = "{\"status\":\"ok\",\"conveyor\":\"" + state + "\",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  server.send(200, "application/json", json);
}
