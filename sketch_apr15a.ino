#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>

const char* ssid = "ESP8266_AccessPoint";
const char* password = "123456789";

// Motor pins
const int IN1 = D0;
const int IN2 = D1;
const int IN3 = D2;
const int IN4 = D3;

ESP8266WebServer server(80);

// Movement state
bool motorActive = false;
unsigned long motorStartTime = 0;
unsigned long motorDuration = 2000;  // 2 seconds
String currentAction = "";

void stopMotors() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  motorActive = false;
  currentAction = "";
}

void moveForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveBackward() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void moveLeft() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void moveRight() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void startMove(String direction) {
  stopMotors(); // Safety

  if (direction == "forward") moveForward();
  else if (direction == "backward") moveBackward();
  else if (direction == "left") moveLeft();
  else if (direction == "right") moveRight();

  motorStartTime = millis();
  motorActive = true;
  currentAction = direction;
}

void setup() {
  Serial.begin(115200);

  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  stopMotors();

  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");
  Serial.println(WiFi.softAPIP());

  server.on("/", HTTP_GET, []() {
    String html = R"====(
      <html><head><title>Motor Control</title>
      <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; background: #f0f0f0; }
        h1 { color: #333; }
        .btn {
          padding: 20px 40px; margin: 10px;
          font-size: 24px; background: #4CAF50; color: white;
          border: none; border-radius: 8px; cursor: pointer;
          transition: background 0.3s;
        }
        .btn:hover { background: #45a049; }
      </style></head><body>
      <h1>Motor Control Panel</h1>
      <form action="/w"><button class="btn">W</button></form>
      <form action="/s"><button class="btn">S</button></form>
      <form action="/a"><button class="btn">A</button></form>
      <form action="/d"><button class="btn">D</button></form>
      </body></html>
    )====";
    server.send(200, "text/html", html);
  });

  server.on("/w", HTTP_GET, []() {
    startMove("forward");
    server.sendHeader("Location", "/");
    server.send(303);
  });

  server.on("/s", HTTP_GET, []() {
    startMove("backward");
    server.sendHeader("Location", "/");
    server.send(303);
  });

  server.on("/a", HTTP_GET, []() {
    startMove("left");
    server.sendHeader("Location", "/");
    server.send(303);
  });

  server.on("/d", HTTP_GET, []() {
    startMove("right");
    server.sendHeader("Location", "/");
    server.send(303);
  });

  server.begin();
}

void loop() {
  server.handleClient();

  // Stop after 2 seconds
  if (motorActive && millis() - motorStartTime >= motorDuration) {
    stopMotors();
  }
}