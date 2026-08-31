#include <SoftwareSerial.h>

// ===== Pin mapping =====
const int MOTOR_PIN = 5;     // 2N7000 gate control (PWM)
const int BT_RX = 10;        // Pro Mini D10 (connected to HM-10 TXD)
const int BT_TX = 11;        // Pro Mini D11 (connected to HM-10 RXD)
const int IR_PIN = A0;       // IR sensor (analog, eyelid distance)

// Vibration strength per state (0-255)
// State from the Raspberry Pi: 0=normal, 1=inattention (weak), 2=drowsy (medium), 3=asleep (strong)
const int VIB_LEVEL_1 = 130;
const int VIB_LEVEL_2 = 190;
const int VIB_LEVEL_3 = 255;

const unsigned long VIB_DURATION = 1000;   // vibration pulse length (ms)
const unsigned long IR_SEND_INTERVAL = 100; // IR send period (ms) - matches the 10 Hz training data rate

// Single BLE link (master HM-10 <-> slave HM-10) carries both directions:
//   Pi -> Pro Mini : vibration state command ('0'-'3')
//   Pro Mini -> Pi : IR sensor value ("IR value: NNN")
SoftwareSerial btSerial(BT_RX, BT_TX);

bool motorActive = false;
unsigned long motorStartTime = 0;
unsigned long lastIrSendTime = 0;

void setup() {
  Serial.begin(9600);       // hardware serial, USB debugging only
  btSerial.begin(9600);     // BLE link to the Raspberry Pi

  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  analogWrite(MOTOR_PIN, 0);

  pinMode(IR_PIN, INPUT);

  Serial.println("READY");
}

void loop() {
  // --- 1. Receive: vibration commands from the Pi ---
  //   '1' inattention -> weak vibration for 1 s
  //   '2' drowsy       -> medium vibration for 1 s
  //   '3' asleep       -> strong vibration for 1 s
  //   '0' normal, or the dismiss button was pressed -> stop immediately
  while (btSerial.available()) {
    char cmd = btSerial.read();
    Serial.println(cmd);   // USB debug echo of what was received

    switch (cmd) {
      case '1':
        startVibration(VIB_LEVEL_1);
        break;
      case '2':
        startVibration(VIB_LEVEL_2);
        break;
      case '3':
        startVibration(VIB_LEVEL_3);
        break;
      case '0':
        stopVibration();
        break;
      default:
        // Unknown byte (e.g. noise, a partial line) - ignore, keep current state
        break;
    }
  }

  // --- 2. Send: IR sensor value to the Pi, every IR_SEND_INTERVAL ms ---
  unsigned long now = millis();
  if (now - lastIrSendTime >= IR_SEND_INTERVAL) {
    lastIrSendTime = now;
    sendIrValue();
  }

  // --- 3. Auto-stop the vibration after VIB_DURATION ---
  if (motorActive && (now - motorStartTime >= VIB_DURATION)) {
    stopVibration();
  }
}

// Reads the IR sensor and sends it over BLE as "IR value: NNN\n",
// matching what integrated_main.py's parse_ir() expects.
void sendIrValue() {
  int irValue = analogRead(IR_PIN);
  btSerial.print("IR value: ");
  btSerial.println(irValue);
}

// Starts (or refreshes) vibration at the given strength and resets the 1 s timer.
void startVibration(int level) {
  analogWrite(MOTOR_PIN, level);
  motorActive = true;
  motorStartTime = millis();
}

// Stops vibration immediately.
void stopVibration() {
  analogWrite(MOTOR_PIN, 0);
  motorActive = false;
}
