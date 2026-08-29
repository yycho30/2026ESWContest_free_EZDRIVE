#include <SoftwareSerial.h>

// 1. 핀 맵핑 정의
const int MOTOR_PIN = 5;     // 2N7000 게이트(Gate) 제어 핀 (PWM 지원)
const int BT_RX = 10;        // 프로미니 D10 (HM-10의 TXD와 연결)
const int BT_TX = 11;        // 프로미니 D11 (HM-10의 RXD와 연결)

// 상태별 진동 세기 설정 (0~255)
// 라즈베리파이가 판단하는 상태: 0=정상, 1=전방주시태만(약), 2=졸음(중), 3=수면(강)
const int VIB_LEVEL_1 = 130;  // 1단계: 약한 진동
const int VIB_LEVEL_2 = 190;  // 2단계: 중간 진동
const int VIB_LEVEL_3 = 255;  // 3단계: 강한 진동

// 진동 지속 시간 (ms)
const unsigned long VIB_DURATION = 1000; // 1초

// 블루투스 통신 객체 (마스터 HM-10 <-> 슬레이브 HM-10 BLE 브릿지를 통해 라즈베리파이와 통신)
SoftwareSerial btSerial(BT_RX, BT_TX);

// 진동 제어용 비차단 타이머/상태
bool motorActive = false;          // 현재 진동 중인지 여부
unsigned long motorStartTime = 0;  // 진동 시작 시각

void setup() {
  // 하드웨어 시리얼 (PC 디버깅용)
  Serial.begin(9600);

  // 소프트웨어 시리얼 (블루투스 통신용)
  btSerial.begin(9600);

  // 진동 모터 핀 초기화 - 부팅 시 반드시 OFF로 고정
  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);
  analogWrite(MOTOR_PIN, 0);
}

void loop() {
  // --- 1. 수신 파트: 진동 명령 처리 ---
  // 라즈베리파이가 판단 결과에 따라 '0'~'3' 명령을 보냅니다.
  //   '1' : 전방주시태만 -> 약한 진동 1초
  //   '2' : 졸음         -> 중간 진동 1초
  //   '3' : 수면         -> 강한 진동 1초
  //   '0' : 정상, 또는 택트 경보해제 버튼 눌림 -> 즉시 진동 정지
  while (btSerial.available()) {
    char cmd = btSerial.read();

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
        // 알 수 없는 바이트는 무시 (안전을 위해 현재 상태 유지)
        break;
    }
  }

  // --- 2. 진동 자동 정지 파트 ---
  if (motorActive && (millis() - motorStartTime >= VIB_DURATION)) {
    stopVibration();
  }
}

// 지정된 세기로 진동을 시작(또는 새 명령으로 갱신)하고 1초 타이머를 리셋합니다.
void startVibration(int level) {
  analogWrite(MOTOR_PIN, level);
  motorActive = true;
  motorStartTime = millis();
}

// 진동을 즉시 정지합니다.
void stopVibration() {
  analogWrite(MOTOR_PIN, 0);
  motorActive = false;
}
