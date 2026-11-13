/*
  Alerta de sonolência — IBM3131 (Ibmec 2026.2)

  Recebe do computador, pela serial (USB CDC, 9600 bps, 8N1), um caractere:
    'N' = normal  -> LED verde
    'W' = atenção -> LED amarelo
    'A' = alarme  -> LED vermelho + buzzer intermitente
  O computador reenvia o estado a cada 1 s. Sem mensagens por 3 s
  (programa fechado ou cabo solto), tudo apaga e o amarelo pisca.

  Ligações (Arduino Uno):
    D5 -> resistor 220 ohm -> LED verde    -> GND
    D6 -> resistor 220 ohm -> LED amarelo  -> GND
    D7 -> resistor 220 ohm -> LED vermelho -> GND
    D8 -> buzzer ativo (+)   ;   buzzer (-) -> GND
*/

const uint8_t LED_VERDE = 5;
const uint8_t LED_AMARELO = 6;
const uint8_t LED_VERMELHO = 7;
const uint8_t BUZZER = 8;

const unsigned long TIMEOUT_MS = 3000;
const unsigned long PERIODO_BIP_MS = 250;

char estado = 'N';
unsigned long ultimaMensagem = 0;
unsigned long ultimoToggle = 0;
bool pulso = false;

void aplicar(char e) {
  digitalWrite(LED_VERDE, e == 'N');
  digitalWrite(LED_AMARELO, e == 'W');
  digitalWrite(LED_VERMELHO, e == 'A');
  if (e != 'A') digitalWrite(BUZZER, LOW);
}

void setup() {
  pinMode(LED_VERDE, OUTPUT);
  pinMode(LED_AMARELO, OUTPUT);
  pinMode(LED_VERMELHO, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  Serial.begin(9600);
  aplicar('N');
  ultimaMensagem = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'N' || c == 'W' || c == 'A') {
      ultimaMensagem = millis();
      if (c != estado) {
        estado = c;
        aplicar(estado);
      }
    }
  }

  unsigned long agora = millis();

  // Sem comunicação: apaga tudo e pisca o amarelo
  if (agora - ultimaMensagem > TIMEOUT_MS) {
    digitalWrite(LED_VERDE, LOW);
    digitalWrite(LED_VERMELHO, LOW);
    digitalWrite(BUZZER, LOW);
    if (agora - ultimoToggle > 500) {
      pulso = !pulso;
      digitalWrite(LED_AMARELO, pulso);
      ultimoToggle = agora;
    }
    estado = '?';
    return;
  }

  // Alarme: buzzer intermitente
  if (estado == 'A' && agora - ultimoToggle > PERIODO_BIP_MS) {
    pulso = !pulso;
    digitalWrite(BUZZER, pulso);
    ultimoToggle = agora;
  }
}
