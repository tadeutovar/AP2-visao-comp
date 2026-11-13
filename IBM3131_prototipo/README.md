# Detecção de sonolência em motoristas por visão computacional

Projeto AP2 de **IBM3131 – Visão Computacional** (Ibmec RJ, 2026.2). Autor: Tadeu Tovar.

Uma webcam comum observa o rosto do motorista. O sistema mede a abertura dos olhos, a abertura da boca e a inclinação da cabeça e decide, em tempo real, se o motorista está alerta, com sinais de sonolência ou cochilando. O alerta aparece na tela, com som no computador e, opcionalmente, num Arduino com LEDs e buzzer.

```
Câmera USB ──USB/UVC──► OpenCV ──► MediaPipe Face Landmarker (478 marcos, blendshapes, pose)
                                   ──► indicadores: EAR · MAR · inclinação  (ou blendshapes)
                                   ──► calibração individual + máquina de estados
                                   ──► tela / bip  ──UART 9600 8N1──► Arduino ──GPIO──► LEDs + buzzer
```

---

## 1. Software utilizado (não foi desenvolvido neste trabalho)

| Software | Versão testada | Papel no sistema |
|---|---|---|
| Python | 3.13 | Linguagem do protótipo |
| [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Google) | 1.2.2 | Modelo pré-treinado: 478 marcos faciais 3D, 52 *blendshapes* e matriz de pose da cabeça |
| OpenCV (`opencv-contrib-python`) | 5.0 | Captura da câmera, desenho na tela, leitura e gravação de vídeo |
| NumPy | 2.x | Cálculo vetorial |
| Matplotlib | 3.x | Gráficos da avaliação |
| pyserial | 3.5 | Comunicação serial com o Arduino |
| Arduino IDE / ATmega328P | — | Firmware do alerta físico |
| pytest | 8.x | Testes automatizados |

O MediaPipe **só entrega coordenadas e escores**: ele não sabe o que é sonolência. Toda a parte de decisão abaixo foi implementada neste trabalho.

## 2. O que foi implementado neste trabalho

| Arquivo | O que faz |
|---|---|
| **`sonolencia/logica.py`** | Núcleo do método: EAR, MAR, ângulos da cabeça, calibração individual, PERCLOS, contagem de bocejos e máquina de estados (NORMAL / ATENÇÃO / ALARME / SEM ROSTO) |
| **`detector_sonolencia.py`** | Aplicação em tempo real: câmera, interface, log CSV por quadro, fotos automáticas a cada alarme, gravação do vídeo, modo vídeo (`--video`), bip e saída serial; dois métodos (`--metodo geometrico` ou `blendshapes`) |
| **`experimento.py`** | Experimento guiado: conduz um roteiro com instruções e bipes e grava o **gabarito automaticamente**; registra os indicadores dos dois métodos sobre o mesmo quadro |
| **`avaliar_experimento.py`** | Avaliação quantitativa: reproduz a máquina de estados para cada método e calcula VP, FN, FP, precisão, revocação, F1, atraso, FPS e latência; gera gráfico e tabela LaTeX |
| `analisar_sessao.py` | Resumo de sessões livres do detector |
| `arduino/alerta_sonolencia/alerta_sonolencia.ino` | Firmware: protocolo de 1 caractere (`N`/`W`/`A`), *timeout* de 3 s, buzzer intermitente |
| `tests/test_logica.py` | 12 testes da lógica com sinais simulados (piscadas, microcochilo, PERCLOS, bocejo, fala, cabeça, ausência de rosto) |

## 3. Customizações sobre o software padrão

O que precisou ser desenvolvido ou adaptado para transformar a saída genérica do MediaPipe e do OpenCV num detector de sonolência:

1. **EAR na malha do MediaPipe.** A EAR (Soukupová e Čech, 2016) foi definida para os 68 pontos do Dlib. Os 6 pontos de cada olho foram mapeados para os índices da malha de 478 pontos (olho direito 33, 160, 158, 133, 153, 144; esquerdo 362, 385, 387, 263, 373, 380).
2. **MAR com lábios internos.** A abertura da boca é a média de 3 distâncias verticais (82–87, 13–14, 312–317) dividida pela largura interna (78–308), o que separa bocejo de fala melhor que o contorno externo.
3. **Limiar calibrado por pessoa.** Nos primeiros 5 s, o sistema mede a EAR de olhos abertos e usa 75% da mediana como limiar (limitado a 0,15–0,30), em vez dos limiares fixos da literatura (0,2 ou 0,25). A pose neutra da cabeça é calibrada no mesmo intervalo.
4. **Pose da cabeça.** A matriz de transformação facial 4×4 do MediaPipe é convertida em ângulos de Euler; o alarme usa o desvio de arfagem (*pitch*) em relação à pose calibrada.
5. **Filtros temporais e estados.** Olhos fechados por ≥ 1,5 s ou cabeça desviada > 20° por ≥ 1,5 s geram ALARME, mantido por mais 2 s. PERCLOS ≥ 15% em 60 s ou 3 bocejos (boca aberta ≥ 1,5 s) em 5 min geram ATENÇÃO. Piscadas e fala ficam abaixo desses tempos.
6. **Segundo método com os *blendshapes*.** Os escores `eyeBlinkLeft/Right` e `jawOpen` do próprio MediaPipe alimentam a mesma máquina de estados, para comparar os métodos de forma justa sobre os mesmos quadros.
7. **Correção para Windows.** MediaPipe e OpenCV não abrem caminhos com acento (ex.: `Área de Trabalho`). O modelo passou a ser lido em memória (`model_asset_buffer`), as fotos são gravadas via `cv2.imencode` e o vídeo é gravado numa pasta temporária e movido no final.
8. **Gravação reprocessável.** O vídeo é salvo sem as anotações da tela e no FPS real medido da câmera, para ser reprocessado depois com `--video`.
9. **Avaliação sem anotação manual.** O experimento guiado grava a fase do roteiro em cada quadro; a avaliação usa isso como gabarito.
10. **Integração com microcontrolador.** Protocolo serial de 1 byte com reenvio a cada 1 s e *timeout* no Arduino.
11. **Dependências.** `opencv-contrib-python` (o mesmo pacote que o MediaPipe usa) evita duas cópias do `cv2`; código testado com as versões atuais (MediaPipe 1.2.2, OpenCV 5.0).

### Dificuldades encontradas

| Dificuldade | Solução |
|---|---|
| `FileNotFoundError` do MediaPipe no Windows, com o projeto na `Área de Trabalho` | Modelo lido em memória; fotos e vídeo gravados sem depender do caminho com acento |
| Vídeo gravado com as anotações da tela por cima, impossível de reprocessar | Gravação do quadro original, antes de desenhar |
| Vídeo gravado com FPS fixo, distorcendo os tempos | FPS medido nos primeiros 60 quadros |
| Anotar manualmente o início e o fim de cada evento | Roteiro guiado que grava o gabarito |
| Duas cópias do OpenCV instaladas (`opencv-python` e `opencv-contrib-python`) | Apenas `opencv-contrib-python` no `requirements.txt` |

---

## 4. Instalação

Python 3.12 ou mais novo. Rode os comandos dentro desta pasta.

Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS / Linux:
```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Na primeira execução, o modelo `face_landmarker.task` é baixado para `modelos/`.

## 5. Uso (demonstração)

```
.venv\Scripts\python detector_sonolencia.py                        # webcam 0
.venv\Scripts\python detector_sonolencia.py --metodo blendshapes   # segundo método
.venv\Scripts\python detector_sonolencia.py --serial COM3          # com o Arduino
.venv\Scripts\python detector_sonolencia.py --video sessao.mp4     # vídeo gravado
```

Nos primeiros **5 s**, olhe para a câmera com os olhos abertos (calibração). Teclas: `q` sai, `c` recalibra, `s` salva foto, espaço marca um evento no log.

| Estado | Regra | Saída |
|---|---|---|
| ALARME | olhos fechados ≥ 1,5 s **ou** cabeça desviada > 20° por ≥ 1,5 s | borda vermelha, bip, LED vermelho + buzzer |
| ATENÇÃO | PERCLOS (60 s) ≥ 15% **ou** 3 bocejos em 5 min | LED amarelo |
| SEM ROSTO | rosto não detectado por ≥ 2 s | LED amarelo |
| NORMAL | — | LED verde |

Parâmetros: `sonolencia/logica.py` (`Parametros`); os principais também pela linha de comando (`--t-microcochilo`, `--fator-ear`, `--mar-bocejo`, `--perclos-atencao`).

## 6. Avaliação quantitativa

```
.venv\Scripts\python experimento.py --rotulo luz_acesa --lux 300
.venv\Scripts\python experimento.py --rotulo penumbra --lux 5
.venv\Scripts\python avaliar_experimento.py logs\experimento_*.csv
```

O roteiro (≈ 5 min) tem calibração, 60 s de comportamento normal (piscar e falar), 15 fechamentos de olhos de 3 s, 8 bocejos e 8 quedas de cabeça. **1 bipe grave = comece a ação; 2 bipes agudos = volte ao normal.** `--rapido` faz um ensaio de cerca de 1 min.

Métodos comparados, todos com a mesma máquina de estados e sobre os mesmos quadros:

| Método | Olhos | Boca |
|---|---|---|
| M1 Geométrico | EAR com limiar calibrado | MAR > 0,6 |
| M2 *Blendshapes* | 1 − `eyeBlink` com limiar calibrado | `jawOpen` > 0,5 |
| M3 Referência | EAR com limiar fixo 0,2 | MAR > 0,6 |

Um evento é verdadeiro positivo se o alarme (ou a contagem do bocejo) começa entre o bipe e o fim do evento + 2 s. Saídas em `resultados/`: `resultados.csv`, `tabela.tex` e um gráfico por sessão.

## 7. Arduino (opcional)

Sketch em `arduino/alerta_sonolencia/`. Ligações: D5 LED verde, D6 amarelo, D7 vermelho (resistores de 220 Ω), D8 buzzer ativo. O computador envia `N`, `W` ou `A` a cada mudança de estado e a cada 1 s; sem mensagem por 3 s, o Arduino apaga tudo e pisca o amarelo.

## 8. Testes

```
.venv\Scripts\python -m pytest tests
```
