#!/usr/bin/env python3
"""
Experimento guiado — gera os dados da avaliação quantitativa (Projeto AP2, IBM3131).

O programa conduz um roteiro na tela, com bipes: calibração, 60 s de
comportamento normal (piscar e falar), fechamentos de olhos, bocejos e quedas
de cabeça. Cada quadro é registrado com os indicadores dos DOIS métodos
(geométrico: EAR/MAR; blendshapes do MediaPipe) e com a fase do roteiro, que
serve de gabarito. Não é preciso anotar nada à mão.

  Windows:  .venv\\Scripts\\python experimento.py --rotulo luz_acesa
            .venv\\Scripts\\python experimento.py --rotulo penumbra
            .venv\\Scripts\\python experimento.py --rapido      (ensaio de ~1 min)

Bipes: 1 bipe grave = COMECE a ação | 2 bipes agudos = VOLTE ao normal.
Tecla q encerra (o que foi gravado até ali é salvo).

Depois:   .venv\\Scripts\\python avaliar_experimento.py logs\\experimento_*.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from detector_sonolencia import (RAIZ, MODELO_PADRAO, garantir_modelo, criar_landmarker,
                                 salvar_imagem)
from sonolencia import (MonitorSonolencia, Parametros, parametros_blendshapes, ear_medio, mar,
                        angulos_euler, ALARME, ATENCAO, CALIBRANDO, SEM_ROSTO)
from sonolencia.logica import OLHO_DIREITO, OLHO_ESQUERDO, BOCA_VERTICAIS, BOCA_CANTOS

COR_ESTADO = {ALARME: (40, 40, 230), ATENCAO: (0, 200, 255), CALIBRANDO: (230, 160, 40),
              SEM_ROSTO: (160, 160, 160)}
COR_NORMAL = (60, 180, 75)


# ----------------------------------------------------------------------
def montar_roteiro(n_olhos: int, n_bocejos: int, n_cabeca: int, t_normal: float):
    """Lista de (fase, duração_s, texto_na_tela, id_do_evento). Texto sem acento (OpenCV)."""
    r = [("calibracao", 7.0, "CALIBRACAO: olhe para a camera, olhos abertos", 0),
         ("normal", t_normal, "NORMAL: pisque e fale naturalmente (leia algo em voz alta)", 0)]
    ev = 0
    r.append(("pausa", 4.0, "Proximo: FECHE OS OLHOS no bipe, abra nos 2 bipes", 0))
    for i in range(n_olhos):
        ev += 1
        r.append(("microcochilo", 3.0, f"FECHE OS OLHOS  ({i + 1}/{n_olhos})", ev))
        r.append(("pausa", 4.0, "Olhos abertos, cabeca parada", 0))
    r.append(("pausa", 4.0, "Proximo: BOCEJE no bipe (boca bem aberta)", 0))
    for i in range(n_bocejos):
        ev += 1
        r.append(("bocejo", 4.0, f"BOCEJE - boca bem aberta  ({i + 1}/{n_bocejos})", ev))
        r.append(("pausa", 5.0, "Boca fechada, relaxe", 0))
    r.append(("pausa", 4.0, "Proximo: ABAIXE A CABECA ~30 graus no bipe, sem sair da camera", 0))
    for i in range(n_cabeca):
        ev += 1
        r.append(("cabeca", 3.0, f"ABAIXE A CABECA, como num cochilo  ({i + 1}/{n_cabeca})", ev))
        r.append(("pausa", 4.0, "Cabeca na posicao normal", 0))
    r.append(("fim", 2.0, "FIM - obrigado!", 0))
    return r


def tocar(freqs: list[int], ms: int = 140):
    """Bipes sem travar o laço de vídeo."""
    def _run():
        if sys.platform.startswith("win"):
            import winsound
            for f in freqs:
                winsound.Beep(f, ms)
        else:
            for _ in freqs:
                sys.stdout.write("\a")
                sys.stdout.flush()
                time.sleep(ms / 1000)
    threading.Thread(target=_run, daemon=True).start()


def indicadores(res, w, h):
    """Indicadores dos dois métodos a partir de UMA inferência do MediaPipe."""
    vazio = dict(rosto=0, ear=math.nan, mar=math.nan, olho_bs=math.nan, jaw_bs=math.nan,
                 blink_e=math.nan, blink_d=math.nan, pitch=math.nan, yaw=math.nan, roll=math.nan)
    if not res.face_landmarks:
        return vazio, None
    pts = np.array([(lm.x * w, lm.y * h) for lm in res.face_landmarks[0]], dtype=np.float32)
    d = dict(vazio, rosto=1, ear=ear_medio(pts), mar=mar(pts))
    if res.face_blendshapes:
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        d["blink_e"] = bs.get("eyeBlinkLeft", math.nan)
        d["blink_d"] = bs.get("eyeBlinkRight", math.nan)
        d["olho_bs"] = 1.0 - 0.5 * (d["blink_e"] + d["blink_d"])
        d["jaw_bs"] = bs.get("jawOpen", math.nan)
    if res.facial_transformation_matrixes:
        d["pitch"], d["yaw"], d["roll"] = angulos_euler(res.facial_transformation_matrixes[0])
    return d, pts


def desenhar_marcos(img, pts):
    if pts is None:
        return
    for i in OLHO_DIREITO + OLHO_ESQUERDO:
        cv2.circle(img, tuple(int(v) for v in pts[i]), 2, (0, 255, 255), -1)
    for a, b in BOCA_VERTICAIS:
        for i in (a, b):
            cv2.circle(img, tuple(int(v) for v in pts[i]), 2, (255, 0, 255), -1)
    for i in BOCA_CANTOS:
        cv2.circle(img, tuple(int(v) for v in pts[i]), 2, (255, 0, 255), -1)


def rotulo_estados(img, lg, lb, d):
    """Caixa com o estado de cada método (usada na tela e nas fotos)."""
    h = img.shape[0]
    linhas = [(f"Geometrico: {lg.estado}  EAR {d['ear']:.2f} (lim {lg.limiar_ear:.2f})  MAR {d['mar']:.2f}",
               COR_ESTADO.get(lg.estado, COR_NORMAL)),
              (f"Blendshapes: {lb.estado}  olho {d['olho_bs']:.2f} (lim {lb.limiar_ear:.2f})  jaw {d['jaw_bs']:.2f}",
               COR_ESTADO.get(lb.estado, COR_NORMAL))]
    y = h - 46
    for texto, cor in linhas:
        cv2.rectangle(img, (6, y - 20), (14, y - 4), cor, -1)
        cv2.putText(img, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, texto, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 26


def faixa_instrucao(img, texto, fase, restante, dur, proximo):
    w = img.shape[1]
    acao = fase in ("microcochilo", "bocejo", "cabeca")
    cor = (40, 40, 200) if acao else ((230, 160, 40) if fase == "calibracao" else (70, 70, 70))
    cv2.rectangle(img, (0, 0), (w, 70), cor, -1)
    cv2.putText(img, texto, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
    frac = 1.0 - max(0.0, restante) / dur if dur > 0 else 1.0
    cv2.rectangle(img, (0, 66), (int(w * frac), 70), (255, 255, 255), -1)
    cv2.putText(img, f"{restante:4.1f}s", (w - 90, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if proximo:
        cv2.putText(img, proximo, (14, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Experimento guiado (AP2 IBM3131)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rotulo", default="sessao", help="ex.: luz_acesa, penumbra")
    ap.add_argument("--lux", type=float, help="iluminância medida (app de luxímetro), opcional")
    ap.add_argument("--n-olhos", type=int, default=15)
    ap.add_argument("--n-bocejos", type=int, default=8)
    ap.add_argument("--n-cabeca", type=int, default=8)
    ap.add_argument("--t-normal", type=float, default=60.0)
    ap.add_argument("--rapido", action="store_true", help="ensaio curto: 2 eventos de cada")
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    args = ap.parse_args()
    if args.rapido:
        args.n_olhos = args.n_bocejos = args.n_cabeca = 2
        args.t_normal = 15.0

    roteiro = montar_roteiro(args.n_olhos, args.n_bocejos, args.n_cabeca, args.t_normal)
    inicios = np.cumsum([0.0] + [p[1] for p in roteiro])
    duracao_total = inicios[-1]
    print(f"Roteiro: {duracao_total / 60:.1f} min | olhos {args.n_olhos}, bocejos {args.n_bocejos}, "
          f"cabeca {args.n_cabeca}, normal {args.t_normal:.0f}s")

    mp, landmarker = criar_landmarker(garantir_modelo(MODELO_PADRAO))
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a câmera.")

    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RAIZ / "logs").mkdir(exist_ok=True)
    pasta_fotos = RAIZ / "fotos"
    pasta_fotos.mkdir(exist_ok=True)
    caminho_log = RAIZ / "logs" / f"experimento_{carimbo}_{args.rotulo}.csv"
    arq = open(caminho_log, "w", newline="", encoding="utf-8")
    log = csv.writer(arq)
    log.writerow([f"# rotulo={args.rotulo}", f"lux={args.lux if args.lux is not None else ''}",
                  f"largura={args.largura}", f"altura={args.altura}"])
    colunas = ["t_s", "quadro", "fase", "evento", "rosto", "ear", "mar", "olho_bs", "jaw_bs",
               "blink_e", "blink_d", "pitch", "yaw", "roll", "lat_ms", "estado_geo", "estado_bs"]
    log.writerow(colunas)

    mon_geo = MonitorSonolencia(Parametros())
    mon_bs = MonitorSonolencia(parametros_blendshapes())
    # fotos para a avaliação qualitativa: (fase, nº do evento dentro da fase, segundos após o início)
    fotos_pedidas = {("normal", 1): 30.0, ("microcochilo", 1): 2.4, ("bocejo", 1): 2.8,
                     ("cabeca", 1): 2.4, ("microcochilo", 2): 2.4, ("bocejo", 2): 2.8}
    fotos_feitas = set()
    contagem_fase = {}
    idx_atual, n_quadro, t0 = -1, 0, None
    print(f"Gravando em {caminho_log}")

    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            agora = time.monotonic()
            if t0 is None:
                t0 = agora
            t = agora - t0
            if t >= duracao_total:
                break
            idx = int(np.searchsorted(inicios, t, side="right") - 1)
            fase, dur, texto, evento = roteiro[idx]
            if idx != idx_atual:  # mudou de passo
                if fase in ("microcochilo", "bocejo", "cabeca"):
                    tocar([600])                 # comece
                    contagem_fase[fase] = contagem_fase.get(fase, 0) + 1
                elif idx_atual >= 0 and roteiro[idx_atual][0] in ("microcochilo", "bocejo", "cabeca"):
                    tocar([1400, 1400])          # volte ao normal
                elif fase == "normal":
                    contagem_fase["normal"] = 1
                    tocar([1000])
                idx_atual = idx
            t_no_passo = t - inicios[idx]

            quadro = cv2.flip(quadro, 1)
            h, w = quadro.shape[:2]
            ti = time.perf_counter()
            rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int(t * 1000))
            d, pts = indicadores(res, w, h)
            lat_ms = (time.perf_counter() - ti) * 1000.0
            lg = mon_geo.atualizar(t, bool(d["rosto"]), d["ear"], d["mar"], d["pitch"])
            lb = mon_bs.atualizar(t, bool(d["rosto"]), d["olho_bs"], d["jaw_bs"], d["pitch"])

            log.writerow([f"{t:.3f}", n_quadro, fase, evento, d["rosto"]] +
                         [f"{d[k]:.4f}" for k in ("ear", "mar", "olho_bs", "jaw_bs", "blink_e", "blink_d")] +
                         [f"{d[k]:.2f}" for k in ("pitch", "yaw", "roll")] +
                         [f"{lat_ms:.1f}", lg.estado, lb.estado])

            # foto limpa (marcos + estados) para o artigo
            chave = (fase, contagem_fase.get(fase, 0))
            if chave in fotos_pedidas and chave not in fotos_feitas and t_no_passo >= fotos_pedidas[chave]:
                foto = quadro.copy()
                desenhar_marcos(foto, pts)
                rotulo_estados(foto, lg, lb, d)
                salvar_imagem(pasta_fotos / f"exp_{args.rotulo}_{fase}{chave[1]}_{carimbo}.png", foto)
                fotos_feitas.add(chave)

            # tela
            desenhar_marcos(quadro, pts)
            prox = ""
            if idx + 1 < len(roteiro) and roteiro[idx + 1][0] in ("microcochilo", "bocejo", "cabeca"):
                prox = "No bipe: " + roteiro[idx + 1][2].split("  (")[0]
            faixa_instrucao(quadro, texto, fase, dur - t_no_passo, dur, prox)
            if d["rosto"]:
                rotulo_estados(quadro, lg, lb, d)
            else:
                cv2.putText(quadro, "ROSTO NAO DETECTADO", (14, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (40, 40, 230), 2)
            cv2.imshow("Experimento guiado - IBM3131", quadro)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                print("Interrompido pelo usuário.")
                break
            n_quadro += 1
    finally:
        cap.release()
        arq.close()
        landmarker.close()
        cv2.destroyAllWindows()

    dur = (time.monotonic() - t0) if t0 else 0.0
    print(f"\n{n_quadro} quadros em {dur:.0f} s ({n_quadro / max(dur, 1e-6):.1f} FPS).")
    print(f"Log: {caminho_log}")
    print(f"Fotos: {len(fotos_feitas)} em {pasta_fotos}")
    print(f'\nResultados:  python avaliar_experimento.py "{caminho_log}"')


if __name__ == "__main__":
    main()
