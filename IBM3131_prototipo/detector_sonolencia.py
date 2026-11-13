#!/usr/bin/env python3
"""
Detector de sonolência em tempo real — IBM3131 Visão Computacional (Ibmec 2026.2)

Câmera USB -> OpenCV -> MediaPipe Face Landmarker (478 marcos)
-> EAR / MAR / inclinação da cabeça -> máquina de estados
-> alerta na tela, sonoro e (opcional) no Arduino via serial.

Exemplos
  python detector_sonolencia.py                     # webcam 0
  python detector_sonolencia.py --camera 1          # outra câmera
  python detector_sonolencia.py --serial COM3       # Arduino no Windows (Linux: /dev/ttyACM0)
  python detector_sonolencia.py --video sessao.mp4  # avaliação offline de um vídeo
  python detector_sonolencia.py --gravar            # grava o vídeo da sessão (para anotação)
  python detector_sonolencia.py --metodo blendshapes  # método alternativo (Semana 10)

Teclas (janela de vídeo)
  q sair | c recalibrar | s salvar foto | espaço marcar evento no log
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from sonolencia import (MonitorSonolencia, Parametros, parametros_blendshapes, ear_medio,
                        mar, angulos_euler, NORMAL, ATENCAO, ALARME, SEM_ROSTO, CALIBRANDO)
from sonolencia.logica import OLHO_DIREITO, OLHO_ESQUERDO, BOCA_VERTICAIS, BOCA_CANTOS

RAIZ = Path(__file__).resolve().parent
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
MODELO_PADRAO = RAIZ / "modelos" / "face_landmarker.task"

CORES = {  # BGR
    NORMAL: (60, 180, 75), ATENCAO: (0, 200, 255), ALARME: (40, 40, 230),
    SEM_ROSTO: (160, 160, 160), CALIBRANDO: (230, 160, 40),
}
CODIGO_SERIAL = {NORMAL: b"N", CALIBRANDO: b"N", ATENCAO: b"W", SEM_ROSTO: b"W", ALARME: b"A"}


# ----------------------------------------------------------------------
def caminho_ascii(p) -> bool:
    """MediaPipe e OpenCV no Windows não abrem caminhos com acento (ex.: 'Área de Trabalho')."""
    try:
        str(p).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def salvar_imagem(caminho: Path, img) -> bool:
    """Equivalente a cv2.imwrite, mas funciona em caminhos com acento."""
    ok, buf = cv2.imencode(caminho.suffix or ".png", img)
    if ok:
        caminho.write_bytes(buf.tobytes())
    return bool(ok)


def garantir_modelo(caminho: Path) -> Path:
    if caminho.exists() and caminho.stat().st_size > 1_000_000:
        return caminho
    caminho.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando o modelo do MediaPipe para {caminho} ...")
    parcial = caminho.with_suffix(".part")
    urllib.request.urlretrieve(MODELO_URL, parcial)
    parcial.replace(caminho)
    return caminho


def criar_landmarker(modelo: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    # O modelo é lido pelo Python e entregue em memória: evita o erro do MediaPipe
    # com caminhos que têm acento ou outros caracteres não ASCII no Windows.
    opcoes = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=Path(modelo).read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mp, vision.FaceLandmarker.create_from_options(opcoes)


class SaidaSerial:
    """Envia N/W/A ao Arduino na mudança de estado e a cada 1 s (heartbeat)."""

    def __init__(self, porta: str | None, baud: int = 9600):
        self.ser, self.ultimo, self.t_ultimo = None, None, 0.0
        if not porta:
            return
        try:
            import serial  # pyserial
            self.ser = serial.Serial(porta, baud, timeout=0)
            time.sleep(2.0)  # o Arduino Uno reinicia ao abrir a porta
            print(f"Serial aberta em {porta} ({baud} bps, 8N1)")
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] não foi possível abrir a serial {porta}: {e}")

    def enviar(self, estado: str):
        if not self.ser:
            return
        cod = CODIGO_SERIAL.get(estado, b"N")
        agora = time.monotonic()
        if cod != self.ultimo or agora - self.t_ultimo >= 1.0:
            try:
                self.ser.write(cod)
                self.ultimo, self.t_ultimo = cod, agora
            except Exception as e:  # noqa: BLE001
                print(f"[aviso] falha na serial: {e}")
                self.ser = None

    def fechar(self):
        if self.ser:
            try:
                self.ser.write(b"N")
                self.ser.close()
            except Exception:  # noqa: BLE001
                pass


class Bip:
    """Bip no computador enquanto o estado for ALARME (sem bloquear o laço)."""

    def __init__(self, ativo: bool):
        self.ativo, self.t = ativo, 0.0

    def tocar(self):
        if not self.ativo or time.monotonic() - self.t < 0.8:
            return
        self.t = time.monotonic()
        threading.Thread(target=self._bip, daemon=True).start()

    @staticmethod
    def _bip():
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(2000, 350)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()


# ----------------------------------------------------------------------
def desenhar(quadro, pts, leitura, fps, lat_ms, metodo, marcador):
    h, w = quadro.shape[:2]
    cor = CORES[leitura.estado]
    if pts is not None:
        for i in OLHO_DIREITO + OLHO_ESQUERDO:
            cv2.circle(quadro, tuple(int(v) for v in pts[i]), 2, (0, 255, 255), -1)
        for a, b in BOCA_VERTICAIS:
            for i in (a, b):
                cv2.circle(quadro, tuple(int(v) for v in pts[i]), 2, (255, 0, 255), -1)
        for i in BOCA_CANTOS:
            cv2.circle(quadro, tuple(int(v) for v in pts[i]), 2, (255, 0, 255), -1)

    cv2.rectangle(quadro, (0, 0), (w, 44), cor, -1)
    titulo = leitura.estado if not leitura.motivo else f"{leitura.estado}: {leitura.motivo}"
    if leitura.estado == CALIBRANDO:
        titulo = f"CALIBRANDO - olhe para a frente com os olhos abertos ({leitura.progresso_calibracao:.0%})"
    cv2.putText(quadro, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    if leitura.estado == ALARME:
        cv2.rectangle(quadro, (0, 0), (w - 1, h - 1), cor, 10)

    def fmt(v, f="{:.3f}"):
        return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)

    nome_olho = "EAR" if metodo == "geometrico" else "olho"
    nome_boca = "MAR" if metodo == "geometrico" else "jawOpen"
    linhas = [
        f"{nome_olho} {fmt(leitura.ear)}  limiar {fmt(leitura.limiar_ear)}",
        f"{nome_boca} {fmt(leitura.mar)}  bocejos(5min) {leitura.bocejos_janela}",
        f"PERCLOS(60s) {fmt(leitura.perclos, '{:.0%}')}  cabeca {fmt(leitura.desvio_cabeca, '{:.0f}')} graus",
        f"{fps:.1f} FPS  latencia {lat_ms:.0f} ms  metodo {metodo}  evento #{marcador}",
    ]
    y = h - 12 - 24 * (len(linhas) - 1)
    for t in linhas:
        cv2.putText(quadro, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(quadro, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24


def extrair(res, w, h, metodo):
    """Retorna (rosto, pts_px, olho, boca, pitch)."""
    if not res.face_landmarks:
        return False, None, float("nan"), float("nan"), float("nan")
    lms = res.face_landmarks[0]
    pts = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)
    pitch = float("nan")
    if res.facial_transformation_matrixes:
        pitch = angulos_euler(res.facial_transformation_matrixes[0])[0]
    if metodo == "blendshapes" and res.face_blendshapes:
        bs = {c.category_name: c.score for c in res.face_blendshapes[0]}
        olho = 1.0 - 0.5 * (bs.get("eyeBlinkLeft", 0.0) + bs.get("eyeBlinkRight", 0.0))
        boca = bs.get("jawOpen", 0.0)
    else:
        olho, boca = ear_medio(pts), mar(pts)
    return True, pts, olho, boca, pitch


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Detector de sonolência (IBM3131)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", type=str, help="arquivo de vídeo para avaliação offline")
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--altura", type=int, default=720)
    ap.add_argument("--serial", type=str, help="porta do Arduino (ex.: COM3, /dev/ttyACM0)")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--metodo", choices=["geometrico", "blendshapes"], default="geometrico")
    ap.add_argument("--modelo", type=Path, default=MODELO_PADRAO)
    ap.add_argument("--sem-janela", action="store_true", help="sem interface gráfica")
    ap.add_argument("--sem-som", action="store_true")
    ap.add_argument("--gravar", action="store_true",
                    help="grava o vídeo da sessão sem anotações (reprocessável com --video)")
    ap.add_argument("--sem-fotos-auto", action="store_true",
                    help="não salvar foto automaticamente a cada novo alarme")
    ap.add_argument("--rotulo", type=str, default="", help="rótulo da sessão (ex.: 300lux)")
    # parâmetros principais (para os experimentos)
    ap.add_argument("--t-microcochilo", type=float)
    ap.add_argument("--fator-ear", type=float)
    ap.add_argument("--mar-bocejo", type=float)
    ap.add_argument("--perclos-atencao", type=float)
    args = ap.parse_args()

    p = parametros_blendshapes() if args.metodo == "blendshapes" else Parametros()
    for nome in ("t_microcochilo", "fator_ear", "mar_bocejo", "perclos_atencao"):
        v = getattr(args, nome)
        if v is not None:
            setattr(p, nome, v)
    monitor = MonitorSonolencia(p)

    mp, landmarker = criar_landmarker(garantir_modelo(args.modelo))

    offline = args.video is not None
    temp_dir = None  # pasta temporária sem acento, se precisar

    def em_temp(nome: str) -> Path:
        nonlocal temp_dir
        if temp_dir is None:
            temp_dir = Path(tempfile.mkdtemp(prefix="ibm3131_"))
            if not caminho_ascii(temp_dir) and sys.platform.startswith("win"):
                temp_dir = Path(tempfile.mkdtemp(prefix="ibm3131_", dir="C:/Users/Public"))
        return temp_dir / nome

    fonte = args.camera
    if offline:
        fonte = args.video
        if not caminho_ascii(Path(args.video).resolve()):
            copia = em_temp("entrada" + Path(args.video).suffix)
            shutil.copyfile(args.video, copia)
            fonte = str(copia)
    cap = cv2.VideoCapture(fonte)
    if not offline:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.altura)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a câmera/vídeo.")
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufixo = f"_{args.rotulo}" if args.rotulo else ""
    (RAIZ / "logs").mkdir(exist_ok=True)
    (RAIZ / "fotos").mkdir(exist_ok=True)
    caminho_log = RAIZ / "logs" / f"sessao_{carimbo}{sufixo}.csv"
    arq = open(caminho_log, "w", newline="", encoding="utf-8")
    log = csv.writer(arq)
    log.writerow(["t_s", "quadro", "rosto", "olho", "limiar_olho", "boca", "desvio_cabeca",
                  "olho_fechado", "t_fechado", "perclos", "bocejos_janela", "novo_bocejo",
                  "estado", "motivo", "latencia_ms", "fps", "evento"])

    gravador, buffer_gravacao = None, []
    caminho_video = RAIZ / "logs" / f"sessao_{carimbo}{sufixo}.mp4"
    destino_video = [caminho_video]

    def abrir_gravador(buffer, fps_fixo):
        """Abre o VideoWriter com o FPS real da câmera, medido nos primeiros quadros."""
        if fps_fixo:
            fps_rec = fps_fixo
        elif len(buffer) > 1 and buffer[-1][0] > buffer[0][0]:
            fps_rec = (len(buffer) - 1) / (buffer[-1][0] - buffer[0][0])
        else:
            fps_rec = 30.0
        alt, larg = buffer[0][1].shape[:2]
        destino = caminho_video if caminho_ascii(caminho_video) else em_temp(caminho_video.name)
        gw = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_rec, (larg, alt))
        if not gw.isOpened():
            print(f"[aviso] não foi possível gravar o vídeo em {destino}")
        destino_video[0] = destino  # para mover ao final, se gravou em pasta temporária
        for _, q in buffer:
            gw.write(q)
        print(f"Gravando vídeo sem anotações a {fps_rec:.1f} FPS: {caminho_video.name}")
        return gw

    serial_out = SaidaSerial(args.serial, args.baud)
    bip = Bip(not args.sem_som)
    marcador, n_quadro, t_ini = 0, 0, time.monotonic()
    fps_suave, estado_anterior, t_anterior = 0.0, None, None
    print(f"Log: {caminho_log}\nCalibrando por {p.t_calibracao:.0f} s: olhe para a frente com os olhos abertos.")

    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                break
            t_quadro = n_quadro / fps_video if offline else time.monotonic() - t_ini
            if not offline:
                quadro = cv2.flip(quadro, 1)  # efeito espelho
            h, w = quadro.shape[:2]
            bruto = quadro.copy() if args.gravar else None

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)
            imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(imagem, int(t_quadro * 1000))
            rosto, pts, olho, boca, pitch = extrair(res, w, h, args.metodo)
            leitura = monitor.atualizar(t_quadro, rosto, olho, boca, pitch)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            # FPS: offline mede a vazão do processamento; ao vivo, o intervalo entre quadros
            if offline:
                fps_inst = 1000.0 / max(lat_ms, 1e-3)
            else:
                dt = t_quadro - t_anterior if t_anterior is not None else 0.0
                fps_inst = 1.0 / dt if dt > 0 else fps_suave
            fps_suave = fps_inst if fps_suave == 0 else 0.9 * fps_suave + 0.1 * fps_inst
            t_anterior = t_quadro

            if leitura.estado == ALARME:
                bip.tocar()
            serial_out.enviar(leitura.estado)

            desenhar(quadro, pts, leitura, fps_suave, lat_ms, args.metodo, marcador)

            if (leitura.estado == ALARME and estado_anterior != ALARME
                    and not args.sem_fotos_auto):
                salvar_imagem(RAIZ / "fotos" / f"alarme_{carimbo}{sufixo}_{t_quadro:07.2f}s.png", quadro)
            estado_anterior = leitura.estado

            log.writerow([f"{t_quadro:.3f}", n_quadro, int(rosto), f"{olho:.4f}",
                          f"{leitura.limiar_ear:.4f}", f"{boca:.4f}",
                          f"{leitura.desvio_cabeca:.2f}", int(leitura.olho_fechado),
                          f"{leitura.t_fechado:.2f}", f"{leitura.perclos:.4f}",
                          leitura.bocejos_janela, int(leitura.novo_bocejo), leitura.estado,
                          leitura.motivo, f"{lat_ms:.1f}", f"{fps_suave:.2f}", marcador])

            if args.gravar and bruto is not None:
                # Grava o quadro SEM anotações, para poder reprocessar o vídeo depois
                # (ex.: comparar os métodos geométrico e blendshapes nas mesmas sessões).
                if gravador is None:
                    buffer_gravacao.append((t_quadro, bruto))
                    if offline or len(buffer_gravacao) >= 60:
                        gravador = abrir_gravador(buffer_gravacao, fps_video if offline else None)
                        buffer_gravacao.clear()
                else:
                    gravador.write(bruto)

            if not args.sem_janela:
                cv2.imshow("Detector de sonolencia - IBM3131", quadro)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("c"):
                    monitor.reiniciar_calibracao()
                    print("Recalibrando...")
                if tecla == ord("s"):
                    foto = RAIZ / "fotos" / f"foto_{carimbo}{sufixo}_{leitura.estado}_{t_quadro:07.2f}s.png"
                    salvar_imagem(foto, quadro)
                    print(f"Foto salva: {foto.name}")
                if tecla == ord(" "):
                    marcador += 1
                    print(f"Evento #{marcador} marcado em t={t_quadro:.2f}s")
            n_quadro += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        arq.close()
        if gravador is None and buffer_gravacao:
            gravador = abrir_gravador(buffer_gravacao, fps_video if offline else None)
        if gravador is not None:
            gravador.release()
            if destino_video[0] != caminho_video and Path(destino_video[0]).exists():
                shutil.move(str(destino_video[0]), str(caminho_video))
                print(f"Vídeo salvo em {caminho_video}")
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        serial_out.fechar()
        landmarker.close()
        cv2.destroyAllWindows()
        print(f"Sessão encerrada: {n_quadro} quadros. Log em {caminho_log}")
        print(f"Resumo: python analisar_sessao.py \"{caminho_log}\"")


if __name__ == "__main__":
    main()
