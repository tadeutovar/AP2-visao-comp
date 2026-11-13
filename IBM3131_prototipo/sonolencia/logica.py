"""
Lógica do detector de sonolência, independente da câmera e do MediaPipe.

Indicadores
-----------
EAR  (Eye Aspect Ratio, Soukupová & Čech, 2016): abertura dos olhos.
MAR  (Mouth Aspect Ratio): abertura da boca (bocejo).
Inclinação: desvio do ângulo de arfagem (pitch) da cabeça em relação à
posição calibrada (queda do queixo / cabeça para trás).

Estados
-------
CALIBRANDO -> NORMAL | ATENCAO | ALARME | SEM_ROSTO
  ALARME   : olhos fechados por >= t_microcochilo  ou  cabeça fora da
             posição por >= t_cabeca (microcochilo).
  ATENCAO  : PERCLOS >= perclos_atencao  ou  >= bocejos_atencao bocejos na
             janela de bocejos (sinais graduais de sonolência).
  SEM_ROSTO: rosto não detectado por >= t_sem_rosto.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import statistics

import numpy as np

# Índices da malha de 468/478 pontos do MediaPipe (ordem p1..p6 da EAR)
OLHO_DIREITO = (33, 160, 158, 133, 153, 144)
OLHO_ESQUERDO = (362, 385, 387, 263, 373, 380)
# Boca (contorno interno dos lábios): 3 pares verticais + cantos internos
BOCA_VERTICAIS = ((82, 87), (13, 14), (312, 317))
BOCA_CANTOS = (78, 308)

NORMAL, ATENCAO, ALARME, SEM_ROSTO, CALIBRANDO = (
    "NORMAL", "ATENCAO", "ALARME", "SEM_ROSTO", "CALIBRANDO")


def _d(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def ear(pts, idx) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|), com pts em pixels."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    largura = _d(p1, p4)
    if largura < 1e-6:
        return 0.0
    return (_d(p2, p6) + _d(p3, p5)) / (2.0 * largura)


def ear_medio(pts) -> float:
    return 0.5 * (ear(pts, OLHO_DIREITO) + ear(pts, OLHO_ESQUERDO))


def mar(pts) -> float:
    """MAR = média das 3 aberturas verticais / largura interna da boca."""
    largura = _d(pts[BOCA_CANTOS[0]], pts[BOCA_CANTOS[1]])
    if largura < 1e-6:
        return 0.0
    vert = sum(_d(pts[a], pts[b]) for a, b in BOCA_VERTICAIS) / len(BOCA_VERTICAIS)
    return vert / largura


def angulos_euler(matriz: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) em graus a partir da matriz de transformação 4x4."""
    r = np.asarray(matriz, dtype=float)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return pitch, yaw, roll


@dataclass
class Parametros:
    t_calibracao: float = 5.0        # s com olhos abertos e cabeça na posição normal
    fator_ear: float = 0.75          # limiar = fator x EAR médio de olhos abertos
    ear_min: float = 0.15            # faixa permitida para o limiar calibrado
    ear_max: float = 0.30
    ear_padrao: float = 0.21         # usado se a calibração falhar
    t_microcochilo: float = 1.5      # s de olhos fechados contínuos -> ALARME
    janela_perclos: float = 60.0     # s
    perclos_atencao: float = 0.15    # fração de tempo com olhos fechados
    perclos_aquecimento: float = 20.0  # s mínimos de janela antes de usar o PERCLOS
    mar_bocejo: float = 0.6
    t_bocejo: float = 1.5            # s de boca aberta para contar um bocejo
    janela_bocejos: float = 300.0    # s
    bocejos_atencao: int = 3
    desvio_cabeca: float = 20.0      # graus em relação à posição calibrada
    t_cabeca: float = 1.5            # s com a cabeça fora da posição -> ALARME
    t_sem_rosto: float = 2.0         # s sem rosto -> SEM_ROSTO
    t_retencao_alarme: float = 2.0   # s que o alarme se mantém após a causa sumir


@dataclass
class Leitura:
    estado: str
    ear: float = float("nan")
    limiar_ear: float = float("nan")
    mar: float = float("nan")
    desvio_cabeca: float = float("nan")
    olho_fechado: bool = False
    t_fechado: float = 0.0
    perclos: float = 0.0
    bocejos_janela: int = 0
    novo_bocejo: bool = False
    motivo: str = ""
    progresso_calibracao: float = 1.0


@dataclass
class MonitorSonolencia:
    p: Parametros = field(default_factory=Parametros)

    def __post_init__(self):
        self.reiniciar_calibracao()

    # ------------------------------------------------------------------
    def reiniciar_calibracao(self):
        self._t0_calib = None
        self._calib_ear: list[float] = []
        self._calib_pitch: list[float] = []
        self.calibrado = False
        self.limiar_ear = self.p.ear_padrao
        self.pitch_base = 0.0
        self.ear_base = float("nan")
        self._perclos = deque()        # (t, fechado)
        self._soma_fechados = 0
        self._bocejos = deque()        # instantes dos bocejos
        self._ini_fechado = None
        self._ini_boca = None
        self._bocejo_contado = False
        self._ini_cabeca = None
        self._ultimo_rosto = None
        self._alarme_ate = -1.0
        self._motivo_alarme = ""

    # ------------------------------------------------------------------
    def atualizar(self, t: float, rosto: bool, ear_v: float = float("nan"),
                  mar_v: float = float("nan"), pitch: float = float("nan")) -> Leitura:
        p = self.p
        if rosto:
            self._ultimo_rosto = t

        # ---------------- calibração ----------------
        if not self.calibrado:
            if self._t0_calib is None:
                self._t0_calib = t
            if rosto:
                self._calib_ear.append(ear_v)
                if not math.isnan(pitch):
                    self._calib_pitch.append(pitch)
            prog = min(1.0, (t - self._t0_calib) / p.t_calibracao)
            if prog >= 1.0:
                self._concluir_calibracao()
            else:
                return Leitura(CALIBRANDO, ear=ear_v, mar=mar_v,
                               limiar_ear=self.limiar_ear, progresso_calibracao=prog)

        # ---------------- sem rosto ----------------
        if not rosto:
            self._ini_fechado = self._ini_boca = self._ini_cabeca = None
            sem = t - (self._ultimo_rosto if self._ultimo_rosto is not None else t)
            if t < self._alarme_ate:
                return Leitura(ALARME, motivo=self._motivo_alarme,
                               limiar_ear=self.limiar_ear)
            estado = SEM_ROSTO if sem >= p.t_sem_rosto else NORMAL
            return Leitura(estado, limiar_ear=self.limiar_ear,
                           perclos=self._valor_perclos(t),
                           bocejos_janela=self._contar_bocejos(t))

        # ---------------- olhos ----------------
        fechado = ear_v < self.limiar_ear
        if fechado:
            if self._ini_fechado is None:
                self._ini_fechado = t
            t_fechado = t - self._ini_fechado
        else:
            self._ini_fechado = None
            t_fechado = 0.0

        self._perclos.append((t, fechado))
        self._soma_fechados += int(fechado)
        while self._perclos and t - self._perclos[0][0] > p.janela_perclos:
            self._soma_fechados -= int(self._perclos.popleft()[1])
        perclos = self._valor_perclos(t)

        # ---------------- boca ----------------
        novo_bocejo = False
        if mar_v > p.mar_bocejo:
            if self._ini_boca is None:
                self._ini_boca, self._bocejo_contado = t, False
            if not self._bocejo_contado and t - self._ini_boca >= p.t_bocejo:
                self._bocejos.append(t)
                self._bocejo_contado = novo_bocejo = True
        else:
            self._ini_boca = None
        n_bocejos = self._contar_bocejos(t)

        # ---------------- cabeça ----------------
        desvio = abs(pitch - self.pitch_base) if not math.isnan(pitch) else float("nan")
        if not math.isnan(desvio) and desvio > p.desvio_cabeca:
            if self._ini_cabeca is None:
                self._ini_cabeca = t
            t_cabeca = t - self._ini_cabeca
        else:
            self._ini_cabeca = None
            t_cabeca = 0.0

        # ---------------- decisão ----------------
        motivo = ""
        if t_fechado >= p.t_microcochilo:
            motivo = f"olhos fechados {t_fechado:.1f}s"
        elif t_cabeca >= p.t_cabeca:
            motivo = f"cabeca fora da posicao {t_cabeca:.1f}s"
        if motivo:
            self._alarme_ate = t + p.t_retencao_alarme
            self._motivo_alarme = motivo

        if t < self._alarme_ate:
            estado, motivo = ALARME, self._motivo_alarme
        elif perclos >= p.perclos_atencao and self._janela_cheia(t):
            estado, motivo = ATENCAO, f"PERCLOS {perclos:.0%}"
        elif n_bocejos >= p.bocejos_atencao:
            estado, motivo = ATENCAO, f"{n_bocejos} bocejos"
        else:
            estado = NORMAL

        return Leitura(estado, ear=ear_v, limiar_ear=self.limiar_ear, mar=mar_v,
                       desvio_cabeca=desvio, olho_fechado=fechado,
                       t_fechado=t_fechado, perclos=perclos,
                       bocejos_janela=n_bocejos, novo_bocejo=novo_bocejo,
                       motivo=motivo)

    # ------------------------------------------------------------------
    def _concluir_calibracao(self):
        p = self.p
        validos = [e for e in self._calib_ear if not math.isnan(e)]
        if len(validos) >= 10:
            self.ear_base = statistics.median(validos)
            self.limiar_ear = min(p.ear_max, max(p.ear_min, p.fator_ear * self.ear_base))
        else:
            self.limiar_ear = p.ear_padrao
        if self._calib_pitch:
            self.pitch_base = statistics.median(self._calib_pitch)
        self.calibrado = True

    def _valor_perclos(self, t: float) -> float:
        return self._soma_fechados / len(self._perclos) if self._perclos else 0.0

    def _janela_cheia(self, t: float) -> bool:
        return bool(self._perclos) and t - self._perclos[0][0] >= self.p.perclos_aquecimento

    def _contar_bocejos(self, t: float) -> int:
        while self._bocejos and t - self._bocejos[0] > self.p.janela_bocejos:
            self._bocejos.popleft()
        return len(self._bocejos)
