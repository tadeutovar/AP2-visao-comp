"""Testes da lógica com sinais simulados (sem câmera). Rode: python -m pytest tests"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sonolencia import (MonitorSonolencia, Parametros, ear, mar, angulos_euler,  # noqa: E402
                        NORMAL, ATENCAO, ALARME, SEM_ROSTO, CALIBRANDO)
from sonolencia.logica import OLHO_DIREITO, BOCA_VERTICAIS, BOCA_CANTOS  # noqa: E402

FPS = 30
DT = 1.0 / FPS


def simular(mon, t0, dur, ear_v=0.30, mar_v=0.05, pitch=0.0, rosto=True):
    leituras, n = [], int(round(dur * FPS))
    for i in range(n):
        leituras.append((t0 + i * DT, mon.atualizar(t0 + i * DT, rosto, ear_v, mar_v, pitch)))
    return t0 + n * DT, leituras


def calibrado(p=None):
    mon = MonitorSonolencia(p or Parametros())
    t, _ = simular(mon, 0.0, 5.1)
    return mon, t


def test_ear_geometria():
    pts = np.zeros((478, 2))
    # olho com largura 10 e duas aberturas de 3 -> EAR = (3+3)/(2*10) = 0,3
    p1, p2, p3, p4, p5, p6 = OLHO_DIREITO
    pts[p1], pts[p4] = (0, 0), (10, 0)
    pts[p2], pts[p6] = (3, -1.5), (3, 1.5)
    pts[p3], pts[p5] = (7, -1.5), (7, 1.5)
    assert math.isclose(ear(pts, OLHO_DIREITO), 0.3)


def test_mar_geometria():
    pts = np.zeros((478, 2))
    a, b = BOCA_CANTOS
    pts[a], pts[b] = (0, 0), (40, 0)
    for (s, i), x in zip(BOCA_VERTICAIS, (10, 20, 30)):
        pts[s], pts[i] = (x, -12), (x, 12)
    assert math.isclose(mar(pts), 24 / 40)


def test_euler_identidade_e_pitch():
    assert np.allclose(angulos_euler(np.eye(4)), (0, 0, 0))
    a = math.radians(25)
    rx = np.eye(4)
    rx[1:3, 1:3] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    assert math.isclose(angulos_euler(rx)[0], 25, abs_tol=1e-6)


def test_calibracao_define_limiar():
    mon = MonitorSonolencia()
    _, lei = simular(mon, 0.0, 4.0)
    assert all(l.estado == CALIBRANDO for _, l in lei)
    simular(mon, 4.0, 1.2)
    assert mon.calibrado and math.isclose(mon.limiar_ear, 0.225)


def test_piscadas_normais_nao_disparam():
    mon, t = calibrado()
    estados = []
    for _ in range(30):  # 30 piscadas de 0,3 s a cada 4 s (2 min)
        t, l1 = simular(mon, t, 3.7)
        t, l2 = simular(mon, t, 0.3, ear_v=0.08)
        estados += [l.estado for _, l in l1 + l2]
    assert ALARME not in estados and ATENCAO not in estados


def test_microcochilo_dispara_alarme():
    mon, t = calibrado()
    t, _ = simular(mon, t, 10)
    ini = t
    t, lei = simular(mon, t, 2.0, ear_v=0.08)
    primeiro = next(tt for tt, l in lei if l.estado == ALARME)
    assert 1.45 <= primeiro - ini <= 1.6
    t, depois = simular(mon, t, 3.0)  # retenção de 2 s e volta ao normal
    assert depois[int(1.5 * FPS)][1].estado == ALARME
    assert depois[-1][1].estado == NORMAL


def test_perclos_alto_gera_atencao():
    mon, t = calibrado()
    for _ in range(30):  # 30 % do tempo com olhos fechados, em fechamentos curtos
        t, _ = simular(mon, t, 0.7)
        t, lei = simular(mon, t, 0.3, ear_v=0.08)
    assert lei[-1][1].perclos > 0.15
    t, lei = simular(mon, t, 0.5)
    assert lei[-1][1].estado == ATENCAO


def test_tres_bocejos_geram_atencao():
    mon, t = calibrado()
    for _ in range(3):
        t, _ = simular(mon, t, 5)
        t, lei = simular(mon, t, 3.0, mar_v=0.8)
        assert sum(l.novo_bocejo for _, l in lei) == 1
    t, lei = simular(mon, t, 1)
    assert lei[-1][1].estado == ATENCAO and lei[-1][1].bocejos_janela == 3


def test_fala_nao_conta_bocejo():
    mon, t = calibrado()
    for _ in range(40):  # aberturas curtas da boca
        t, _ = simular(mon, t, 0.3, mar_v=0.7)
        t, lei = simular(mon, t, 0.3)
    assert lei[-1][1].bocejos_janela == 0


def test_queda_da_cabeca_dispara_alarme():
    mon, t = calibrado()
    t, lei = simular(mon, t, 2.0, pitch=-30.0)
    assert lei[-1][1].estado == ALARME
    assert "cabeca" in lei[-1][1].motivo


def test_olhar_retrovisor_rapido_nao_dispara():
    mon, t = calibrado()
    t, lei = simular(mon, t, 1.0, pitch=25.0)
    assert all(l.estado != ALARME for _, l in lei)


def test_sem_rosto():
    mon, t = calibrado()
    t, lei = simular(mon, t, 3.0, rosto=False)
    assert lei[int(1.0 * FPS)][1].estado == NORMAL
    assert lei[-1][1].estado == SEM_ROSTO
