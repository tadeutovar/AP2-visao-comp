#!/usr/bin/env python3
"""
Resumo quantitativo e gráfico de uma sessão gravada pelo detector.

  python analisar_sessao.py logs/sessao_20261105_190000_300lux.csv
  python analisar_sessao.py logs/sessao_...csv --eventos eventos_300lux.csv

Sem --eventos: FPS, latência (mediana e p95), tempo em cada estado, nº de
alarmes e de bocejos, e um gráfico (PNG) do indicador dos olhos e da boca.

Com --eventos (CSV com colunas inicio_s,fim_s,tipo; tipo em
{microcochilo, bocejo, cabeca}): conta VP, FN e FP por tipo de evento,
precisão, revocação, F1 e tempo até o alarme.
  - microcochilo/cabeca: VP se um ALARME começa entre inicio_s e fim_s + tolerância.
  - bocejo: VP se um novo bocejo é contado nesse intervalo.
  - FP: alarmes/bocejos que não caem em nenhum evento do mesmo grupo.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def carregar(caminho: Path) -> dict[str, np.ndarray]:
    with open(caminho, encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    if not linhas:
        raise SystemExit("Log vazio.")
    dados = {}
    for k in linhas[0]:
        col = [r[k] for r in linhas]
        try:
            dados[k] = np.array([float(v) for v in col])
        except ValueError:
            dados[k] = np.array(col)
    return dados


def inicios(mascara: np.ndarray) -> np.ndarray:
    m = mascara.astype(int)
    return np.flatnonzero(np.diff(np.concatenate([[0], m])) == 1)


def avaliar(d, eventos, tol):
    t = d["t_s"]
    alarmes = t[inicios(d["estado"] == "ALARME")]
    bocejos = t[d["novo_bocejo"] == 1]
    grupos = {"microcochilo": alarmes, "cabeca": alarmes, "bocejo": bocejos}
    res = {}
    usados = {"alarme": set(), "bocejo": set()}
    for tipo in ("microcochilo", "cabeca", "bocejo"):
        evs = [e for e in eventos if e["tipo"] == tipo]
        if not evs:
            continue
        deteccoes = grupos[tipo]
        chave = "bocejo" if tipo == "bocejo" else "alarme"
        vp, atrasos = 0, []
        for e in evs:
            hit = [i for i, td in enumerate(deteccoes) if e["inicio_s"] <= td <= e["fim_s"] + tol]
            if hit:
                vp += 1
                usados[chave].add(hit[0])
                atrasos.append(deteccoes[hit[0]] - e["inicio_s"])
        res[tipo] = {"eventos": len(evs), "VP": vp, "FN": len(evs) - vp,
                     "atraso_medio_s": float(np.mean(atrasos)) if atrasos else float("nan")}
    for chave, tipos in (("alarme", ("microcochilo", "cabeca")), ("bocejo", ("bocejo",))):
        dets = grupos["bocejo"] if chave == "bocejo" else alarmes
        fp = len(dets) - len(usados[chave])
        presentes = [tp for tp in tipos if tp in res]
        if presentes:
            res[presentes[0]]["FP_grupo"] = fp
    for r in res.values():
        fp = r.get("FP_grupo", 0)
        prec = r["VP"] / (r["VP"] + fp) if r["VP"] + fp else float("nan")
        rev = r["VP"] / r["eventos"]
        r["precisao"], r["revocacao"] = prec, rev
        r["F1"] = 2 * prec * rev / (prec + rev) if prec + rev and not np.isnan(prec) else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--eventos", type=Path)
    ap.add_argument("--tolerancia", type=float, default=2.0, help="s após o fim do evento")
    ap.add_argument("--sem-grafico", action="store_true")
    args = ap.parse_args()

    d = carregar(args.log)
    t = d["t_s"]
    cal = d["estado"] != "CALIBRANDO"
    dur = t[-1] - t[0]
    print(f"\nSessão: {args.log.name}")
    print(f"Duração: {dur:.1f} s | quadros: {len(t)} | rosto detectado em {d['rosto'][cal].mean():.1%} dos quadros")
    print(f"FPS médio: {len(t) / dur:.1f} | latência mediana {np.median(d['latencia_ms']):.1f} ms"
          f" | p95 {np.percentile(d['latencia_ms'], 95):.1f} ms")
    print("Tempo em cada estado:")
    for e in ("CALIBRANDO", "NORMAL", "ATENCAO", "ALARME", "SEM_ROSTO"):
        frac = (d["estado"] == e).mean()
        if frac:
            print(f"  {e:<11} {frac:6.1%}")
    print(f"Alarmes disparados: {len(inicios(d['estado'] == 'ALARME'))} | bocejos contados: {int(d['novo_bocejo'].sum())}"
          f" | PERCLOS máx.: {d['perclos'].max():.1%}")

    if args.eventos:
        with open(args.eventos, encoding="utf-8") as f:
            evs = [{"inicio_s": float(r["inicio_s"]), "fim_s": float(r["fim_s"]), "tipo": r["tipo"].strip()}
                   for r in csv.DictReader(f)]
        print("\nAvaliação por evento (VP / FN / FP):")
        for tipo, r in avaliar(d, evs, args.tolerancia).items():
            print(f"  {tipo:<12} n={r['eventos']:>3}  VP={r['VP']:>3}  FN={r['FN']:>3}  FP={r.get('FP_grupo', 0):>3}"
                  f"  precisão={r['precisao']:.2f}  revocação={r['revocacao']:.2f}  F1={r['F1']:.2f}"
                  f"  atraso médio={r['atraso_medio_s']:.2f} s")

    if not args.sem_grafico:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        a1.plot(t, d["olho"], lw=0.8, color="#1f5fa8", label="EAR")
        a1.plot(t, d["limiar_olho"], "--", lw=1, color="#555555", label="limiar")
        a2.plot(t, d["boca"], lw=0.8, color="#a8551f", label="MAR")
        for ax in (a1, a2):
            alarme = d["estado"] == "ALARME"
            ax.fill_between(t, 0, 1, where=alarme, transform=ax.get_xaxis_transform(),
                            color="#d62728", alpha=0.15, lw=0, label="alarme")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)
        a1.set_ylabel("Olhos")
        a2.set_ylabel("Boca")
        a2.set_xlabel("Tempo (s)")
        fig.tight_layout()
        saida = args.log.with_suffix(".png")
        fig.savefig(saida, dpi=200)
        print(f"\nGráfico salvo em {saida}")


if __name__ == "__main__":
    main()
