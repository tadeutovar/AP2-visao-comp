#!/usr/bin/env python3
"""
Avaliação quantitativa do experimento guiado — compara os métodos sobre os MESMOS quadros.

  python avaliar_experimento.py logs/experimento_*_luz_acesa.csv logs/experimento_*_penumbra.csv

Métodos comparados (mesma máquina de estados, entradas diferentes):
  M1  Geométrico, EAR com limiar calibrado por pessoa (0,75 x EAR de olhos abertos) + MAR
  M2  Blendshapes do MediaPipe (1 - eyeBlink, jawOpen), limiar também calibrado
  M3  Geométrico com limiar FIXO de EAR = 0,2 (como na literatura), referência

Gabarito = fases do roteiro gravadas no log. Para cada evento do roteiro:
  microcochilo / cabeca: verdadeiro positivo (VP) se um ALARME começa entre o bipe de
                         início e o fim do evento + tolerância (padrão 2 s)
  bocejo:                VP se um bocejo é contado nessa janela
Falsos positivos (FP): alarmes (ou bocejos) que começam fora de qualquer evento do tipo.

Saídas (pasta resultados/): resultados.csv, tabela.tex, um gráfico PNG por sessão.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np

from sonolencia import MonitorSonolencia, Parametros, parametros_blendshapes, ALARME

RAIZ = Path(__file__).resolve().parent
TIPOS = ("microcochilo", "cabeca", "bocejo")
NOME_TIPO = {"microcochilo": "Olhos fechados", "cabeca": "Queda da cabeça", "bocejo": "Bocejo"}


def metodos():
    return {
        "M1 Geométrico (EAR calibrada)": ("ear", "mar", Parametros()),
        "M2 Blendshapes (MediaPipe)": ("olho_bs", "jaw_bs", parametros_blendshapes()),
        "M3 Geométrico (EAR fixa 0,2)": ("ear", "mar", Parametros(ear_min=0.2, ear_max=0.2, ear_padrao=0.2)),
    }


# ----------------------------------------------------------------------
def carregar(caminho: Path):
    meta, linhas = {}, []
    with open(caminho, encoding="utf-8") as f:
        brutas = list(f)
    for ln in brutas:
        if ln.startswith("#"):
            for campo in ln.strip().split(","):
                if "=" in campo:
                    k, v = campo.lstrip("# ").split("=", 1)
                    meta[k] = v
    leitor = csv.DictReader([ln for ln in brutas if not ln.startswith("#")])
    for r in leitor:
        linhas.append(r)
    if not linhas:
        raise SystemExit(f"{caminho}: log vazio")
    num = lambda k: np.array([float(r[k]) if r[k] not in ("", "nan") else math.nan for r in linhas])
    d = {k: num(k) for k in ("t_s", "rosto", "ear", "mar", "olho_bs", "jaw_bs", "pitch", "lat_ms", "evento")}
    d["fase"] = np.array([r["fase"] for r in linhas])
    d["meta"] = meta
    d["nome"] = meta.get("rotulo", caminho.stem)
    return d


def eventos(d):
    """Lista de (tipo, inicio, fim) a partir das fases do roteiro."""
    out, t, fase, ev = [], d["t_s"], d["fase"], d["evento"]
    i, n = 0, len(t)
    while i < n:
        if fase[i] in TIPOS:
            j = i
            while j + 1 < n and ev[j + 1] == ev[i] and fase[j + 1] == fase[i]:
                j += 1
            fim = t[j + 1] if j + 1 < n else t[j]
            out.append((fase[i], t[i], fim))
            i = j + 1
        else:
            i += 1
    return out


def reproduzir(d, col_olho, col_boca, params):
    """Roda a máquina de estados do protótipo sobre os indicadores gravados."""
    mon = MonitorSonolencia(params)
    estados, bocejos, limiar = [], [], math.nan
    for t, r, o, b, p in zip(d["t_s"], d["rosto"], d[col_olho], d[col_boca], d["pitch"]):
        rosto = bool(r) and not math.isnan(o)
        lt = mon.atualizar(float(t), rosto, float(o), float(b), float(p))
        estados.append(lt.estado)
        bocejos.append(lt.novo_bocejo)
    return np.array(estados), np.array(bocejos, dtype=bool), mon.limiar_ear


def inicios_alarme(t, estados):
    a = estados == ALARME
    return t[np.flatnonzero(a & ~np.concatenate([[False], a[:-1]]))]


def pontuar(evs, deteccoes, tipos, tol):
    """VP/FN por tipo e FP do grupo (detecções fora de qualquer janela dos tipos)."""
    usadas, res = set(), {}
    for tipo in tipos:
        vp, atrasos, n = 0, [], 0
        for tp, ini, fim in evs:
            if tp != tipo:
                continue
            n += 1
            cand = [k for k, td in enumerate(deteccoes) if ini <= td <= fim + tol and k not in usadas]
            if cand:
                vp += 1
                usadas.add(cand[0])
                atrasos.append(deteccoes[cand[0]] - ini)
        res[tipo] = dict(n=n, VP=vp, FN=n - vp, atraso=float(np.mean(atrasos)) if atrasos else math.nan)
    janelas = [(ini, fim + tol) for tp, ini, fim in evs if tp in tipos]
    fp = [td for k, td in enumerate(deteccoes)
          if k not in usadas and not any(a <= td <= b for a, b in janelas)]
    return res, fp


def avaliar(d, tol):
    t, evs = d["t_s"], eventos(d)
    normal = d["fase"] == "normal"
    dur_normal = float(t[normal][-1] - t[normal][0]) if normal.any() else 0.0
    saida = {}
    for nome, (co, cb, params) in metodos().items():
        estados, bocejos, limiar = reproduzir(d, co, cb, params)
        alarmes = inicios_alarme(t, estados)
        r_al, fp_al = pontuar(evs, list(alarmes), ("microcochilo", "cabeca"), tol)
        r_bo, fp_bo = pontuar(evs, list(t[bocejos]), ("bocejo",), tol)
        r = {**r_al, **r_bo}
        vp = sum(r[k]["VP"] for k in TIPOS)
        n = sum(r[k]["n"] for k in TIPOS)
        fp = len(fp_al) + len(fp_bo)
        prec = vp / (vp + fp) if vp + fp else math.nan
        rev = vp / n if n else math.nan
        f1 = 2 * prec * rev / (prec + rev) if prec + rev and not math.isnan(prec) else math.nan
        fp_normal = sum(1 for td in fp_al + fp_bo if normal[min(np.searchsorted(t, td), len(t) - 1)])
        saida[nome] = dict(por_tipo=r, VP=vp, n=n, FP=fp, FP_alarme=len(fp_al), FP_bocejo=len(fp_bo),
                           FP_normal=fp_normal, dur_normal=dur_normal, precisao=prec, revocacao=rev,
                           F1=f1, limiar=limiar, estados=estados, alarmes=alarmes,
                           bocejos=t[bocejos])
    return evs, saida


# ----------------------------------------------------------------------
def grafico(d, evs, res, destino: Path, janela=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cores = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a"}
    tinta, tinta2, grade = "#262624", "#5b5a55", "#e4e2dc"
    t = d["t_s"]
    sel = np.ones_like(t, dtype=bool) if janela is None else (t >= janela[0]) & (t <= janela[1])
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": tinta2, "axes.labelcolor": tinta,
                         "xtick.color": tinta2, "ytick.color": tinta2, "font.family": "serif"})
    fig, eixos = plt.subplots(3, 1, figsize=(7.0, 4.6), sharex=True)
    m1, m2, m3 = (res[k] for k in res)
    paineis = [
        (eixos[0], d["ear"], "EAR", [(m1["limiar"], "M1", "limiar calibrado"), (0.2, "M3", "limiar fixo 0,2")], "M1"),
        (eixos[1], d["olho_bs"], "1 − eyeBlink", [(m2["limiar"], "M2", "limiar calibrado")], "M2"),
        (eixos[2], d["jaw_bs"], "jawOpen", [], "M2"),
    ]
    for ax, y, rot, limiares, chave in paineis:
        for tp, ini, fim in evs:
            if janela and (fim < janela[0] or ini > janela[1]):
                continue
            ax.axvspan(ini, fim, color="#d9d7d0", alpha=0.6, lw=0)
        ax.plot(t[sel], y[sel], color=cores[chave], lw=1.0, label=rot)
        for valor, k, txt in limiares:
            ax.axhline(valor, color=cores[k], lw=1.0, ls="--", label=txt)
        ax.set_ylabel(rot, fontsize=8)
        ax.grid(axis="y", color=grade, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    eixos[2].plot(t[sel], d["mar"][sel], color=cores["M1"], lw=1.0, label="MAR")
    eixos[2].axhline(0.6, color=cores["M1"], lw=1.0, ls="--", label="limiar MAR 0,6")
    eixos[2].axhline(0.5, color=cores["M2"], lw=1.0, ls=":", label="limiar jawOpen 0,5")
    # marcas de detecção
    for ax, k, nome in ((eixos[0], "M1", list(res)[0]), (eixos[1], "M2", list(res)[1])):
        topo = ax.get_ylim()[1]
        for ta in res[nome]["alarmes"]:
            if janela is None or janela[0] <= ta <= janela[1]:
                ax.plot(ta, topo * 0.97, marker="v", ms=4, color=tinta, ls="none")
    # rótulos dos tipos de evento no topo
    vistos = set()
    for tp, ini, fim in evs:
        if tp in vistos or (janela and (fim < janela[0] or ini > janela[1])):
            continue
        vistos.add(tp)
        eixos[0].annotate(NOME_TIPO[tp], xy=(ini, 1.02), xycoords=("data", "axes fraction"),
                          fontsize=7, color=tinta2)
    eixos[2].set_ylabel("Boca", fontsize=8)
    for ax in eixos:
        ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9, facecolor="white",
                  edgecolor="none", ncol=3)
    eixos[2].set_xlabel("Tempo (s) — faixas cinza: eventos do roteiro; ▼: início de alarme", fontsize=8)
    if janela:
        eixos[2].set_xlim(*janela)
    fig.tight_layout()
    fig.savefig(destino, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
def fmt(x, casas=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{casas}f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", help="CSVs do experimento (padrão: logs/experimento_*.csv)")
    ap.add_argument("--tolerancia", type=float, default=2.0)
    ap.add_argument("--janela", type=float, nargs=2, metavar=("INI", "FIM"),
                    help="trecho do gráfico em segundos (padrão: sessão inteira)")
    args = ap.parse_args()
    # O cmd/PowerShell do Windows não expande "*": o próprio script expande.
    padroes = args.logs or [str(RAIZ / "logs" / "experimento_*.csv")]
    caminhos = []
    for p in padroes:
        achados = sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p]
        caminhos += [Path(a) for a in achados]
    if not caminhos:
        raise SystemExit(f"Nenhum log encontrado em: {', '.join(padroes)}")
    args.logs = caminhos
    pasta = RAIZ / "resultados"
    pasta.mkdir(exist_ok=True)

    linhas_csv, linhas_tex = [], []
    for caminho in args.logs:
        d = carregar(caminho)
        evs, res = avaliar(d, args.tolerancia)
        t, lat = d["t_s"], d["lat_ms"]
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else math.nan
        rosto = np.nanmean(d["rosto"])
        print(f"\n=== {d['nome']}  (lux={d['meta'].get('lux') or '?'}) ===")
        print(f"Duração {t[-1]:.0f} s | {len(t)} quadros | {fps:.1f} FPS | latência mediana "
              f"{np.nanmedian(lat):.0f} ms (p95 {np.nanpercentile(lat, 95):.0f} ms) | rosto em {rosto:.1%} dos quadros")
        n_tipo = {k: sum(1 for e in evs if e[0] == k) for k in TIPOS}
        print("Eventos: " + ", ".join(f"{NOME_TIPO[k]} {n_tipo[k]}" for k in TIPOS))
        cab = f"{'Método':32s} {'Olhos':>7s} {'Cabeça':>7s} {'Bocejo':>7s} {'FP':>4s} {'FPnorm':>6s} {'Prec':>5s} {'Rev':>5s} {'F1':>5s} {'Atraso olhos':>12s}  limiar"
        print(cab)
        for nome, r in res.items():
            pt = r["por_tipo"]
            print(f"{nome:32s} {pt['microcochilo']['VP']:>3d}/{pt['microcochilo']['n']:<3d} "
                  f"{pt['cabeca']['VP']:>3d}/{pt['cabeca']['n']:<3d} {pt['bocejo']['VP']:>3d}/{pt['bocejo']['n']:<3d} "
                  f"{r['FP']:>4d} {r['FP_normal']:>6d} {fmt(r['precisao']):>5s} {fmt(r['revocacao']):>5s} "
                  f"{fmt(r['F1']):>5s} {fmt(pt['microcochilo']['atraso']) + ' s':>12s}  {r['limiar']:.3f}")
            linhas_csv.append(dict(sessao=d["nome"], lux=d["meta"].get("lux", ""), metodo=nome,
                                   olhos_vp=pt["microcochilo"]["VP"], olhos_n=pt["microcochilo"]["n"],
                                   cabeca_vp=pt["cabeca"]["VP"], cabeca_n=pt["cabeca"]["n"],
                                   bocejo_vp=pt["bocejo"]["VP"], bocejo_n=pt["bocejo"]["n"],
                                   fp=r["FP"], fp_alarme=r["FP_alarme"], fp_bocejo=r["FP_bocejo"],
                                   fp_normal=r["FP_normal"], precisao=r["precisao"], revocacao=r["revocacao"],
                                   f1=r["F1"], atraso_olhos_s=pt["microcochilo"]["atraso"],
                                   atraso_cabeca_s=pt["cabeca"]["atraso"], atraso_bocejo_s=pt["bocejo"]["atraso"],
                                   limiar=r["limiar"], fps=fps, lat_mediana_ms=float(np.nanmedian(lat)),
                                   lat_p95_ms=float(np.nanpercentile(lat, 95)), rosto=rosto))
            linhas_tex.append(f"{d['nome'].replace('_', ' ')} & {nome.split(' ', 1)[0]} & "
                              f"{pt['microcochilo']['VP']}/{pt['microcochilo']['n']} & "
                              f"{pt['cabeca']['VP']}/{pt['cabeca']['n']} & {pt['bocejo']['VP']}/{pt['bocejo']['n']} & "
                              f"{r['FP']} & {fmt(r['precisao'])} & {fmt(r['revocacao'])} & {fmt(r['F1'])} \\\\")
        destino = pasta / f"grafico_{d['nome']}.png"
        grafico(d, evs, res, destino, args.janela)
        print(f"Gráfico: {destino}")

    with open(pasta / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_csv[0]))
        w.writeheader()
        w.writerows(linhas_csv)
    (pasta / "tabela.tex").write_text("\n".join(linhas_tex) + "\n", encoding="utf-8")
    print(f"\nTabela: {pasta / 'resultados.csv'}  |  LaTeX: {pasta / 'tabela.tex'}")
    print("Mande a pasta 'resultados' e os arquivos de 'logs' e 'fotos' para o Claude.")


if __name__ == "__main__":
    main()
