from .logica import (MonitorSonolencia, Parametros, Leitura, ear, ear_medio, mar,
                     angulos_euler, NORMAL, ATENCAO, ALARME, SEM_ROSTO, CALIBRANDO)


def parametros_blendshapes() -> Parametros:
    """Parâmetros do método alternativo (blendshapes do MediaPipe).

    Nesse método a "abertura do olho" é 1 - eyeBlink (aberto ~0,8-1,0;
    fechado ~0-0,4) e a "abertura da boca" é o escore jawOpen (0-1).
    Valores iniciais, a ajustar nos experimentos.
    """
    return Parametros(fator_ear=0.6, ear_min=0.35, ear_max=0.65, ear_padrao=0.5,
                      mar_bocejo=0.5)
