"""Núcleo matemático del reconocimiento de género por banco de filtros.

Solo depende de numpy. Implementa las fórmulas del profesor:
FFT, espectro promedio, índices de corte, energía por subbanda y
diferencia absoluta. Sin I/O, sin micrófono, sin gráficas.
"""

import numpy as np


def espectro(x):
    """Calcula ||X(k)||, la mitad útil del espectro.

    Fórmula del profesor:
        y = fft(x)
        z = |y|
    Se devuelve solo la mitad útil (0 a fs/2) usando rfft.
    """
    x = np.asarray(x)
    y = np.fft.rfft(x)
    z = np.abs(y)
    return z


def espectro_promedio(lista):
    """Promedia los espectros de una clase: ||C(k)||_m.

    Fórmula del profesor:
        ||C(k)||_m = (1/M) * sum_{m=1}^{M} ||X_m(k)||
    donde M es el número de grabaciones de la clase.
    """
    lista = [np.asarray(e) for e in lista]
    if not lista:
        return np.array([])
    return np.mean(np.array(lista), axis=0)


def frecuencias(N, fs):
    """Frecuencias correspondientes a los bins de rfft.

    f(k) = k * fs / N, para k = 0 ... N/2.
    """
    return np.fft.rfftfreq(N, d=1.0 / fs)


def indices_de_corte(frec, cortes_hz):
    """Convierte frecuencias de corte (Hz) a índices k.

    Para cada frecuencia de corte se busca el índice del bin más cercano.
    Devuelve los índices ordenados y sin duplicados.
    """
    frec = np.asarray(frec)
    indices = []
    for fc in cortes_hz:
        idx = int(np.argmin(np.abs(frec - fc)))
        indices.append(idx)
    indices = sorted(set(indices))
    # Evitar índices en los extremos para no crear subbandas vacías.
    indices = [i for i in indices if 0 < i < len(frec)]
    return indices


def energia(subbanda):
    """Energía de una subbanda.

    Fórmula del profesor:
        E = (1 / N_i) * sum ||C_i(k)||^2
    donde N_i es el número de puntos de la subbanda.
    """
    subbanda = np.asarray(subbanda)
    N_i = subbanda.size
    if N_i == 0:
        return 0.0
    return float(np.sum(subbanda ** 2) / N_i)


def energias_subbandas(z, indices):
    """Calcula [E_1 ... E_S] para S subbandas.

    Los índices de corte dividen el espectro en subbandas contiguas que
    cubren todo el espectro sin traslapes:
        subbanda 1: z[0 : idx_1]
        subbanda j: z[idx_{j-1} : idx_j]
        subbanda S: z[idx_{S-1} : fin]
    """
    z = np.asarray(z)
    cortes = sorted(indices)
    limites = [0] + cortes + [len(z)]
    resultado = []
    for i in range(len(limites) - 1):
        a = limites[i]
        b = limites[i + 1]
        resultado.append(energia(z[a:b]))
    return resultado


def diferencias_absolutas(E_X, E_C):
    """Diferencia absoluta por subbanda: |E_Xi - E_Ci|."""
    E_X = np.asarray(E_X, dtype=float)
    E_C = np.asarray(E_C, dtype=float)
    return np.abs(E_X - E_C)


def suma_diferencias_absolutas(E_X, E_C):
    """Suma de las diferencias absolutas: E_XC = sum_i |E_Xi - E_Ci|."""
    return float(np.sum(diferencias_absolutas(E_X, E_C)))
