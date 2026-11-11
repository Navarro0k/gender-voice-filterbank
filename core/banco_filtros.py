import numpy as np


def espectro(x):
    x = np.asarray(x)
    y = np.fft.rfft(x)
    z = np.abs(y)
    return z


def espectro_promedio(lista):
    lista = [np.asarray(e) for e in lista]
    if not lista:
        return np.array([])
    return np.mean(np.array(lista), axis=0)


def frecuencias(N, fs):
    return np.fft.rfftfreq(N, d=1.0 / fs)


def indices_de_corte(frec, cortes_hz):
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
    subbanda = np.asarray(subbanda)
    N_i = subbanda.size
    if N_i == 0:
        return 0.0
    return float(np.sum(subbanda ** 2) / N_i)


def energias_subbandas(z, indices):
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
    E_X = np.asarray(E_X, dtype=float)
    E_C = np.asarray(E_C, dtype=float)
    return np.abs(E_X - E_C)


def suma_diferencias_absolutas(E_X, E_C):
    return float(np.sum(diferencias_absolutas(E_X, E_C)))
