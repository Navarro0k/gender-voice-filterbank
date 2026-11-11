# recognizer.py - reconoce el género a partir del micrófono
import json

import numpy as np
import sounddevice as sd

import config
from trainer import vector_energias
from core.banco_filtros import (diferencias_absolutas, energia,
                               energias_subbandas, espectro, frecuencias,
                               indices_de_corte, suma_diferencias_absolutas)


def cargar_modelo(ruta=None):
    if ruta is None:
        ruta = config.RUTA_MODELO
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def grabar(fs, n_muestras):
    """Graba n_muestras del micrófono a fs Hz."""
    audio = sd.rec(n_muestras, samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    return audio.flatten()


def reconocer(modelo=None):
    """Graba, calcula E_X y compara contra cada clase.

    Devuelve (clase_ganadora, resultados) donde resultados es una lista
    de diccionarios ordenada de menor a mayor suma de diferencias.
    """
    if modelo is None:
        modelo = cargar_modelo()

    fs = modelo["fs"]
    N = modelo.get("N", int(fs * config.DURACION))
    cortes_hz = modelo["cortes_hz"]
    energias_clases = modelo["energias"]

    x = grabar(fs, N)
    z = espectro(x)

    frec = frecuencias(N, fs)
    indices = indices_de_corte(frec, cortes_hz)
    E_X = vector_energias(z, indices)

    resultados = []
    for clase, E_C in energias_clases.items():
        diffs = diferencias_absolutas(E_X, E_C)
        suma = suma_diferencias_absolutas(E_X, E_C)
        resultados.append({
            "clase": clase,
            "diferencias": [float(d) for d in diffs],
            "suma": float(suma),
        })

    resultados.sort(key=lambda r: r["suma"])
    clase_ganadora = resultados[0]["clase"]
    return clase_ganadora, resultados, E_X, indices, z, fs, N, cortes_hz


def main():
    clase, resultados, E_X, indices, z, fs, N, cortes_hz = reconocer()
    print("Género reconocido:", clase)
    print("E_X:", E_X)
    for r in resultados:
        print(r["clase"], "suma =", r["suma"], "diffs =", r["diferencias"])


if __name__ == "__main__":
    main()
