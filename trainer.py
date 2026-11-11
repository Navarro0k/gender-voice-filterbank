import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import config
from core.banco_filtros import (energias_subbandas, espectro,
                               espectro_promedio, frecuencias,
                               indices_de_corte)


def vector_energias(z, indices):
    E = np.asarray(energias_subbandas(z, indices), dtype=float)
    total = E.sum()
    if total <= 0:
        return E
    return E / total


def vectores_individuales_por_clase(datos, indices):
    """Devuelve {clase: arreglo(n_audios, n_subbandas)} con los vectores de cada audio."""
    resultado = {}
    for clase in config.CLASES:
        lista = [vector_energias(z, indices) for z in datos[clase]]
        resultado[clase] = np.array(lista, dtype=float)
    return resultado


def vectores_por_clase(datos, indices):
    individuales = vectores_individuales_por_clase(datos, indices)
    vectores = {}
    for clase in config.CLASES:
        vectores[clase] = [float(v) for v in np.mean(individuales[clase], axis=0)]
    return vectores


def estadisticas_por_clase(datos, indices):
    """Calcula media, desviación estándar y varianza por clase y subbanda."""
    individuales = vectores_individuales_por_clase(datos, indices)
    energias = {}
    estadisticas = {}
    for clase in config.CLASES:
        arr = individuales[clase]
        energias[clase] = [float(v) for v in np.mean(arr, axis=0)]
        estadisticas[clase] = {
            "n": int(arr.shape[0]),
            "desviacion_estandar": [float(v) for v in np.std(arr, axis=0, ddof=0)],
            "varianza": [float(v) for v in np.var(arr, axis=0, ddof=0)],
        }
    return energias, estadisticas


def leer_wav(ruta):
    """Lee un .wav y lo deja listo para la FFT (mono, a FS, longitud N)."""
    fs, data = wavfile.read(ruta)

    # Pasar a mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Escalar enteros a [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float) / 32768.0
    else:
        data = data.astype(float)

    # Se asume que la grabación ya está a FS (no se remuestrea)
    if fs != config.FS:
        raise ValueError(
            "La grabación %s tiene fs=%d Hz, se esperaba %d Hz"
            % (ruta, fs, config.FS)
        )

    # Recortar o rellenar con ceros hasta N muestras
    if len(data) > config.N:
        data = data[:config.N]
    elif len(data) < config.N:
        data = np.pad(data, (0, config.N - len(data)))

    return data


def espectros_por_clase():
    """Devuelve {clase: [espectros]} leyendo grabaciones/<clase>/*.wav."""
    resultado = {}
    for clase in config.CLASES:
        carpeta = os.path.join(config.RUTA_GRABACIONES, clase)
        espectros = []
        if os.path.isdir(carpeta):
            for nombre in sorted(os.listdir(carpeta)):
                if nombre.lower().endswith(".wav"):
                    ruta = os.path.join(carpeta, nombre)
                    x = leer_wav(ruta)
                    espectros.append(espectro(x))
        resultado[clase] = espectros
    return resultado


def main():
    datos = espectros_por_clase()

    # Promediar espectros por clase
    promedios = {}
    for clase in config.CLASES:
        promedios[clase] = espectro_promedio(datos[clase])

    # Graficar los espectros promedio para elegir los cortes a ojo
    frec = frecuencias(config.N, config.FS)
    plt.figure("Espectros promedio")
    for clase in config.CLASES:
        plt.plot(frec, promedios[clase], label=clase)
    for fc in config.CORTES_HZ:
        plt.axvline(fc, color="gray", linestyle="--")
    plt.xscale("log")
    plt.xlim(40, config.FS / 2)
    plt.xlabel("Frecuencia (Hz, escala logarítmica)")
    plt.ylabel("Magnitud |X(k)|")
    plt.title("Espectros promedio por clase")
    plt.legend()
    plt.grid(True)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close("all")

    # Calcular índices de corte, vectores de energía y estadísticas
    indices = indices_de_corte(frec, config.CORTES_HZ)
    energias, estadisticas = estadisticas_por_clase(datos, indices)

    # Guardar JSON
    os.makedirs(os.path.dirname(config.RUTA_MODELO), exist_ok=True)
    modelo = {
        "fs": config.FS,
        "N": config.N,
        "cortes_hz": list(config.CORTES_HZ),
        "energias": energias,
        "estadisticas": estadisticas,
    }
    with open(config.RUTA_MODELO, "w", encoding="utf-8") as f:
        json.dump(modelo, f, indent=2, ensure_ascii=False)

    print("Modelo guardado en", config.RUTA_MODELO)
    print("Indices de corte:", indices)
    for clase in config.CLASES:
        print(clase, "energias (media):", energias[clase])
        print(clase, "desviacion estandar:", estadisticas[clase]["desviacion_estandar"])
        print(clase, "varianza:", estadisticas[clase]["varianza"])
        print(clase, "n audios:", estadisticas[clase]["n"])


if __name__ == "__main__":
    main()
