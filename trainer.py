# trainer.py - entrena el modelo y guarda models/modelo.json
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

import config
from core.banco_filtros import (energias_subbandas, espectro,
                               espectro_promedio, frecuencias,
                               indices_de_corte)


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

    # Remuestrear a FS si hace falta
    if fs != config.FS:
        n_nuevo = int(len(data) * config.FS / fs)
        data = resample(data, n_nuevo)

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
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud |X(k)|")
    plt.title("Espectros promedio por clase")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Calcular índices de corte y vectores de energía
    indices = indices_de_corte(frec, config.CORTES_HZ)
    energias = {}
    for clase in config.CLASES:
        energias[clase] = energias_subbandas(promedios[clase], indices)

    # Guardar JSON
    os.makedirs(os.path.dirname(config.RUTA_MODELO), exist_ok=True)
    modelo = {
        "fs": config.FS,
        "N": config.N,
        "cortes_hz": list(config.CORTES_HZ),
        "energias": energias,
    }
    with open(config.RUTA_MODELO, "w", encoding="utf-8") as f:
        json.dump(modelo, f, indent=2, ensure_ascii=False)

    print("Modelo guardado en", config.RUTA_MODELO)
    print("Indices de corte:", indices)
    for clase in config.CLASES:
        print(clase, energias[clase])


if __name__ == "__main__":
    main()
