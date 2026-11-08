# evaluar.py - evalúa el error (%) (por persona)
import json
import os
import re

import numpy as np

import config
import trainer
from core.banco_filtros import (energias_subbandas, espectro,
                               espectro_promedio, frecuencias,
                               indices_de_corte)


def personas_por_clase():
    """Agrupa archivos .wav por persona. Devuelve {clase: {persona: [rutas]}}."""
    grupos = {}
    patron = re.compile(r"^[^_]+_(P\d+)_", re.IGNORECASE)
    for clase in config.CLASES:
        carpeta = os.path.join(config.RUTA_GRABACIONES, clase)
        personas = {}
        if os.path.isdir(carpeta):
            for nombre in os.listdir(carpeta):
                if not nombre.lower().endswith(".wav"):
                    continue
                m = patron.match(nombre)
                persona = m.group(1) if m else "P00"
                personas.setdefault(persona, []).append(os.path.join(carpeta, nombre))
        grupos[clase] = personas
    return grupos


def entrenar_con_archivos(archivos_por_clase, cortes_hz):
    """Entrena (promedia espectros) con las listas de archivos dadas."""
    promedios = {}
    for clase in config.CLASES:
        espectros = []
        for ruta in archivos_por_clase.get(clase, []):
            x = trainer.leer_wav(ruta)
            espectros.append(espectro(x))
        promedios[clase] = espectro_promedio(espectros)
    frec = frecuencias(config.N, config.FS)
    indices = indices_de_corte(frec, cortes_hz)
    energias = {}
    for clase in config.CLASES:
        energias[clase] = energias_subbandas(promedios[clase], indices)
    return energias, indices


def clasificar(x, energias, indices):
    """Predice la clase de una señal x."""
    z = espectro(x)
    E_X = energias_subbandas(z, indices)
    mejor = None
    mejor_suma = None
    for clase, E_C in energias.items():
        from core.banco_filtros import suma_diferencias_absolutas
        s = suma_diferencias_absolutas(E_X, E_C)
        if mejor_suma is None or s < mejor_suma:
            mejor_suma = s
            mejor = clase
    return mejor


def main():
    grupos = personas_por_clase()

    # Lista de todas las personas (clase, persona)
    personas = []
    for clase in config.CLASES:
        for persona in grupos[clase]:
            personas.append((clase, persona))

    if not personas:
        print("No hay grabaciones en", config.RUTA_GRABACIONES)
        return

    # Matriz de confusión (no requerida)
    total = 0
    correctos = 0

    cortes_hz = config.CORTES_HZ

    for clase_test, persona_test in personas:
        # Archivos de entrenamiento: todas las personas excepto la de prueba
        train_files = {c: [] for c in config.CLASES}
        for clase in config.CLASES:
            for persona, archivos in grupos[clase].items():
                if clase == clase_test and persona == persona_test:
                    continue
                train_files[clase].extend(archivos)

        # Si no queda entrenamiento para alguna clase, saltar
        if any(len(train_files[c]) == 0 for c in config.CLASES):
            continue

        energias, indices = entrenar_con_archivos(train_files, cortes_hz)

        # Probar todas las grabaciones de la persona dejada fuera
        for ruta in grupos[clase_test][persona_test]:
            x = trainer.leer_wav(ruta)
            pred = clasificar(x, energias, indices)
            total += 1
            if pred == clase_test:
                correctos += 1

    if total == 0:
        print("No se pudo evaluar (faltan personas o grabaciones).")
        return

    error = (total - correctos) / total * 100.0
    print("Total de pruebas:", total)
    print("Correctos:", correctos)
    print("Error: %.2f %%" % error)


if __name__ == "__main__":
    main()
