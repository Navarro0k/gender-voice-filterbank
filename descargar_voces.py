# descargar_voces.py - prepara un banco de voces (una palabra) desde MSWC
#
# Uso:
#   pip install datasets scipy numpy
#   python descargar_voces.py --palabra hola --max-por-genero 100
#
# Descarga audios de una sola palabra en español, separa por género
# (hombre/mujer) y guarda WAV mono a FS (config.py) en grabaciones/.

import argparse
import os
import random
import shutil

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--palabra", default="hola", help="palabra a descargar")
    parser.add_argument("--max-por-genero", type=int, default=100,
                        help="máximo de audios por género")
    parser.add_argument("--salida", default=config.RUTA_GRABACIONES,
                        help="carpeta de salida")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Falta la librería 'datasets'. Instálala con: pip install datasets")
        return

    palabra = args.palabra.strip().lower()
    print("Descargando subset español de MSWC...")
    try:
        ds = load_dataset("MLCommons/ml_spoken_words", "es_wav",
                          split="train+validation+test", streaming=True)
    except Exception:
        ds = load_dataset("MLCommons/ml_spoken_words", "es_wav",
                          split="train", streaming=True)

    # gender en MSWC: 0=MALE, 1=FEMALE (según la tarjeta del dataset)
    male = []
    female = []
    target = args.max_por_genero

    print("Filtrando palabra '%s'..." % palabra)
    for ex in ds:
        if not ex.get("is_valid", True):
            continue
        if str(ex.get("keyword", "")).strip().lower() != palabra:
            continue
        g = ex.get("gender")
        audio = ex.get("audio")
        if audio is None:
            continue
        speaker = ex.get("speaker_id", "NA")

        item = {
            "audio": np.asarray(audio["array"], dtype=np.float32),
            "sr": audio["sampling_rate"],
            "speaker": speaker,
        }

        if g == 0 and len(male) < target:
            male.append(item)
        elif g == 1 and len(female) < target:
            female.append(item)

        if len(male) >= target and len(female) >= target:
            break

    print("Encontrados: hombres=%d, mujeres=%d" % (len(male), len(female)))
    n = min(len(male), len(female))
    if n == 0:
        print("No se encontraron audios para la palabra '%s'." % palabra)
        return

    random.seed(42)
    random.shuffle(male)
    random.shuffle(female)
    male = male[:n]
    female = female[:n]

    # Preparar carpetas (limpiar solo .wav previos de la palabra)
    for genero in ("hombre", "mujer"):
        carpeta = os.path.join(args.salida, genero)
        os.makedirs(carpeta, exist_ok=True)

    def guardar(lista, genero):
        # Asignar número de persona por speaker único
        speakers = {}
        contador_persona = 0
        rep_por_persona = {}
        for item in lista:
            sp = item["speaker"]
            if sp not in speakers:
                contador_persona += 1
                speakers[sp] = contador_persona
            persona = speakers[sp]
            rep_por_persona.setdefault(persona, 0)
            rep_por_persona[persona] += 1
            rep = rep_por_persona[persona]

            audio = item["audio"]
            sr = item["sr"]
            # Remuestrear a config.FS si hace falta
            if sr != config.FS:
                n_nuevo = int(len(audio) * config.FS / sr)
                audio = resample(audio, n_nuevo)
            # Normalizar y convertir a int16
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            audio_i16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

            # Recortar/rellenar a N
            if len(audio_i16) > config.N:
                audio_i16 = audio_i16[:config.N]
            elif len(audio_i16) < config.N:
                audio_i16 = np.pad(audio_i16, (0, config.N - len(audio_i16)))

            nombre = "%s_P%02d_%02d.wav" % (genero, persona, rep)
            ruta = os.path.join(args.salida, genero, nombre)
            wavfile.write(ruta, config.FS, audio_i16)

    guardar(male, "hombre")
    guardar(female, "mujer")

    print("Listo: %d audios por género en '%s'." % (n, args.salida))


if __name__ == "__main__":
    main()
