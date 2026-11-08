# descargar_voces.py - prepara un banco de voces (una palabra) desde MSWC
#
# Uso:
#   pip install soundfile scipy numpy
#   python descargar_voces.py --palabra tiempo --max-por-genero 100
#
# Descarga los splits (pequeños) y el audio (grande, ~6.8 GB) de MSWC
# en español, filtra una palabra, equilibra hombres/mujeres y guarda
# WAV mono a FS (config.py) en grabaciones/.

import argparse
import csv
import os
import random
import tarfile
import urllib.request

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample

import config

SPLITS_URL = "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com/splits/es.tar.gz"
AUDIO_URL = "https://mlc-datasets.oss-cn-guangzhou.aliyuncs.com/audio/es.tar.gz"


def descargar(url, destino):
    if os.path.exists(destino):
        print("Ya existe:", destino)
        return
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    print("Descargando", url)
    urllib.request.urlretrieve(url, destino)
    print("Descargado:", destino)


def asegurar_splits(splits_dir):
    os.makedirs(splits_dir, exist_ok=True)
    csvs = ["es_train.csv", "es_dev.csv", "es_test.csv"]
    if all(os.path.exists(os.path.join(splits_dir, c)) for c in csvs):
        return
    tar_path = os.path.join(splits_dir, "es_splits.tar.gz")
    descargar(SPLITS_URL, tar_path)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(splits_dir)
    print("Splits extraídos en", splits_dir)


def recolectar_candidatos(splits_dir, palabra):
    palabra = palabra.strip().lower()
    csvs = ["es_train.csv", "es_dev.csv", "es_test.csv"]
    male = []
    female = []
    for nombre in csvs:
        ruta = os.path.join(splits_dir, nombre)
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("VALID", "").strip().lower() != "true":
                    continue
                if row.get("WORD", "").strip().lower() != palabra:
                    continue
                gender = row.get("GENDER", "").strip().upper()
                link = row.get("LINK", "").strip()
                speaker = row.get("SPEAKER", "").strip()
                if gender == "MALE":
                    male.append((link, speaker))
                elif gender == "FEMALE":
                    female.append((link, speaker))
    return male, female


def extraer_necesarios(audio_tar, extract_dir, needed_links):
    os.makedirs(extract_dir, exist_ok=True)
    needed_members = set("es/clips/" + link for link in needed_links)
    found = set()
    # Si ya están extraídos, no repetir
    pendientes = [m for m in needed_members
                  if not os.path.exists(os.path.join(extract_dir, m))]
    if not pendientes:
        print("Audios ya extraídos.")
        return
    print("Extrayendo %d archivos del tar..." % len(pendientes))
    pendientes_set = set(pendientes)
    with tarfile.open(audio_tar, "r:*") as tar:
        for member in tar:
            if member.name in pendientes_set:
                tar.extract(member, extract_dir)
                found.add(member.name)
                if len(found) >= len(pendientes_set):
                    break
    print("Extraídos:", len(found))


def guardar(lista, genero, extract_dir, salida):
    import soundfile as sf

    speakers = {}
    contador_persona = 0
    rep_por_persona = {}
    for link, speaker in lista:
        if speaker not in speakers:
            contador_persona += 1
            speakers[speaker] = contador_persona
        persona = speakers[speaker]
        rep_por_persona.setdefault(persona, 0)
        rep_por_persona[persona] += 1
        rep = rep_por_persona[persona]

        opus_path = os.path.join(extract_dir, "es", "clips", link)
        if not os.path.exists(opus_path):
            print("Falta archivo:", opus_path)
            continue

        audio, sr = sf.read(opus_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != config.FS:
            n_nuevo = int(len(audio) * config.FS / sr)
            audio = resample(audio, n_nuevo)

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        audio_i16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)

        if len(audio_i16) > config.N:
            audio_i16 = audio_i16[:config.N]
        elif len(audio_i16) < config.N:
            audio_i16 = np.pad(audio_i16, (0, config.N - len(audio_i16)))

        carpeta = os.path.join(salida, genero)
        os.makedirs(carpeta, exist_ok=True)
        nombre = "%s_P%02d_%02d.wav" % (genero, persona, rep)
        wavfile.write(os.path.join(carpeta, nombre), config.FS, audio_i16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--palabra", default="tiempo")
    parser.add_argument("--max-por-genero", type=int, default=100)
    parser.add_argument("--salida", default=config.RUTA_GRABACIONES)
    parser.add_argument("--splits-dir", default=os.path.join(
        os.path.expanduser("~"), ".cache", "mswc_splits"))
    parser.add_argument("--audio-tar", default=os.path.join(
        os.path.expanduser("~"), ".cache", "mswc_es_audio.tar.gz"))
    args = parser.parse_args()

    try:
        import soundfile
    except ImportError:
        print("Falta 'soundfile'. Instálala con: pip install soundfile")
        return

    asegurar_splits(args.splits_dir)
    male, female = recolectar_candidatos(args.splits_dir, args.palabra)
    print("Encontrados: hombres=%d, mujeres=%d" % (len(male), len(female)))

    n = min(len(male), len(female), args.max_por_genero)
    if n == 0:
        print("No hay suficientes datos para '%s'." % args.palabra)
        return

    random.seed(42)
    random.shuffle(male)
    random.shuffle(female)
    male = male[:n]
    female = female[:n]

    needed_links = [item[0] for item in male + female]
    descargar(AUDIO_URL, args.audio_tar)
    extract_dir = os.path.join(os.path.dirname(args.audio_tar), "mswc_es_extracted")
    extraer_necesarios(args.audio_tar, extract_dir, needed_links)

    guardar(male, "hombre", extract_dir, args.salida)
    guardar(female, "mujer", extract_dir, args.salida)

    print("Listo: %d audios por género en '%s'." % (n, args.salida))


if __name__ == "__main__":
    main()