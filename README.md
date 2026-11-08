# gender-voice-filterbank

Reconocimiento automático de género (voz masculina / femenina) a partir de un comando de voz, usando análisis de banco de filtros sobre el espectro (FFT) y comparación por diferencia absoluta contra un vector de energías guardado en JSON.

Laboratorio 2 - Procesamiento de Señales e Imágenes (Ing. de Sistemas, Universidad de los Llanos). Profesor: Fabian Velasquez Clavijo.

## Técnica

1. `y = fft(x)`, `z = |y|` (mitad útil, 0 a fs/2).
2. Por clase se promedian los espectros de todas sus grabaciones.
3. Se eligen entre 2 y 5 subbandas marcando los cortes sobre la gráfica de los espectros promedio.
4. Energía de la subbanda i: `E = (1/N_i) * sum ||C_i(k)||^2`.
5. El vector de energías de cada clase se guarda en `models/modelo.json` junto con `fs` y los cortes.
6. En tiempo real se graba el comando y se calcula su vector `E_X` con los mismos cortes.
7. Comparación por diferencia absoluta: `E_XC = sum_i |E_Xi - E_Ci|`. Gana la clase con menor `E_XC`.

Sin normalización ni filtro de media móvil: la señal pasa directo a la FFT.

## Estructura

```
gender-voice-filterbank/
├── README.md
├── requirements.txt
├── config.py
├── trainer.py
├── recognizer.py
├── app.py
├── models/
│   └── modelo.json        # generado por trainer.py
├── core/
│   ├── __init__.py
│   └── banco_filtros.py
└── grabaciones/
    ├── hombre/
    └── mujer/
```

## Instalación

```
pip install -r requirements.txt
```

## Uso

1. Grabar el banco de grabaciones en `grabaciones/hombre/` y `grabaciones/mujer/`
   con el formato `genero_Pnn_rr.wav` (ej. `hombre_P03_02.wav`).
2. Entrenar: `python trainer.py` (genera `models/modelo.json`).
3. Reconocer en vivo: `python recognizer.py`.
4. Interfaz gráfica: `python app.py`.

## Decisiones sin confirmar (TODO)

1. Número de subbandas y cortes: provisional S = 4 (3 cortes) en `config.py`.
2. fs y duración: provisional 22050 Hz y 2 s.
3. Palabra del comando.
