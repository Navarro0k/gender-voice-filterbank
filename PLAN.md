# PLAN: gender-voice-filterbank

Reconocimiento de género (voz hombre / voz mujer) por comando de voz: FFT, energía por subbanda y comparación por diferencia absoluta contra un vector guardado en JSON.

Leer primero `AGENTS.md`: técnica del profesor, reglas de `core/` y decisiones sin confirmar.

**Descripción del repo (GitHub):** Reconocimiento de género (voz masculina/femenina) con análisis de banco de filtros: FFT, energía por subbanda y comparación por diferencia absoluta contra un vector de energías guardado en JSON. Núcleo matemático separado en `core/`. Laboratorio 2, Procesamiento de Señales e Imágenes.

## Estructura final

```
gender-voice-filterbank/
├── README.md
├── requirements.txt
├── config.py
├── trainer.py
├── recognizer.py
├── evaluar.py
├── app.py
├── models/
│   └── modelo.json            # generado por trainer.py
├── core/
│   ├── __init__.py
│   └── banco_filtros.py
└── grabaciones/
    ├── hombre/
    └── mujer/
```

## Pasos (en orden; detenerse y resumir al terminar cada uno)

### 1. `config.py` y `core/banco_filtros.py`

- `config.py`: `FS`, `DURACION`, `N = int(FS * DURACION)`, `N_CORTES` (S − 1, con S entre 2 y 5), cortes iniciales en Hz, rutas y clases `["hombre", "mujer"]`.
- `core/banco_filtros.py`:
  - `espectro(x)`: y = fft(x), z = |y|, mitad útil.
  - `espectro_promedio(lista)`: promedio de los espectros de una clase.
  - `frecuencias(N, fs)` e `indices_de_corte(frec, cortes_hz)`.
  - `energia(subbanda)`: E = (1/N_i) · Σ ||C_i(k)||².
  - `energias_subbandas(z, indices)`: devuelve [E_1 ... E_S].
  - `diferencias_absolutas(E_X, E_C)` y `suma_diferencias_absolutas(E_X, E_C)`.

Verificación: con un ejemplo pequeño hecho a mano, comprobar que las subbandas cubren todo el espectro sin traslapes. Script desechable, no se commitea.

### 2. `trainer.py`

- Leer los `.wav` de `grabaciones/<clase>/` (mono, a `FS`, longitud N) y calcular el espectro de cada uno.
- Promediar los espectros de cada clase.
- Graficar los dos espectros promedio para elegir los cortes a ojo.
- Con los cortes de `config.py`, calcular el vector de energías de cada clase y guardar `models/modelo.json`.
- Se ejecuta con `python trainer.py`.

### 3. `recognizer.py`

- Cargar `models/modelo.json`.
- Grabar `DURACION` segundos del micrófono a `FS`.
- Calcular E_X con los cortes del JSON y las diferencias absolutas contra cada clase.
- Devolver `(clase, diferencias, suma)` ordenado de menor a mayor suma; la primera es el género reconocido.

### 4. `evaluar.py`

- Separar por persona (entrenamiento / prueba, o dejando una persona fuera cada vez).
- Entrenar solo con las de entrenamiento y reconocer las de prueba con la misma lógica de `recognizer.py`.
- Imprimir el error (%) y la matriz de confusión.
- **Meta: error ≤ 5%.** Si no se cumple, ajustar primero los cortes y el número de subbandas, luego ampliar el banco de grabaciones.

### 5. `app.py`

Dos pestañas:

- **Entrenamiento:** cargar espectros promedio, marcar los cortes con clic sobre la gráfica, entrenar y guardar el JSON, mostrar la tabla de vectores por clase.
- **Reconocimiento:** botón para grabar, espectro de la señal con los cortes, tabla de diferencias por subbanda y clase ganadora resaltada.

Va al final: la interfaz vale el 30% de la nota, pero la lógica (70%) debe estar probada antes.

### 6. `README.md` y `requirements.txt`

- README: descripción, técnica en pocas líneas, cómo grabar, entrenar, evaluar y usar la app.
- `requirements.txt`: `numpy`, `scipy`, `sounddevice`, `matplotlib`, `customtkinter`.

## Banco de grabaciones

- Muchas personas distintas, hombres y mujeres, de distintas edades; la misma palabra, varias repeticiones por persona.
- Nombrar como `hombre_P03_02.wav`.
- Como no hay normalización, grabar con el mismo micrófono, ganancia y distancia aproximada que se usarán en tiempo real.

## Decisiones sin confirmar (preguntar al usuario)

1. Número de subbandas y cortes. Provisional: S = 4.
2. fs y duración. Provisional: 22050 Hz y 2 s.
3. Palabra del comando.

## Criterios de aceptación

- `core/` solo importa `numpy` y sigue las fórmulas del profesor.
- Entre 2 y 5 subbandas; modelo guardado en JSON y cargado al reconocer.
- Comparación por diferencia absoluta; gana la menor suma.
- Misma fs y mismo N en entrenamiento y tiempo real.
- Sin normalización ni media móvil.
- Error ≤ 5% sobre personas no vistas en entrenamiento.
- `app.py` reconoce el género en vivo con el micrófono.
- La estructura de archivos coincide con la de este documento, sin archivos extra.