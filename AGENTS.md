# AGENTS.md

Contexto persistente del proyecto para agentes de código. El orden de desarrollo está en `PLAN.md`.

## Proyecto

**gender-voice-filterbank**: sistema en Python que reconoce el género (voz hombre / voz mujer) a partir de un comando de voz, con análisis de banco de filtros sobre el espectro (FFT). Laboratorio 2 de Procesamiento de Señales e Imágenes (Ing. de Sistemas, Universidad de los Llanos). Profesor: Fabian Velasquez Clavijo.

Requisitos: error máximo del 5%, entre 2 y 5 subbandas, captura en tiempo real con el micrófono y banco de grabaciones muy diverso (muchas personas).

## Técnica del profesor (explicada en clase)

1. `y = fft(x)`, `z = |y|` → ||X(k)||, solo la mitad útil (0 a fs/2).
2. Por clase (hombre, mujer) se promedian los espectros de todas sus grabaciones: ||C(k)||_m.
3. Se eligen S de 2 a 5 subbandas marcando S−1 frecuencias de corte sobre la gráfica de los espectros promedio. Los cortes se convierten a índices k.
4. Energía de la subbanda i de la clase: E_Ci = (1/N_i) · Σ ||C_i(k)||_m², con N_i el número de puntos de la subbanda.
5. El vector [E_C1 ... E_CS] de cada clase se **guarda en un JSON** dentro del proyecto, con fs y los cortes.
6. En tiempo real se graba el comando y se calcula su vector E_X con los mismos cortes.
7. Comparación por **diferencia absoluta**: E_XC = Σ_i |E_Xi − E_Ci| contra cada clase. Gana la clase con menor E_XC.

Formato del JSON:

```
{ "fs": ..., "cortes_hz": [...], "energias": { "hombre": [...], "mujer": [...] } }
```

## Sin preprocesado

Por decisión del usuario **no hay normalización ni filtro de media móvil**: la señal grabada pasa directo a la FFT. No agregarlos. Al leer un `.wav` solo se hace lo necesario para que coincida con la grabación en vivo: pasar a mono, escalar enteros a [−1, 1] (dividir por 32768 si es int16), remuestrear a `FS` si hace falta y recortar o rellenar con ceros hasta N muestras.

## Arquitectura

```
config.py            FS, DURACION, N, N_CORTES, cortes iniciales, rutas
trainer.py           lee .wav, promedia espectros por clase, guarda el JSON
recognizer.py        graba del micrófono, calcula E_X y compara con el JSON
evaluar.py           mide el error (%) y la matriz de confusión
app.py               interfaz gráfica (CustomTkinter + matplotlib)
models/              modelo.json (generado)
core/
  banco_filtros.py   fórmulas del profesor (solo numpy)
grabaciones/
  hombre/  mujer/
```

No crear archivos ni carpetas fuera de esta estructura sin preguntar.

## Reglas de `core/`

- Solo depende de `numpy`; la FFT con `np.fft.rfft`.
- Sin I/O: no lee archivos, no usa el micrófono, no grafica, no importa `config.py` ni los scripts de la raíz. Recibe parámetros y arreglos, devuelve resultados.
- Cada función lleva un docstring con la fórmula del profesor que implementa.
- Funciones cortas: el núcleo va en el informe (máximo media página) y se sustenta en vivo.
- Funciones esperadas: `espectro`, `espectro_promedio`, `frecuencias`, `indices_de_corte`, `energia`, `energias_subbandas`, `diferencias_absolutas`, `suma_diferencias_absolutas`.

Lo que queda fuera de `core/` (lectura de `.wav`, remuestreo, grabación, JSON, gráficas, interfaz) vive en la raíz, donde sí se permiten `scipy`, `sounddevice`, `matplotlib` y `customtkinter`.

## Decisiones sin confirmar

Cada una vive en un solo lugar, marcada con `# TODO confirmar`:

1. **Número de subbandas y cortes.** Provisional: S = 4 (3 cortes), elegidos sobre la gráfica de los espectros promedio donde hombre y mujer más se diferencian. Entre 2 y 5.
2. **fs y duración.** Provisional: `FS = 22050`, `DURACION = 2` s. Entrenamiento y tiempo real deben tener la misma fs y el mismo N, para que los índices de corte coincidan.
3. **Palabra del comando.**

No resolverlas por cuenta propia ni presentarlas como definitivas.

## Riesgo conocido

Sin normalización, la energía depende del volumen y de la distancia al micrófono. Grabar el banco y probar en tiempo real con el mismo micrófono y ganancia, y a una distancia parecida. Si `evaluar.py` muestra un error alto por esta causa, avisar al usuario en lugar de agregar normalización por cuenta propia.

## Convenciones

- Comentarios, docstrings y mensajes al usuario en **español**.
- Constantes en `config.py` en mayúsculas.
- Grabaciones: `genero_Pnn_rr.wav` (por ejemplo `hombre_P03_02.wav`), con Pnn la persona y rr la repetición.
- Al evaluar, separar por **persona**, no por grabación.

## Comandos

```
pip install -r requirements.txt
python trainer.py
python evaluar.py
python app.py
```

Dependencias permitidas: `numpy`, `scipy`, `sounddevice`, `matplotlib`, `customtkinter`. No agregar otras sin preguntar.

## Forma de trabajar

- Seguir el orden de `PLAN.md`; al terminar cada paso, detenerse y resumir antes de continuar.
- Si algo es ambiguo, preguntar en vez de inventar.
- Cambios pequeños y enfocados; no refactorizar ni "mejorar" la arquitectura.

## Criterios de aceptación

- `core/` solo importa `numpy` y sigue las fórmulas del profesor, cada función con su docstring.
- Entre 2 y 5 subbandas.
- El vector de cada clase se guarda en JSON y `recognizer.py` lo carga; no se reentrena al reconocer.
- Comparación por diferencias absolutas; gana la menor suma.
- Misma fs y mismo N en entrenamiento y tiempo real.
- Error ≤ 5% sobre personas no vistas en entrenamiento.
- `app.py` reconoce el género en vivo con el micrófono.