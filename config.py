# config.py - constantes del proyecto
# Las decisiones sin confirmar están marcadas con TODO confirmar.

# --- Audio (TODO confirmar) ---
FS = 22050            # frecuencia de muestreo en Hz
DURACION = 2           # duración de la grabación en segundos
N = int(FS * DURACION) # número de muestras

# --- Banco de filtros (TODO confirmar) ---
# S = número de subbandas (entre 2 y 5). N_CORTES = S - 1.
N_CORTES = 3           # S = 4 subbandas
# Frecuencias de corte en Hz. El corte en 160 Hz separa la frecuencia
# fundamental típica de hombre (85-180 Hz) de la de mujer (165-255 Hz).
# Elegidos con validación por persona sobre el banco de grabaciones.
CORTES_HZ = [80, 160, 6000]

# --- Clases y rutas ---
CLASES = ["hombre", "mujer"]
RUTA_GRABACIONES = "grabaciones"
RUTA_MODELO = "models/modelo.json"
