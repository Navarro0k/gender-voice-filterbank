# config.py - constantes del proyecto
# Las decisiones sin confirmar están marcadas con TODO confirmar.

# --- Audio (TODO confirmar) ---
FS = 22050            # frecuencia de muestreo en Hz
DURACION = 2           # duración de la grabación en segundos
N = int(FS * DURACION) # número de muestras

# --- Banco de filtros (TODO confirmar) ---
# S = número de subbandas (entre 2 y 5). N_CORTES = S - 1.
N_CORTES = 3           # S = 4 subbandas
# Frecuencias de corte provisionales en Hz (se ajustan sobre la gráfica)
CORTES_HZ = [1000, 3000, 6000]   # TODO confirmar

# --- Clases y rutas ---
CLASES = ["hombre", "mujer"]
RUTA_GRABACIONES = "grabaciones"
RUTA_MODELO = "models/modelo.json"
