"""Constantes y rutas del proyecto. Contrato: docs/CONTRATOS_DATOS.md §1.

Este archivo es de SOLO LECTURA para los subagentes; cualquier cambio lo hace el orquestador.
"""
from pathlib import Path

# ----------------------------------------------------------------- rutas
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
for _d in (INTERIM, PROCESSED, MODELS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Archivos crudos
RAW_SGC_ANTIGUO = RAW / "sgc_1993_2018.csv"
RAW_SGC_NUEVO = RAW / "sgc_2018_hoy.csv"
RAW_DIVIPOLA = RAW / "dane_divipola.csv"
RAW_POBLACION = RAW / "dane_proyeccion_poblacion_municipal_2020_2035.xlsx"
RAW_UNGRD = [RAW / "ungrd_emergencias_2019_2024.csv", RAW / "ungrd_emergencias_2025.csv"]

# Archivos intermedios y procesados
EVENTOS_CSV = INTERIM / "eventos_unificados.csv"
MUNICIPIOS_CSV = INTERIM / "municipios.csv"
DATASET_CSV = PROCESSED / "dataset_santander.csv"
DICCIONARIO_CSV = DOCS / "diccionario_datos.csv"
PREDICCIONES_CSV = PROCESSED / "predicciones_supervisado.csv"
CLUSTERS_CSV = PROCESSED / "clusters_municipios.csv"
PRIORIZACION_CSV = PROCESSED / "priorizacion_2026.csv"

# ----------------------------------------------------------------- dominio
BOX = dict(lat_min=5.7, lat_max=8.2, lon_min=-74.6, lon_max=-72.4)
NIDO_LAT, NIDO_LON = 6.8, -73.1          # Nido sísmico de Bucaramanga (Mesa de Los Santos)
MAG_MIN = 2.5                             # magnitud de completitud conservadora desde 2002
ANIO_PANEL_INICIO = 1999                  # para calcular rezagos
ANIO_INICIO, ANIO_FIN = 2002, 2025        # filas del dataset
TRAIN, VAL, TEST = (2002, 2015), (2016, 2019), (2020, 2024)
WALKFORWARD_INICIO = 2006
UMBRAL_MEDIO, UMBRAL_ALTO = 1, 5          # n_m25 en t+1: 0 → bajo, 1–4 → medio, ≥5 → alto
CLASES = ["bajo", "medio", "alto"]
RMS_MAX, ERR_PROF_MAX = 2.0, 50.0         # criterio de atípico
PROF_SUPERFICIAL = 30.0
PROF_INTERMEDIA = (70.0, 300.0)
COD_DPTO_SANTANDER = "68"

# ----------------------------------------------------------------- aprendizaje por refuerzo
K_PRESUPUESTO = 10
SEMILLA = 42
SEMILLAS_RL = list(range(10))
RL_LR, RL_ENTROPIA, RL_EPOCAS = 0.05, 0.01, 300
TRAIN_RL, VAL_RL, TEST_RL = (2006, 2015), (2016, 2019), (2020, 2024)

# ----------------------------------------------------------------- listas de variables
FEATURES_SUP = [
    "n_m25", "n_m30", "n_m35", "mag_max", "mag_media", "mag_std", "b_value",
    "log_energia_j", "prof_media_km", "prof_min_km", "pct_superficial", "pct_intermedio",
    "dias_con_sismo", "rms_medio", "gap_medio", "err_prof_medio", "fases_medias",
    "sin_eventos", "n_m25_prom_3a", "tendencia", "mag_max_historica", "dist_nido_km",
    "lat", "lon", "periodo_red",
]
FEATURES_EXPOSICION = ["poblacion_total", "pct_rural"]
FEATURES_VULNERABILIDAD = [
    "emergencias_prom_anual", "mov_masa_total", "viviendas_afectadas_total",
    "personas_afectadas_total",
]
OBJETIVO = "nivel_actividad_t1"
AUXILIARES_FUTURO = ["n_m25_t1", "log_energia_t1"]   # NUNCA se usan como entrada

FEATURES_PERFIL = [
    "tasa_m25_anual", "tasa_m30_anual", "tasa_m35_anual", "mag_max", "mag_media", "b_value",
    "prof_media_km", "pct_superficial", "pct_intermedio", "log_energia_anual",
    "fraccion_anios_activos",
]

# Orden fijo del estado del agente. Los one-hot de cluster_rl se insertan en
# evaluacion_rl.construir_estados como "cluster_rl_<i>" después de dist_nido_km.
FEATURES_RL_AMENAZA = ["p_alto", "p_medio", "n_m25_prom_3a_log", "mag_max_historica", "dist_nido_km"]
FEATURES_RL_EXPOSICION = ["log_poblacion", "pct_rural"]
FEATURES_RL_VULNERABILIDAD = ["emergencias_log", "mov_masa_log", "viviendas_log"]
FEATURES_RL = FEATURES_RL_AMENAZA + FEATURES_RL_EXPOSICION + FEATURES_RL_VULNERABILIDAD
