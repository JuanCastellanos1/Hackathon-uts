# Contratos de datos e interfaces

**Fuente de verdad** para los subagentes. Si algo de aquí choca con otro documento, **manda este**. Ningún subagente cambia un contrato: si lo cree necesario, lo reporta al orquestador.

Convenciones globales:
- Python 3.12 en `.venv/`. Ejecutar siempre con `.venv/bin/python`.
- Importar el paquete desde los scripts así: `sys.path.insert(0, str(Path(__file__).resolve().parent))` y luego `from sismo import config`.
- `cod_mpio`: **siempre `str` de 5 dígitos** (`"68001"`). Al leer un CSV, usar `dtype={"cod_mpio": str}`.
- CSV en UTF-8, separador `,`, decimal `.`, sin índice (`index=False`).
- Semilla global `config.SEMILLA = 42`. Todo lo aleatorio recibe una semilla explícita.
- Nombres de columnas en `snake_case`, sin tildes.
- Figuras PNG a 120 dpi en `reports/figures/`, con títulos en español.
- Logs con `print()` (sin librería de logging) y prefijo `[nombre_modulo]`.

---

## 1. `src/sismo/config.py` (lo escribe el orquestador en la Fase 0; para los subagentes es solo lectura)

```python
ROOT = Path(__file__).resolve().parents[2]
RAW, INTERIM, PROCESSED = ROOT/"data/raw", ROOT/"data/interim", ROOT/"data/processed"
MODELS, REPORTS, FIGURES, DOCS = ROOT/"models", ROOT/"reports", ROOT/"reports/figures", ROOT/"docs"
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
PROF_SUPERFICIAL, PROF_INTERMEDIA = 30.0, (70.0, 300.0)
COD_DPTO_SANTANDER = "68"
K_PRESUPUESTO = 10
SEMILLA = 42
SEMILLAS_RL = list(range(10))
RL_LR, RL_ENTROPIA, RL_EPOCAS = 0.05, 0.01, 300
TRAIN_RL, VAL_RL, TEST_RL = (2006, 2015), (2016, 2019), (2020, 2024)
FEATURES_SUP = [...]      # §3, lista "sup"
FEATURES_PERFIL = [...]   # §5
FEATURES_RL = [...]       # §6
```

---

## 2. `data/interim/eventos_unificados.csv` (un sismo por fila)

| columna | tipo | unidad | regla |
|---|---|---|---|
| event_id | str | – | `f"{catalogo}_{i:07d}"` |
| fecha_hora_utc | str ISO | UTC | `YYYY-MM-DD HH:MM:SS` |
| anio | int | año | |
| lat, lon | float | grados | |
| prof_km | float | km | |
| mag | float | – | antiguo: `MAGNITUD Ml` · nuevo: `mag` |
| tipo_mag | str | – | antiguo: `"Ml"` · nuevo: `tipo_mag` |
| mw | float | – | antiguo: `MAGNITUD Mw` (puede ser NaN) · nuevo: NaN |
| fases | int | – | antiguo `# FASES` · nuevo `fases` |
| rms_s | float | s | |
| gap_grados | float | grados | |
| err_lat_km, err_lon_km, err_prof_km | float | km | |
| municipio_sgc | str | – | antiguo: `f"{MUNICIPIO} - {DEPARTAMENTO}"` · nuevo: `region` |
| catalogo | str | – | `"earthworm_seisan"` (1993–2018) o `"seiscomp3"` (2018–hoy) |
| periodo_red | int | 0/1 | 0 = antiguo, 1 = nuevo |
| es_atipico | bool | – | `rms_s > RMS_MAX or err_prof_km > ERR_PROF_MAX` |
| cod_mpio | str(5) | – | cabecera DIVIPOLA más cercana (haversine) dentro de BOX |
| dist_cabecera_km | float | km | |

Columnas de entrada en `data/raw/sgc_1993_2018.csv`: `FECHA, HORA_UTC, LATITUD (grados), LONGITUD (grados), PROFUNDIDAD (Km), MAGNITUD Ml, MAGNITUD Mw, DEPARTAMENTO, MUNICIPIO, # FASES, RMS (Seg), GAP (grados), ERROR LATITUD (Km), ERROR LONGITUD (Km), ERROR PROFUNDIDAD (Km), ESTADO`. Trae 16 duplicados exactos, que se eliminan.

Columnas de entrada en `data/raw/sgc_2018_hoy.csv`: `fecha_hora_utc, lat, lon, prof_km, mag, tipo_mag, fases, rms_s, gap_grados, err_lat_km, err_lon_km, err_prof_km, region, quakeml, fases_link, mapa, estado`.

Numéricos: `pd.to_numeric(errors="coerce")`. Las filas con NaN en lat, lon, prof o mag se descartan y se cuentan en el log.

---

## 3. `data/interim/municipios.csv` (un municipio por fila, solo cabeceras dentro de BOX: 176)

| columna | tipo | fuente | regla |
|---|---|---|---|
| cod_mpio | str(5) | DIVIPOLA | `str(int).zfill(5)` |
| nom_mpio | str | DIVIPOLA | |
| cod_dpto | str(2) | DIVIPOLA | `str(int).zfill(2)` |
| dpto | str | DIVIPOLA | |
| lat, lon | float | DIVIPOLA | `latitud`/`longitud` vienen con coma decimal → `str.replace(",", ".")` |
| es_santander | int 0/1 | DIVIPOLA | `cod_dpto == "68"` |
| dist_nido_km | float | derivada | haversine a (NIDO_LAT, NIDO_LON) |
| poblacion_total | int | DANE | `ÁREA GEOGRÁFICA == "Total"` y `AÑO == 2020` |
| pct_rural | float 0–1 | DANE | `1 − Cabecera Municipal / Total` (2020) |
| n_emergencias_total | int | UNGRD | conteo de registros en los dos archivos |
| emergencias_prom_anual | float | UNGRD | `n_emergencias_total / n_anios_cubiertos` (años distintos en los datos) |
| mov_masa_total | int | UNGRD | eventos que contienen `"MOVIMIENTO EN MASA"` |
| viviendas_afectadas_total | int | UNGRD | Σ destruidas + averiadas |
| personas_afectadas_total | int | UNGRD | Σ `personas` |

Lectura de DANE: `pd.read_excel(path, header=8)`. Columnas: `DP, DPNOM, MPIO (float, p. ej. 5001.0), DPMP (nombre), AÑO, ÁREA GEOGRÁFICA, Población`. Hay que quitar las filas de pie de página (MPIO NaN). `cod_mpio = str(int(MPIO)).zfill(5)`.

Lectura de UNGRD:
- `ungrd_emergencias_2019_2024.csv`: columnas `fecha, departamento, municipio, evento, divipola, personas, viviendas_destruidas, viviendas_averiadas, …`.
- `ungrd_emergencias_2025.csv`: columnas `fecha, …, evento, codificaci_n_segun_divipola, personas, viv_destru, viv_aver, …`.
- `divipola` es numérico sin cero inicial → `zfill(5)`. Los numéricos pueden venir como texto → `to_numeric(errors="coerce").fillna(0)`.
- Un municipio sin registros queda con 0. Un municipio sin población queda con NaN, y se reporta (no debería ocurrir).

---

## 4. `data/processed/dataset_santander.csv` — **TABLA ÚNICA** (municipio × año)

Filas: 176 municipios × años 2002–2025 = **4.224 filas**. Llave única: (`cod_mpio`, `anio`).
Las variables sísmicas usan **solo eventos con `mag ≥ MAG_MIN` y `es_atipico == False`**.

| # | columna | tipo | unidad | rol | cálculo (año t, municipio m) |
|---|---|---|---|---|---|
| – | cod_mpio, nom_mpio, dpto | str | – | llave | |
| – | anio | int | año | llave | |
| 1 | n_m25 | int | sismos | sup | conteo M ≥ 2,5 |
| 2 | n_m30 | int | sismos | sup | conteo M ≥ 3,0 |
| 3 | n_m35 | int | sismos | sup | conteo M ≥ 3,5 |
| 4 | mag_max | float | M | sup | máximo (0 si no hay eventos) |
| 5 | mag_media | float | M | sup | media (0 si no hay eventos) |
| 6 | mag_std | float | M | sup | desviación (0 si hay menos de 2 eventos) |
| 7 | b_value | float | – | sup | Aki-MLE en la ventana t−4..t: `log10(e) / (media(M) − (MAG_MIN − 0,05))` si n ≥ 30; si no, el b regional de la misma ventana (todos los municipios) |
| 8 | log_energia_j | float | log10 J | sup | `log10(Σ 10^(1,5·M + 4,8))`; 0 si no hay eventos |
| 9 | prof_media_km | float | km | sup | media; −1 si no hay eventos |
| 10 | prof_min_km | float | km | sup | mínimo; −1 si no hay eventos |
| 11 | pct_superficial | float | 0–1 | sup | fracción con prof < 30 km (0 si no hay eventos) |
| 12 | pct_intermedio | float | 0–1 | sup | fracción con 70 ≤ prof ≤ 300 km |
| 13 | dias_con_sismo | int | días | sup | número de días distintos con un evento |
| 14 | rms_medio | float | s | sup | media; −1 si no hay eventos |
| 15 | gap_medio | float | grados | sup | media; −1 si no hay eventos |
| 16 | err_prof_medio | float | km | sup | media; −1 si no hay eventos |
| 17 | fases_medias | float | – | sup | media; −1 si no hay eventos |
| 18 | sin_eventos | int 0/1 | – | sup | 1 si n_m25 == 0 |
| 19 | n_m25_prom_3a | float | sismos | sup | media de n_m25 en t−1, t−2, t−3 |
| 20 | tendencia | float | – | sup | `n_m25 / (n_m25_prom_3a + 1)` |
| 21 | mag_max_historica | float | M | sup | máximo de mag (M ≥ 2,5) desde 1993 hasta t |
| 22 | dist_nido_km | float | km | sup | de municipios |
| 23 | lat | float | grados | sup | cabecera |
| 24 | lon | float | grados | sup | cabecera |
| 25 | periodo_red | int 0/1 | – | sup | 1 si anio ≥ 2018 |
| 26 | es_santander | int 0/1 | – | territorial | |
| 27 | poblacion_total | int | hab. | exposición | DANE 2020 |
| 28 | pct_rural | float | 0–1 | exposición | DANE 2020 |
| 29 | emergencias_prom_anual | float | eventos/año | vulnerabilidad | UNGRD |
| 30 | mov_masa_total | int | eventos | vulnerabilidad | UNGRD |
| 31 | viviendas_afectadas_total | int | viviendas | vulnerabilidad | UNGRD |
| 32 | personas_afectadas_total | int | personas | vulnerabilidad | UNGRD |
| Y | **nivel_actividad_t1** | str | – | **objetivo** | `n_m25` de t+1: 0 → `"bajo"`, 1–4 → `"medio"`, ≥5 → `"alto"`. NaN en 2025 |
| A1 | n_m25_t1 | float | sismos | auxiliar futuro | n_m25 de t+1 (NaN en 2025). **Nunca es entrada** |
| A2 | log_energia_t1 | float | log10 J | auxiliar futuro | log_energia_j de t+1 (NaN en 2025). **Nunca es entrada**; solo sirve para la recompensa RL |

`FEATURES_SUP` = filas 1–25 (rol `sup`). Exposición y vulnerabilidad **no** entran al supervisado: modela solo la amenaza, y además así se evita la fuga temporal de UNGRD.

Validaciones obligatorias de `construir_dataset()` (con `assert`):
- No hay filas duplicadas por llave.
- No hay NaN en `FEATURES_SUP`.
- `nivel_actividad_t1` es NaN si y solo si `anio == 2025`.
- Hay al menos 20 columnas de variables.

### `docs/diccionario_datos.csv`
Columnas: `variable, tipo, unidad, fuente, rol, descripcion, justificacion`, con una fila por cada columna de la tabla. `fuente` ∈ {`SGC catálogo RSNC`, `DANE DIVIPOLA`, `DANE proyecciones población 2020`, `UNGRD emergencias (datos.gov.co)`, `Derivada`}.

---

## 5. Salidas del componente no supervisado

`FEATURES_PERFIL` (perfil por municipio con los años 2002–2025; en la versión RL, solo 2002–2015):
`tasa_m25_anual, tasa_m30_anual, tasa_m35_anual, mag_max, mag_media, b_value, prof_media_km, pct_superficial, pct_intermedio, log_energia_anual, fraccion_anios_activos`.
Transformación: `log1p` a las tasas, luego `StandardScaler`.

`data/processed/clusters_municipios.csv`: `cod_mpio, nom_mpio, dpto, es_santander, lat, lon, cluster, cluster_nombre, pca1, pca2, cluster_rl, <FEATURES_PERFIL>`.
- `cluster`: K-Means sobre el periodo completo, para el análisis descriptivo.
- `cluster_rl`: el mismo K-Means **ajustado solo con 2002–2015**, que es el que usa el agente (sin fuga).

`reports/metricas_no_supervisado.json`:
```json
{"k_elegido": int, "seleccion_k": [{"k":2,"silueta":..,"davies_bouldin":..,"inercia":..,"gmm_bic":..}, ...],
 "silueta_final": float, "centroides": {"<nombre>": {"<feature>": valor_real}},
 "comparacion_territorial": {"<dpto>": {"<cluster_nombre>": n}},
 "importancia_variables": {"<feature>": float},
 "dbscan_fuentes": [{"fuente": int, "n": int, "lat": .., "lon": .., "prof_media_km": .., "mag_max": ..}]}
```
Figuras: `nosup_seleccion_k.png`, `nosup_mapa_clusters.png`, `nosup_pca.png`, `nosup_comparacion_territorial.png`.

---

## 6. Salidas del supervisado y del agente RL

`data/processed/predicciones_supervisado.csv` (walk-forward): `cod_mpio, anio, p_bajo, p_medio, p_alto, pred, y_true` para anio 2006–2025 (y_true NaN en 2025).

`reports/metricas_supervisado.json`:
```json
{"modelo_elegido": str, "params": {..}, "val": {"f1_macro":..}, 
 "test": {"f1_macro":.., "balanced_accuracy":.., "recall_alto":.., "matriz_confusion": [[..]], "reporte": {..}},
 "lineas_base_test": {"mayoritaria": {"f1_macro":..}, "persistencia": {"f1_macro":..}},
 "bootstrap_f1_test": {"media":.., "p2_5":.., "p97_5":..},
 "importancias": {"<feature>": float}, "distribucion_clases": {"train": {..}, "test": {..}}}
```
Figuras: `sup_matriz_confusion.png`, `sup_importancias.png`.

### Estado y recompensa del agente (solo los 87 municipios de Santander, ordenados por cod_mpio)

`FEATURES_RL` (orden fijo):
`p_alto, p_medio, n_m25_prom_3a_log, mag_max_historica, dist_nido_km, cluster_rl_<i> (one-hot), log_poblacion, pct_rural, emergencias_log, mov_masa_log, viviendas_log`
- `*_log = log1p(x)`.
- Estandarizar con la media y la desviación de los años `TRAIN_RL` (una desviación de 0 se reemplaza por 1).

**Recompensa** del municipio i en el año t:
```
amenaza_i   = log_energia_t1_i / max(log_energia_t1 en Santander, años TRAIN_RL)   (recortada a [0,1])
exposicion_i = minmax(log1p(poblacion_total)) en Santander
vulnerab_i  = media(minmax(log1p(emergencias_prom_anual)), minmax(log1p(mov_masa_total)), minmax(log1p(viviendas_afectadas_total)))
r_i = amenaza_i × (0.5·exposicion_i + 0.5·vulnerab_i)
```

`reports/metricas_rl.json`:
```json
{"config": {"k":10, "lr":.., "epocas":.., "semillas":10, "anios_train":[..], "anios_test":[..]},
 "test": {"agente_rl": {"recompensa_media":.., "std_semillas":.., "pct_oraculo":.., "precision_at_k":..},
          "azar": {..}, "mayor_poblacion": {..}, "mayor_actividad_pasada": {..},
          "ranking_supervisado": {..}, "indice_pesos_iguales": {..}, "oraculo": {"recompensa_media":..}},
 "pesos_politica": {"<feature>": {"media":.., "std":..}},
 "curva_aprendizaje": "reports/curva_aprendizaje_rl.csv"}
```
`precision_at_k` = |top-K elegido ∩ top-K del oráculo| / K, promediado por año.

`reports/curva_aprendizaje_rl.csv`: `semilla, epoca, recompensa_train, pct_oraculo_val`.

`data/processed/priorizacion_2026.csv` (87 filas, ordenadas por rango): `rango, cod_mpio, nom_mpio, lat, lon, puntaje, prob_top_k, confianza, contrib_amenaza, contrib_exposicion, contrib_vulnerabilidad, p_alto, cluster_nombre, poblacion_total, n_m25_2025`.
- `puntaje`: media entre semillas del puntaje `w·x` estandarizado.
- `prob_top_k`: fracción de semillas en que el municipio queda en el top-K greedy.
- `confianza`: `"alta"` si ≥ 0,8 · `"media"` si ≥ 0,5 · `"baja"` en otro caso.
- Contribuciones: suma de `w·x` por grupo. **Amenaza** = p_alto, p_medio, n_m25_prom_3a_log, mag_max_historica, dist_nido_km y cluster_rl_*. **Exposición** = log_poblacion y pct_rural. **Vulnerabilidad** = emergencias_log, mov_masa_log y viviendas_log.

Figuras: `rl_curva_aprendizaje.png`, `rl_comparacion_lineas_base.png`, `rl_pesos_politica.png`, `rl_mapa_priorizacion.png`.

`models/agente_rl.npz`: `pesos` (n_semillas × n_feat), `features` (nombres), `media`, `std` (estandarización).
