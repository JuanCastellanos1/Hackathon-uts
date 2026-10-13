# Plan detallado — Reto Avanzado · Hackathon CIAI UTSmart 2026

**Título del proyecto (propuesto):** *Priorización exploratoria del riesgo sísmico en Santander con un agente de refuerzo*
**Cierre de entregas:** 12:00 m. · **Presentación:** 10 min + 3 de preguntas

---

## 0. Diagnóstico de los datos ya descargados (¿sirven?)

**Sí sirven.** Se verificó con análisis directo:

| Verificación | Resultado | Implicación |
|---|---|---|
| Catálogo SGC 1993–2018 (Earthworm/SEISAN) | 111.908 eventos (16 duplicados exactos; al unificar ambos catálogos se eliminan 38 repetidos del mismo sismo) | OK: deduplicados |
| Catálogo SGC 2018–hoy (SeisComP3) | 110.177 eventos, sin duplicados, 99,9 % revisados manualmente | OK |
| Nulos en fecha, coordenadas, profundidad y magnitud | 0 | OK |
| Columnas distintas entre consultas | Ml/Mw + depto/municipio **vs.** mag + tipo_mag + región en texto | Unificación al esquema común `t, lat, lon, prof, mag, rms, gap, fases, errores` |
| Magnitud de completitud (Mc, máxima curvatura) | ≈3,0 (1993–2000) · ≈2,5 (2001–2008) · ≈1,9 (2009–2026) | **La red se densificó.** Los conteos crudos crecen por detección, no por actividad. **Solo se usan eventos M ≥ 2,5 y años desde 2002** |
| Sesgo de magnitud entre catálogos (M ≥ 2,5) | media 3,01 (antiguo) vs 2,88 (nuevo) | Diferencia pequeña. Se declara como limitación y se agrega una variable `periodo_red` |
| Asignación de cada sismo a un municipio (cabecera DANE más cercana) | 176 municipios en el recuadro (87 de Santander) · coincide en **95,3 %** con el municipio del propio SGC | Método uniforme y válido para ambos catálogos |
| Cruce DIVIPOLA ↔ UNGRD | 96 % de los municipios | OK |
| Cruce DIVIPOLA ↔ DANE población | clave numérica (`5001.0`) → se normaliza a `05001` | OK tras normalizar |
| 2026 | incompleto (agosto: 280 eventos) | Queda fuera de la evaluación. Se usa solo para la priorización final |
| Distribución del objetivo (M ≥ 2,5 por municipio-año) | 64 % de las celdas son 0 y Los Santos concentra la mayoría | **No se pueden usar terciles.** Se definen umbrales absolutos (ver §2) |
| Población DANE | solo 2020–2035 | Exposición tratada como **estática (2020)** y declarada como limitación |
| Emergencias UNGRD | 2019–2022 y 2025 | Vulnerabilidad tratada como **estática** (promedio anual). **No entra al modelo supervisado** para evitar fuga temporal; solo entra a la priorización y al agente |

Fuentes (con URL, fecha de descarga UTC y filtros en `data/raw/metadata_*.json`):
1. **SGC – Catálogo RSNC** (principal) → `bdrsnc.sgc.gov.co/paginas1/catalogo/`, recuadro lat 5,7–8,2 y lon −74,6 a −72,4, todas las magnitudes y profundidades.
2. **DANE – DIVIPOLA** (datos.gov.co `gdxc-w37w`) → códigos y coordenadas de las cabeceras.
3. **DANE – Proyecciones de población municipal 2020–2035** (post-COVID, base CNPV 2018).
4. **UNGRD – Consolidado de emergencias** (datos.gov.co `wwkg-r6te`, `2343-nuqp`).

> Marco conceptual que se le cuenta al jurado: **Riesgo = Amenaza × Exposición × Vulnerabilidad.**
> - Amenaza: SGC, estimada con los modelos supervisado y no supervisado.
> - Exposición: población DANE.
> - Vulnerabilidad: emergencias UNGRD.
> - El agente de refuerzo aprende a combinarlas para priorizar.

---

## 1. Tabla única (`data/processed/dataset_santander.csv`)

- **Unidad de observación:** municipio × año. Hay 176 municipios del recuadro y años de 2002 a 2025, lo que da unas **4.200 filas**.
- **Asignación espacial:** epicentro → cabecera municipal más cercana (aproximación de Voronoi).
- **Limpieza:**
  - Eliminar duplicados.
  - Unificar unidades (km, grados, s) y fechas en UTC.
  - Excluir los eventos de calidad muy baja, marcados como atípicos (RMS > 2 s o error de profundidad > 50 km) y reportados en el log.
  - Un municipio-año sin sismos queda con conteos en 0 y las estadísticas de magnitud imputadas (documentado).

### Variables (≥ 20 justificadas; el detalle va en `docs/diccionario_datos.csv`)

| # | Variable | Fuente | Por qué |
|---|---|---|---|
| 1 | `n_eventos_m25` | SGC | Tasa de sismicidad sobre la completitud |
| 2 | `n_eventos_m30` | SGC | Sismos moderados |
| 3 | `n_eventos_m35` | SGC | Sismos potencialmente sentidos |
| 4 | `mag_max` | SGC | Evento máximo del año |
| 5 | `mag_media` | SGC | Nivel típico |
| 6 | `mag_std` | SGC | Dispersión |
| 7 | `b_value` (Aki-MLE, con mínimo de eventos; si no, el valor regional) | SGC | Ley de Gutenberg-Richter: valores b bajos indican mayor proporción de sismos grandes |
| 8 | `log_energia_J` (Σ 10^(1,5M+4,8)) | SGC | Energía liberada |
| 9 | `prof_media_km` | SGC | Los sismos superficiales son más dañinos |
| 10 | `prof_min_km` | SGC | El más superficial |
| 11 | `pct_superficial` (< 30 km) | SGC | Fuente cortical |
| 12 | `pct_intermedio` (70–300 km) | SGC | Firma del Nido de Bucaramanga |
| 13 | `dias_con_sismo` | SGC | Persistencia temporal |
| 14 | `rms_medio` | SGC | Calidad de la localización (incertidumbre) |
| 15 | `gap_medio` | SGC | Cobertura de la red |
| 16 | `err_prof_medio` | SGC | Incertidumbre de la profundidad |
| 17 | `fases_medias` | SGC | Densidad de estaciones |
| 18 | `n_m25_prom_3a` (rezago) | SGC derivada | Memoria de la actividad reciente |
| 19 | `tendencia` (n_t / prom_3a) | SGC derivada | Aceleración o desaceleración |
| 20 | `mag_max_historica` (hasta t) | SGC derivada | Máximo histórico conocido |
| 21 | `dist_nido_km` | SGC + DANE | Cercanía al Nido de Bucaramanga (6,8° N, −73,1° O) |
| 22 | `lat`, `lon` de la cabecera | DANE | Ubicación |
| 23 | `periodo_red` (0 = Earthworm, 1 = SeisComP3) | SGC | Control del cambio instrumental |
| 24 | `poblacion_total` | DANE | Exposición |
| 25 | `pct_rural` | DANE | Exposición o vulnerabilidad (vivienda rural) |
| 26 | `emergencias_prom_anual` | UNGRD | Vulnerabilidad histórica |
| 27 | `mov_masa_total` | UNGRD | Laderas inestables (efecto secundario de los sismos) |
| 28 | `viviendas_afectadas_total` | UNGRD | Vulnerabilidad física |
| 29 | `personas_afectadas_total` | UNGRD | Vulnerabilidad social |
| 30 | `es_santander` | DANE | Filtro de la priorización y de la comparación territorial |
| **Y** | `nivel_actividad_t+1` | SGC (año siguiente) | **Variable objetivo** |

---

## 2. Componente supervisado (reto Básico) — `src/04_supervisado.py`

- **Objetivo:** `nivel_actividad_t+1` ∈ {bajo, medio, alto}, según el número de sismos M ≥ 2,5 en el municipio **el año siguiente**.
  - Umbrales propuestos: bajo = 0 · medio = 1–4 · alto = ≥ 5. Se ajustan para tener clases razonables y se fijan antes de entrenar.
  - **No es circular:** se calcula con datos de t+1 y todas las entradas son de t o anteriores.
- **Validación temporal:**
  - Entrenamiento: t de 2002 a 2015.
  - Validación: t de 2016 a 2019, para elegir hiperparámetros.
  - **Prueba:** t de 2020 a 2024. Nunca se mezcla futuro con pasado.
- **Modelos:** Random Forest y Gradient Boosting con `class_weight`, comparados contra dos líneas base: la clase más frecuente y la persistencia ("igual que este año").
- **Métricas:** F1-macro, matriz de confusión, recall de la clase "alto" y balanced accuracy.
- **Explicación e incertidumbre:**
  - Importancia por permutación.
  - Probabilidades por clase (la confianza de cada predicción).
  - Desviación de la F1 por bootstrap sobre el conjunto de prueba.

## 3. Componente no supervisado (reto Medio, separado) — `src/05_no_supervisado.py`

- **Datos:** perfil sísmico agregado de cada uno de los 176 municipios (2002–2025): tasas, magnitudes, profundidades, b-value, energía y distancia al Nido. **Sin etiquetas.**
- **Modelos:**
  - K-Means con k elegido por silueta, Davies-Bouldin y codo, comparado con un GMM.
  - Para entender las fuentes: DBSCAN sobre los epicentros (lat, lon, profundidad), con el que debería aparecer el Nido sin habérselo indicado.
- **Comparación territorial:**
  - Santander frente a Boyacá, Norte de Santander y los demás municipios del recuadro: en qué grupo cae cada uno y cómo se reparten.
  - Mapa y PCA en 2D.
- **Interpretación:**
  - Centroides de cada grupo en unidades reales.
  - Importancia de variables con un árbol sustituto entrenado sobre los clusters y con la varianza entre grupos.

## 4. Agente de aprendizaje por refuerzo (componente Avanzado) — `src/06_agente_rl.py`

**Problema de decisión:** con un presupuesto limitado de *K = 10 estudios o inspecciones por año*, ¿a qué municipios de Santander (87) se les da prioridad?

| Elemento | Definición |
|---|---|
| **Entorno** | `PriorizacionSantanderEnv` (API de Gymnasium). Cada episodio es un año t. Los candidatos son los 87 municipios de Santander |
| **Estado** | Por cada candidato: probabilidades P(alto) y P(medio) del modelo supervisado, cluster no supervisado (one-hot), log de la población, % rural, emergencias, movimientos en masa, viviendas afectadas, dist_nido y la bandera "ya elegido". Además, el presupuesto restante |
| **Acción** | Elegir 1 municipio aún no elegido. El episodio termina tras K elecciones |
| **Recompensa** | Por cada municipio elegido: **riesgo realizado en t+1 = amenaza observada (energía de los sismos M ≥ 2,5 en t+1, normalizada) × (0,5 exposición + 0,5 vulnerabilidad)**. El agente no ve t+1; solo recibe la recompensa |
| **Algoritmo** | Policy gradient (REINFORCE con línea base): política softmax lineal sobre los candidatos, en numpy. Los pesos aprendidos son interpretables: dicen qué criterio valora el agente |
| **Entrenamiento** | Episodios con los años 2002–2015, muchos pases. 10 semillas |
| **Evaluación** (años de prueba 2020–2024) | Recompensa acumulada y % del óptimo (oráculo con información perfecta), comparada con: azar, mayor población, mayor actividad pasada, ranking por P(alto) del modelo supervisado e índice fijo de pesos iguales. Curva de aprendizaje |
| **Incertidumbre** | Media ± desviación entre semillas. **Estabilidad del ranking:** frecuencia con que cada municipio entra en el top-K (bootstrap y semillas) |
| **Salida** | **Priorización exploratoria 2026**: se aplica la política al estado de 2025. Da un ranking con puntaje, nivel de confianza y explicación (contribución de cada criterio) |

## 5. Demo — `app/app.py` (Streamlit)

Pestañas:
1. **Problema y datos**: fuentes, recuadro y diccionario.
2. **Supervisado**: métricas y matriz de confusión.
3. **No supervisado**: mapa de clusters y comparación territorial.
4. **Agente RL**: curva de aprendizaje y comparación con las líneas base.
5. **Priorización**: mapa de Santander, slider de K, tabla top-K con confianza y explicación.

Todo es reproducible con `make all` o `python run_all.py`, que re-ejecuta del 01 al 06 con semillas fijas.

## 6. Entregables → rúbrica

| Entregable / criterio | Archivo | Puntos |
|---|---|---|
| Dataset limpio + diccionario | `data/processed/dataset_santander.csv`, `docs/diccionario_datos.csv` | Datos 15 |
| Fuente citada | `README.md` §Fuentes + `metadata_*.json` | Datos 15 |
| Código reproducible | `src/01…06`, `run_all.py`, `requirements.txt` | Flujo 15 |
| Supervisado + no supervisado + RL con métricas | `reports/metricas_*.json`, figuras en `reports/figures/` | RL 20 · Validación 10 |
| Demo de la priorización | `app/app.py` | Impacto 20 |
| Supuestos, limitaciones y extensiones | `README.md` §Limitaciones | Impacto 20 |
| Pitch | `docs/presentacion` (slides) | Presentación 20 |

## 7. Cronograma (hoy, cierre 12:00)

| Hora | Tarea |
|---|---|
| 09:15 – 09:45 | `03_tabla.py`: limpieza, unificación, asignación a municipios, variables y diccionario |
| 09:45 – 10:10 | `04_supervisado.py` + métricas |
| 10:10 – 10:30 | `05_no_supervisado.py` |
| 10:30 – 11:10 | `06_agente_rl.py`: entorno, entrenamiento, evaluación y priorización 2026 |
| 11:10 – 11:35 | App de Streamlit + README (fuentes, limitaciones) |
| 11:35 – 11:50 | Slides del pitch + commit/push al repo del equipo |
| 11:50 – 12:00 | Colchón y verificación final (`run_all.py` desde cero) |

**Reparto sugerido para el equipo:** una persona revisa los resultados y prepara el pitch (problema, impacto y Santander); otra prueba la app y ensaya la demo; mientras tanto se construye el pipeline.

## 8. Supuestos y limitaciones (se declaran)

- Asignación por la cabecera más cercana (Voronoi) en vez de polígonos oficiales del IGAC: coincide en un 95 % con la etiqueta del SGC.
- Mc variable: solo se usa M ≥ 2,5 desde 2002. El cambio de software en 2018 se trata con la variable `periodo_red`.
- La exposición (2020) y la vulnerabilidad (2019–2025) son estáticas.
- El **"riesgo realizado" es un proxy**: no hay datos de daño por sismo con resolución suficiente.
- La priorización es **exploratoria**: orienta estudios de microzonificación o inspección, **no** predice sismos.
- **Extensiones:**
  - Polígonos IGAC.
  - Amenaza NSR-10 (Aa/Av).
  - Sismicidad histórica con daños del SISH.
  - Vulnerabilidad estructural del catastro.
  - Recompensas con costo por inspección.
