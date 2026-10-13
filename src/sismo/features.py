"""Construcción de la tabla única municipio × año (CONTRATOS §4).

Diagrama de clases: docs/uml/03_clases.md (módulo `features`). Actividad: docs/uml/05_actividad_y_estados.md §5.1.

Flujo de `construir_dataset()`:
  eventos -> municipios -> asignar_municipio -> panel 1999–2025 (agregar_anual) -> rezagos ->
  objetivo (t+1) -> territorio -> recorte 2002–2025 -> validaciones -> guardado.

Regla anti-fuga: las variables de entrada usan solo años <= t (rezagos t−1..t−3, ventana t−4..t).
El objetivo y los auxiliares usan t+1 y nunca entran como features.
"""
import numpy as np
import pandas as pd

from sismo import catalogo, config, territorio

PREFIJO = "[features]"

ANIO_HISTORIA_INICIO = 1993     # el máximo histórico y la ventana del b-value empiezan antes del panel
ANIO_RED_NUEVA = 2018           # inicio del catálogo SeisComP3 (periodo_red = 1)
MAG_M30, MAG_M35 = 3.0, 3.5     # umbrales de conteo adicionales (CONTRATOS §4, vars 2 y 3)
VENTANA_B = 5                   # años de la ventana del b-value (t−4..t)
MIN_EVENTOS_B = 30              # por debajo de este n, el b-value es el regional

LLAVES = ["cod_mpio", "nom_mpio", "dpto", "anio"]
COLUMNAS_TERRITORIO = [
    "cod_mpio", "nom_mpio", "dpto", "es_santander", "dist_nido_km", "lat", "lon",
    "poblacion_total", "pct_rural", "emergencias_prom_anual", "mov_masa_total",
    "viviendas_afectadas_total", "personas_afectadas_total",
]
COLUMNAS_PANEL = [
    "cod_mpio", "anio", "n_m25", "n_m30", "n_m35", "mag_max", "mag_media", "mag_std",
    "b_value", "log_energia_j", "prof_media_km", "prof_min_km", "pct_superficial",
    "pct_intermedio", "dias_con_sismo", "rms_medio", "gap_medio", "err_prof_medio",
    "fases_medias", "sin_eventos", "mag_max_historica",
]


def b_value_aki(mags, mc):
    """b-value de Gutenberg-Richter por máxima verosimilitud (Aki, 1965).

    b = log10(e) / (media(M) − (mc − 0,05)), usando solo magnitudes M >= mc.
    Devuelve NaN si hay menos de 2 eventos válidos.
    """
    m = np.asarray(mags, dtype=float)
    m = m[np.isfinite(m) & (m >= mc)]
    if m.size < 2:
        return np.nan
    # Con M >= mc la media siempre supera (mc − 0,05), así que el denominador es > 0.
    denominador = m.mean() - (mc - 0.05)
    return float(np.log10(np.e) / denominador)


def nivel_actividad(n):
    """Clasifica un conteo mensual/anual en bajo / medio / alto con los umbrales de config.

    0 → bajo, 1–4 → medio, ≥5 → alto (con UMBRAL_MEDIO=1 y UMBRAL_ALTO=5). NaN se mantiene NaN.
    """
    serie = n if isinstance(n, pd.Series) else pd.Series(np.asarray(n, dtype=float))
    clase = np.where(
        serie < config.UMBRAL_MEDIO,
        "bajo",
        np.where(serie < config.UMBRAL_ALTO, "medio", "alto"),
    )
    return pd.Series(clase, index=serie.index, dtype=object).where(serie.notna())


def _grilla(municipios, anios):
    """Producto cartesiano municipio × año, ordenado por cod_mpio y luego por año."""
    cods = sorted(municipios["cod_mpio"].unique())
    return pd.MultiIndex.from_product([cods, list(anios)], names=["cod_mpio", "anio"]).to_frame(index=False)


def agregar_anual(eventos, municipios):
    """Panel municipio × año 1999–2025 con las variables 1–18 y 21 de CONTRATOS §4.

    Usa solo eventos con mag >= MAG_MIN y es_atipico == False. Los años sin eventos quedan con
    los rellenos del contrato (0 o −1). El b-value usa la ventana t−4..t, por eso se leen eventos
    desde 1993; el máximo histórico también acumula desde 1993.
    """
    anios_panel = list(range(config.ANIO_PANEL_INICIO, config.ANIO_FIN + 1))

    ev = eventos.loc[
        eventos["cod_mpio"].notna()
        & (~eventos["es_atipico"].astype(bool))
        & (eventos["mag"] >= config.MAG_MIN)
        & (eventos["anio"] >= ANIO_HISTORIA_INICIO)
        & (eventos["anio"] <= config.ANIO_FIN)
    ].copy()
    print(f"{PREFIJO} eventos usados (M >= {config.MAG_MIN}, no atípicos, {ANIO_HISTORIA_INICIO}–"
          f"{config.ANIO_FIN}): {len(ev)}")

    # Columnas auxiliares para agregar en una sola pasada.
    ev["fecha"] = ev["fecha_hora_utc"].str[:10]
    ev["m30"] = (ev["mag"] >= MAG_M30).astype(int)
    ev["m35"] = (ev["mag"] >= MAG_M35).astype(int)
    ev["energia"] = 10.0 ** (1.5 * ev["mag"] + 4.8)
    ev["superficial"] = (ev["prof_km"] < config.PROF_SUPERFICIAL).astype(int)
    lo, hi = config.PROF_INTERMEDIA
    ev["intermedio"] = ((ev["prof_km"] >= lo) & (ev["prof_km"] <= hi)).astype(int)

    estadisticas = ev.groupby(["cod_mpio", "anio"]).agg(
        n_m25=("mag", "size"),
        n_m30=("m30", "sum"),
        n_m35=("m35", "sum"),
        mag_max=("mag", "max"),
        mag_media=("mag", "mean"),
        mag_std=("mag", "std"),          # ddof=1; NaN con un solo evento -> 0 abajo
        energia_total=("energia", "sum"),
        prof_media_km=("prof_km", "mean"),
        prof_min_km=("prof_km", "min"),
        pct_superficial=("superficial", "mean"),
        pct_intermedio=("intermedio", "mean"),
        dias_con_sismo=("fecha", "nunique"),
        rms_medio=("rms_s", "mean"),
        gap_medio=("gap_grados", "mean"),
        err_prof_medio=("err_prof_km", "mean"),
        fases_medias=("fases", "mean"),
    ).reset_index()

    panel = _grilla(municipios, anios_panel).merge(estadisticas, on=["cod_mpio", "anio"], how="left")

    # Rellenos del contrato: conteos y estadísticas de magnitud a 0; profundidad, calidad a −1.
    for col in ["n_m25", "n_m30", "n_m35", "dias_con_sismo", "energia_total", "mag_max",
                "mag_media", "mag_std", "pct_superficial", "pct_intermedio"]:
        panel[col] = panel[col].fillna(0.0)
    for col in ["prof_media_km", "prof_min_km", "rms_medio", "gap_medio",
                "err_prof_medio", "fases_medias"]:
        panel[col] = panel[col].fillna(-1.0)
    for col in ["n_m25", "n_m30", "n_m35", "dias_con_sismo"]:
        panel[col] = panel[col].astype("int64")

    energia = panel["energia_total"].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        log_e = np.log10(np.where(energia > 0, energia, 1.0))
    panel["log_energia_j"] = np.where(energia > 0, log_e, 0.0)
    panel["sin_eventos"] = (panel["n_m25"] == 0).astype(int)

    # b-value en la ventana t−4..t: municipal si n >= MIN_EVENTOS_B, si no el regional de la misma ventana.
    cods = sorted(municipios["cod_mpio"].unique())
    filas_b = []
    for t in anios_panel:
        ventana = ev.loc[(ev["anio"] >= t - (VENTANA_B - 1)) & (ev["anio"] <= t)]
        b_reg = b_value_aki(ventana["mag"].to_numpy(), config.MAG_MIN)
        grupos = ventana.groupby("cod_mpio")["mag"]
        n_ventana = grupos.size().reindex(cods).fillna(0).astype(int).to_numpy()
        b_mun = grupos.apply(lambda s: b_value_aki(s.to_numpy(), config.MAG_MIN)).reindex(cods).to_numpy(dtype=float)
        b_val = np.where(n_ventana >= MIN_EVENTOS_B, b_mun, b_reg)
        filas_b.append(pd.DataFrame({"cod_mpio": cods, "anio": t, "b_value": b_val}))
    panel = panel.merge(pd.concat(filas_b, ignore_index=True), on=["cod_mpio", "anio"], how="left")
    print(f"{PREFIJO} b_value: NaN tras rellenar = {int(panel['b_value'].isna().sum())}")

    # Máximo histórico acumulado desde 1993 (se calcula sobre la grilla completa 1993–ANIO_FIN).
    anios_hist = list(range(ANIO_HISTORIA_INICIO, config.ANIO_FIN + 1))
    maximo_anual = ev.groupby(["cod_mpio", "anio"], as_index=False)["mag"].max().rename(
        columns={"mag": "mag_max_anual"})
    hist = _grilla(municipios, anios_hist).merge(maximo_anual, on=["cod_mpio", "anio"], how="left")
    hist["mag_max_anual"] = hist["mag_max_anual"].fillna(0.0)
    hist["mag_max_historica"] = hist.groupby("cod_mpio")["mag_max_anual"].cummax()
    panel = panel.merge(hist[["cod_mpio", "anio", "mag_max_historica"]], on=["cod_mpio", "anio"], how="left")

    panel = panel[COLUMNAS_PANEL].sort_values(["cod_mpio", "anio"]).reset_index(drop=True)
    print(f"{PREFIJO} panel anual: {len(panel)} filas ({panel['cod_mpio'].nunique()} municipios × "
          f"{len(anios_panel)} años)")
    return panel


def agregar_rezagos(panel):
    """Agrega n_m25_prom_3a, tendencia y periodo_red (variables 19, 20 y 25 de CONTRATOS §4).

    Los rezagos son por municipio y solo miran t−1..t−3. Los primeros años del panel quedan NaN
    (no hay datos antes de 1999) y se recortan después, así que no llegan al dataset final.
    """
    p = panel.sort_values(["cod_mpio", "anio"]).reset_index(drop=True).copy()
    grupo = p.groupby("cod_mpio")["n_m25"]
    p["n_m25_prom_3a"] = (grupo.shift(1) + grupo.shift(2) + grupo.shift(3)) / 3.0
    p["tendencia"] = p["n_m25"] / (p["n_m25_prom_3a"] + 1.0)
    p["periodo_red"] = (p["anio"] >= ANIO_RED_NUEVA).astype(int)
    return p


def agregar_objetivo(panel):
    """Agrega n_m25_t1, log_energia_t1 y nivel_actividad_t1 (CONTRATOS §4, objetivo y auxiliares).

    Se toma el valor del año t+1 del mismo municipio mediante un merge (no un shift posicional),
    así que si falta el año siguiente queda NaN. Para anio == ANIO_FIN ambos auxiliares son NaN.
    """
    p = panel.copy()
    siguiente = p[["cod_mpio", "anio", "n_m25", "log_energia_j"]].copy()
    siguiente["anio"] = siguiente["anio"] - 1     # la fila de t+1 se alinea con t
    siguiente = siguiente.rename(columns={"n_m25": "n_m25_t1", "log_energia_j": "log_energia_t1"})
    p = p.merge(siguiente, on=["cod_mpio", "anio"], how="left")

    p.loc[p["anio"] == config.ANIO_FIN, ["n_m25_t1", "log_energia_t1"]] = np.nan
    p["nivel_actividad_t1"] = nivel_actividad(p["n_m25_t1"])
    return p


def unir_territorio(panel, municipios):
    """Agrega las columnas territoriales (llaves, es_santander, distancia al nido, exposición, vulnerabilidad)."""
    terr = municipios[COLUMNAS_TERRITORIO]
    if terr["cod_mpio"].duplicated().any():
        raise ValueError(f"{PREFIJO} municipios con cod_mpio duplicado")
    salida = panel.merge(terr, on="cod_mpio", how="left", validate="many_to_one")
    faltan = int(salida["nom_mpio"].isna().sum())
    if faltan:
        raise ValueError(f"{PREFIJO} {faltan} filas del panel sin municipio en la tabla territorial")
    return salida


def diccionario_datos():
    """Diccionario de datos: una fila por columna del dataset final (CONTRATOS §4)."""
    S_CAT = "SGC catálogo RSNC"
    S_DIV = "DANE DIVIPOLA"
    S_POB = "DANE proyecciones población 2020"
    S_UNG = "UNGRD emergencias (datos.gov.co)"
    S_DER = "Derivada"
    filas = [
        # llaves
        ("cod_mpio", "str(5)", "–", S_DIV, "llave",
         "Código DIVIPOLA de 5 dígitos del municipio",
         "Junto con anio forma la llave única; se lee como texto para conservar el cero inicial."),
        ("nom_mpio", "str", "–", S_DIV, "llave",
         "Nombre oficial del municipio",
         "Permite leer y presentar resultados sin consultar el código."),
        ("dpto", "str", "–", S_DIV, "llave",
         "Departamento al que pertenece el municipio",
         "Permite comparar Santander con los departamentos vecinos."),
        ("anio", "int", "año", S_DER, "llave",
         "Año de observación de la fila",
         "Eje temporal del panel; las particiones de entrenamiento y prueba se hacen por año."),
        # sup (amenaza sísmica, solo años <= t)
        ("n_m25", "int", "sismos", S_CAT, "sup",
         "Sismos del año con magnitud M >= 2,5 y no atípicos",
         "Medida principal de actividad anual; es la base del objetivo y del rezago."),
        ("n_m30", "int", "sismos", S_CAT, "sup",
         "Sismos del año con magnitud M >= 3,0",
         "Captura la sismicidad más fuerte, menos afectada por la completitud del catálogo."),
        ("n_m35", "int", "sismos", S_CAT, "sup",
         "Sismos del año con magnitud M >= 3,5",
         "Señal de eventos grandes, los que más importan para el riesgo."),
        ("mag_max", "float", "M", S_CAT, "sup",
         "Magnitud máxima del año (0 si no hubo eventos)",
         "Indica el evento más fuerte del año y su potencial de daño."),
        ("mag_media", "float", "M", S_CAT, "sup",
         "Magnitud media del año (0 si no hubo eventos)",
         "Resume el nivel típico de magnitud de la actividad anual."),
        ("mag_std", "float", "M", S_CAT, "sup",
         "Desviación estándar de la magnitud del año (0 con menos de dos eventos)",
         "Describe la dispersión de magnitudes; sin dos eventos no se puede estimar."),
        ("b_value", "float", "–", S_CAT, "sup",
         "b-value de Aki-MLE en la ventana t−4..t; regional si el municipio tiene menos de 30 eventos",
         "La pendiente de Gutenberg-Richter resume la proporción de sismos grandes frente a pequeños."),
        ("log_energia_j", "float", "log10 J", S_CAT, "sup",
         "Log10 de la energía sísmica liberada en el año (0 si no hubo eventos)",
         "La energía pondera los eventos grandes y resume mejor la liberación de tensión que el conteo."),
        ("prof_media_km", "float", "km", S_CAT, "sup",
         "Profundidad media de los hipocentros del año (−1 si no hubo eventos)",
         "Separa la sismicidad cortical de la intermedia asociada a la subducción."),
        ("prof_min_km", "float", "km", S_CAT, "sup",
         "Profundidad mínima de los hipocentros del año (−1 si no hubo eventos)",
         "Indica si hubo sismos muy superficiales, los de mayor impacto local."),
        ("pct_superficial", "float", "0–1", S_CAT, "sup",
         "Fracción de sismos del año con profundidad menor a 30 km",
         "Mide la proporción de sismicidad cortical, la más relevante para zonas urbanas."),
        ("pct_intermedio", "float", "0–1", S_CAT, "sup",
         "Fracción de sismos del año con profundidad entre 70 y 300 km",
         "Mide la actividad del nido sísmico intermedio de Bucaramanga."),
        ("dias_con_sismo", "int", "días", S_CAT, "sup",
         "Número de días distintos con al menos un sismo en el año",
         "Distingue actividad repartida en el año de enjambres concentrados en pocos días."),
        ("rms_medio", "float", "s", S_CAT, "sup",
         "RMS medio de la localización de los sismos del año (−1 si no hubo eventos)",
         "Indicador de calidad: residuales altos sugieren localizaciones poco confiables."),
        ("gap_medio", "float", "grados", S_CAT, "sup",
         "GAP azimutal medio de la localización (−1 si no hubo eventos)",
         "Cobertura de estaciones; valores altos indican una geometría de localización pobre."),
        ("err_prof_medio", "float", "km", S_CAT, "sup",
         "Error medio de la profundidad (−1 si no hubo eventos)",
         "Incertidumbre de la profundidad, necesaria para interpretar pct_superficial."),
        ("fases_medias", "float", "–", S_CAT, "sup",
         "Número medio de fases usadas en la localización (−1 si no hubo eventos)",
         "Cantidad de lecturas por evento, otra medida de calidad de la localización."),
        ("sin_eventos", "int", "0/1", S_DER, "sup",
         "1 si el año no tuvo sismos con M >= 2,5",
         "Marca explícita de ausencia de actividad; muchos municipios tienen años en cero."),
        ("n_m25_prom_3a", "float", "sismos", S_DER, "sup",
         "Media de n_m25 en los tres años previos (t−1, t−2, t−3)",
         "Memoria reciente de la actividad, construida solo con años anteriores a t."),
        ("tendencia", "float", "–", S_DER, "sup",
         "Cociente n_m25 / (n_m25_prom_3a + 1)",
         "Indica si el año está por encima o por debajo de su historia reciente."),
        ("mag_max_historica", "float", "M", S_CAT, "sup",
         "Máximo de magnitud (M >= 2,5) desde 1993 hasta el año, inclusive",
         "Recoge el mayor evento conocido, que condiciona el riesgo aunque el año esté tranquilo."),
        ("dist_nido_km", "float", "km", S_DER, "sup",
         "Distancia haversine de la cabecera al nido sísmico de Bucaramanga",
         "Mide la cercanía a la fuente sísmica principal de la región."),
        ("lat", "float", "grados", S_DIV, "sup",
         "Latitud de la cabecera municipal",
         "Ubicación geográfica del municipio; junto con lon aporta la posición espacial."),
        ("lon", "float", "grados", S_DIV, "sup",
         "Longitud de la cabecera municipal",
         "Ubicación geográfica del municipio; junto con lat aporta la posición espacial."),
        ("periodo_red", "int", "0/1", S_DER, "sup",
         "1 desde 2018 (catálogo SeisComP3), 0 antes",
         "Controla el cambio de red y de catálogo para que el modelo no lo confunda con actividad real."),
        # territorial
        ("es_santander", "int", "0/1", S_DIV, "territorial",
         "1 si el municipio pertenece a Santander (código de departamento 68)",
         "Separa la zona de estudio de los municipios vecinos."),
        # exposición
        ("poblacion_total", "int", "hab.", S_POB, "exposicion",
         "Población total del municipio en 2020",
         "Mide cuánta población quedaría expuesta; entra solo al ranking del agente RL."),
        ("pct_rural", "float", "0–1", S_POB, "exposicion",
         "Fracción de población rural en 2020",
         "Indica exposición diferenciada entre zonas urbanas y rurales."),
        # vulnerabilidad
        ("emergencias_prom_anual", "float", "eventos/año", S_UNG, "vulnerabilidad",
         "Emergencias UNGRD registradas por año, en promedio",
         "Proxy de la vulnerabilidad histórica; fuera del supervisado para evitar fuga temporal."),
        ("mov_masa_total", "int", "eventos", S_UNG, "vulnerabilidad",
         "Emergencias UNGRD por movimiento en masa, acumuladas",
         "Los movimientos en masa acompañan a la sismicidad fuerte; se usa en la recompensa RL."),
        ("viviendas_afectadas_total", "int", "viviendas", S_UNG, "vulnerabilidad",
         "Viviendas destruidas más averiadas, acumuladas en UNGRD",
         "Mide el daño material registrado en el territorio."),
        ("personas_afectadas_total", "int", "personas", S_UNG, "vulnerabilidad",
         "Personas afectadas, acumuladas en UNGRD",
         "Mide el impacto humano registrado en el territorio."),
        # objetivo
        ("nivel_actividad_t1", "str", "–", S_DER, "objetivo",
         "Clase bajo/medio/alto de n_m25 del año siguiente (0, 1–4, ≥5); NaN en 2025",
         "Es la variable a predecir; los umbrales salen de config y definen el riesgo operativo."),
        # auxiliares futuros (nunca entran como features)
        ("n_m25_t1", "float", "sismos", S_DER, "auxiliar_futuro",
         "n_m25 del año siguiente; NaN en 2025",
         "Permite verificar la clase real del objetivo; nunca es entrada del modelo."),
        ("log_energia_t1", "float", "log10 J", S_DER, "auxiliar_futuro",
         "log_energia_j del año siguiente; NaN en 2025",
         "Es la base de la amenaza en la recompensa del agente RL; nunca es entrada del modelo."),
    ]
    return pd.DataFrame(
        filas,
        columns=["variable", "tipo", "unidad", "fuente", "rol", "descripcion", "justificacion"],
    )


def _validar(df, n_municipios):
    """Validaciones de CONTRATOS §4 (con assert, como pide el contrato)."""
    assert not df.duplicated(subset=["cod_mpio", "anio"]).any(), "hay filas duplicadas por (cod_mpio, anio)"
    nulos = [c for c in config.FEATURES_SUP if df[c].isna().any()]
    assert not nulos, f"NaN en FEATURES_SUP: {nulos}"
    es_nan_objetivo = df[config.OBJETIVO].isna()
    assert (es_nan_objetivo == (df["anio"] == config.ANIO_FIN)).all(), \
        "nivel_actividad_t1 debe ser NaN si y solo si anio == ANIO_FIN"
    n_variables = len(df.columns) - len(LLAVES)
    assert n_variables >= 20, f"solo {n_variables} variables (se piden al menos 20)"
    n_esperado = n_municipios * (config.ANIO_FIN - config.ANIO_INICIO + 1)
    assert len(df) == n_esperado, f"{len(df)} filas; se esperaban {n_esperado}"
    print(f"{PREFIJO} validaciones OK: {len(df)} filas, {n_variables} variables, sin duplicados, sin NaN en FEATURES_SUP")


def construir_dataset():
    """Encadena todo el pipeline T1.3 y guarda los 4 artefactos. Devuelve el dataset final."""
    eventos = catalogo.construir_eventos()
    municipios = territorio.construir_municipios()
    eventos = territorio.asignar_municipio(eventos, municipios)
    eventos.to_csv(config.EVENTOS_CSV, index=False)
    municipios.to_csv(config.MUNICIPIOS_CSV, index=False)
    print(f"{PREFIJO} guardado {config.EVENTOS_CSV.name} ({len(eventos)} filas) y {config.MUNICIPIOS_CSV.name}")

    panel = agregar_anual(eventos, municipios)
    panel = agregar_rezagos(panel)
    panel = agregar_objetivo(panel)
    panel = unir_territorio(panel, municipios)
    panel = panel.loc[panel["anio"].between(config.ANIO_INICIO, config.ANIO_FIN)].copy()

    orden = LLAVES + config.FEATURES_SUP + ["es_santander"] + config.FEATURES_EXPOSICION \
        + config.FEATURES_VULNERABILIDAD + [config.OBJETIVO] + config.AUXILIARES_FUTURO
    panel = panel[orden].sort_values(["cod_mpio", "anio"]).reset_index(drop=True)

    dic = diccionario_datos()
    assert set(dic["variable"]) == set(panel.columns), "el diccionario no coincide con las columnas del dataset"
    _validar(panel, n_municipios=len(municipios))

    panel.to_csv(config.DATASET_CSV, index=False)
    dic.to_csv(config.DICCIONARIO_CSV, index=False)
    print(f"{PREFIJO} guardado {config.DATASET_CSV.name} y {config.DICCIONARIO_CSV.name}")
    return panel
