"""Lectura y unificación de los catálogos sísmicos del SGC.

Contrato: docs/CONTRATOS_DATOS.md §2. Diagrama de clases: docs/uml/03_clases.md (módulo `catalogo`).

- leer_catalogo_antiguo / leer_catalogo_nuevo: llevan cada CSV crudo al esquema común.
- unificar: concatena, parsea la fecha, descarta NaN y duplicados, ordena y asigna event_id.
- marcar_atipicos: agrega la bandera es_atipico.
- construir_eventos: encadena todo y devuelve el DataFrame. NO lo guarda a disco (eso lo hace T1.3).
"""
import numpy as np
import pandas as pd

from sismo import config

PREFIJO = "[catalogo]"

# Columnas comunes que devuelven los lectores (antes de anio, event_id y es_atipico).
COLUMNAS_COMUNES = [
    "fecha_hora_utc", "lat", "lon", "prof_km", "mag", "tipo_mag", "mw", "fases",
    "rms_s", "gap_grados", "err_lat_km", "err_lon_km", "err_prof_km",
    "municipio_sgc", "catalogo", "periodo_red",
]

# Esquema final: CONTRATOS §2 menos cod_mpio y dist_cabecera_km (que agrega T1.3).
COLUMNAS_EVENTOS = [
    "event_id", "fecha_hora_utc", "anio", "lat", "lon", "prof_km", "mag", "tipo_mag",
    "mw", "fases", "rms_s", "gap_grados", "err_lat_km", "err_lon_km", "err_prof_km",
    "municipio_sgc", "catalogo", "periodo_red", "es_atipico",
]

COLUMNAS_NUMERICAS = [
    "lat", "lon", "prof_km", "mag", "mw", "fases", "rms_s", "gap_grados",
    "err_lat_km", "err_lon_km", "err_prof_km", "periodo_red",
]
CLAVE_NAN = ["lat", "lon", "prof_km", "mag"]
CLAVE_DUPLICADOS = ["fecha_hora_utc", "lat", "lon", "prof_km", "mag"]


def leer_catalogo_antiguo(path):
    """Lee el catálogo 1993–2018 (earthworm/SEISAN) y lo lleva al esquema común.

    `mag` es MAGNITUD Ml, `tipo_mag` es "Ml" y `mw` es MAGNITUD Mw (puede ser NaN).
    """
    crudo = pd.read_csv(path)
    df = pd.DataFrame({
        "fecha_hora_utc": crudo["FECHA"].astype(str) + " " + crudo["HORA_UTC"].astype(str),
        "lat": crudo["LATITUD (grados)"],
        "lon": crudo["LONGITUD (grados)"],
        "prof_km": crudo["PROFUNDIDAD (Km)"],
        "mag": crudo["MAGNITUD Ml"],
        "tipo_mag": "Ml",
        "mw": crudo["MAGNITUD Mw"],
        "fases": crudo["# FASES"],
        "rms_s": crudo["RMS (Seg)"],
        "gap_grados": crudo["GAP (grados)"],
        "err_lat_km": crudo["ERROR LATITUD (Km)"],
        "err_lon_km": crudo["ERROR LONGITUD (Km)"],
        "err_prof_km": crudo["ERROR PROFUNDIDAD (Km)"],
        "municipio_sgc": crudo["MUNICIPIO"].astype(str) + " - " + crudo["DEPARTAMENTO"].astype(str),
        "catalogo": "earthworm_seisan",
        "periodo_red": 0,
    })
    return df[COLUMNAS_COMUNES]


def leer_catalogo_nuevo(path):
    """Lee el catálogo 2018–hoy (SeisComP3) y lo lleva al esquema común.

    Este catálogo no trae magnitud Mw, así que `mw` queda NaN.
    """
    crudo = pd.read_csv(path)
    df = pd.DataFrame({
        "fecha_hora_utc": crudo["fecha_hora_utc"].astype(str),
        "lat": crudo["lat"],
        "lon": crudo["lon"],
        "prof_km": crudo["prof_km"],
        "mag": crudo["mag"],
        "tipo_mag": crudo["tipo_mag"],
        "mw": np.nan,
        "fases": crudo["fases"],
        "rms_s": crudo["rms_s"],
        "gap_grados": crudo["gap_grados"],
        "err_lat_km": crudo["err_lat_km"],
        "err_lon_km": crudo["err_lon_km"],
        "err_prof_km": crudo["err_prof_km"],
        "municipio_sgc": crudo["region"],
        "catalogo": "seiscomp3",
        "periodo_red": 1,
    })
    return df[COLUMNAS_COMUNES]


def unificar(antiguo, nuevo):
    """Concatena ambos catálogos y deja el DataFrame de eventos sin marca de atípico.

    Pasos: parsear fecha y crear `anio`, convertir numéricos (coerce), descartar filas con
    NaN en lat/lon/prof_km/mag, eliminar duplicados exactos sobre (fecha_hora_utc, lat, lon,
    prof_km, mag), ordenar por fecha y asignar `event_id = f"{catalogo}_{i:07d}"`.
    """
    df = pd.concat([antiguo, nuevo], ignore_index=True)
    print(f"{PREFIJO} filas antes de limpiar: {len(df)}")

    # Strict: si una fecha no tiene el formato esperado, el error debe verse de inmediato.
    fecha = pd.to_datetime(df["fecha_hora_utc"], format="%Y-%m-%d %H:%M:%S")
    df["anio"] = fecha.dt.year.astype(int)

    for col in COLUMNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sin_nan = df[CLAVE_NAN].isna().any(axis=1)
    print(f"{PREFIJO} descartadas {int(sin_nan.sum())} filas con NaN en {'/'.join(CLAVE_NAN)}")
    df = df.loc[~sin_nan].copy()

    duplicado = df.duplicated(subset=CLAVE_DUPLICADOS, keep="first")
    print(f"{PREFIJO} eliminados {int(duplicado.sum())} duplicados exactos "
          f"sobre ({', '.join(CLAVE_DUPLICADOS)})")
    df = df.loc[~duplicado].copy()

    df = df.sort_values("fecha_hora_utc", kind="mergesort").reset_index(drop=True)
    df["event_id"] = [f"{cat}_{i:07d}" for i, cat in enumerate(df["catalogo"])]

    return df[[c for c in COLUMNAS_EVENTOS if c != "es_atipico"]]


def marcar_atipicos(eventos):
    """Agrega es_atipico = (rms_s > RMS_MAX) | (err_prof_km > ERR_PROF_MAX).

    Un NaN en cualquiera de las dos columnas no cuenta como atípico.
    """
    eventos = eventos.copy()
    eventos["es_atipico"] = (
        (eventos["rms_s"] > config.RMS_MAX) | (eventos["err_prof_km"] > config.ERR_PROF_MAX)
    ).astype(bool)
    return eventos


def construir_eventos():
    """Encadena lectura, unificación y marca de atípicos. Devuelve el DataFrame (no guarda)."""
    antiguo = leer_catalogo_antiguo(config.RAW_SGC_ANTIGUO)
    nuevo = leer_catalogo_nuevo(config.RAW_SGC_NUEVO)
    print(f"{PREFIJO} leídas: earthworm_seisan={len(antiguo)} filas, seiscomp3={len(nuevo)} filas")

    eventos = marcar_atipicos(unificar(antiguo, nuevo))

    for cat, n in eventos["catalogo"].value_counts().sort_index().items():
        print(f"{PREFIJO} {cat}: {int(n)} eventos limpios")
    print(f"{PREFIJO} total eventos: {len(eventos)}; atípicos (es_atipico): "
          f"{int(eventos['es_atipico'].sum())}")
    return eventos
