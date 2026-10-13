"""Pruebas del módulo `catalogo` (T1.1). Contrato: docs/CONTRATOS_DATOS.md §2."""
import numpy as np
import pandas as pd

from sismo import catalogo, config


def _fila_antiguo(**kw):
    """Una fila con las columnas crudas del catálogo 1993–2018."""
    fila = {
        "FECHA": "2000-01-01", "HORA_UTC": "10:00:00",
        "LATITUD (grados)": 7.0, "LONGITUD (grados)": -73.0, "PROFUNDIDAD (Km)": 100.0,
        "MAGNITUD Ml": 3.0, "MAGNITUD Mw": np.nan,
        "DEPARTAMENTO": "SANTANDER", "MUNICIPIO": "LOS_SANTOS",
        "# FASES": 5, "RMS (Seg)": 0.5, "GAP (grados)": 100,
        "ERROR LATITUD (Km)": 2.0, "ERROR LONGITUD (Km)": 2.0, "ERROR PROFUNDIDAD (Km)": 3.0,
        "ESTADO": "Revisado",
    }
    fila.update(kw)
    return fila


def _fila_nuevo(**kw):
    """Una fila con las columnas crudas del catálogo 2018–hoy."""
    fila = {
        "fecha_hora_utc": "2020-05-01 12:00:00", "lat": 7.0, "lon": -73.0, "prof_km": 10.0,
        "mag": 2.8, "tipo_mag": "MLr_3", "fases": 12, "rms_s": 0.3, "gap_grados": 120,
        "err_lat_km": 1.0, "err_lon_km": 1.0, "err_prof_km": 2.0,
        "region": "Los Santos - Santander, Colombia",
        "quakeml": "", "fases_link": "", "mapa": "", "estado": "manual",
    }
    fila.update(kw)
    return fila


def _escribir_csv(tmp_path, nombre, filas):
    ruta = tmp_path / nombre
    pd.DataFrame(filas).to_csv(ruta, index=False)
    return ruta


def test_leer_catalogo_antiguo_renombra_al_esquema_comun(tmp_path):
    ruta = _escribir_csv(tmp_path, "antiguo.csv", [_fila_antiguo()])
    df = catalogo.leer_catalogo_antiguo(ruta)

    assert list(df.columns) == catalogo.COLUMNAS_COMUNES
    fila = df.iloc[0]
    assert fila["fecha_hora_utc"] == "2000-01-01 10:00:00"
    assert fila["lat"] == 7.0 and fila["prof_km"] == 100.0
    assert fila["mag"] == 3.0          # MAGNITUD Ml
    assert fila["tipo_mag"] == "Ml"
    assert np.isnan(fila["mw"])        # MAGNITUD Mw vacía
    assert fila["municipio_sgc"] == "LOS_SANTOS - SANTANDER"
    assert fila["catalogo"] == "earthworm_seisan"
    assert fila["periodo_red"] == 0


def test_leer_catalogo_nuevo_renombra_al_esquema_comun(tmp_path):
    ruta = _escribir_csv(tmp_path, "nuevo.csv", [_fila_nuevo(mag=3.4, tipo_mag="MLr")])
    df = catalogo.leer_catalogo_nuevo(ruta)

    assert list(df.columns) == catalogo.COLUMNAS_COMUNES
    fila = df.iloc[0]
    assert fila["fecha_hora_utc"] == "2020-05-01 12:00:00"
    assert fila["mag"] == 3.4 and fila["tipo_mag"] == "MLr"
    assert fila["rms_s"] == 0.3 and fila["err_prof_km"] == 2.0
    assert np.isnan(fila["mw"])                                  # el nuevo no trae Mw
    assert fila["municipio_sgc"] == "Los Santos - Santander, Colombia"  # viene de region
    assert fila["catalogo"] == "seiscomp3"
    assert fila["periodo_red"] == 1


def test_unificar_elimina_duplicados_y_nan_y_asigna_id(tmp_path, capsys):
    antiguo = catalogo.leer_catalogo_antiguo(_escribir_csv(tmp_path, "a.csv", [
        _fila_antiguo(),                                   # A
        _fila_antiguo(),                                   # A duplicada exacta
        _fila_antiguo(FECHA="1999-06-01", **{"MAGNITUD Ml": 4.1}),  # B, más antigua
    ]))
    nuevo = catalogo.leer_catalogo_nuevo(_escribir_csv(tmp_path, "n.csv", [
        _fila_nuevo(),                                     # C
        _fila_nuevo(lat=np.nan, fecha_hora_utc="2020-06-01 00:00:00"),  # NaN en lat: se descarta
    ]))

    df = catalogo.unificar(antiguo, nuevo)

    # 5 filas de entrada: 1 duplicada exacta y 1 con NaN en lat -> quedan 3.
    assert len(df) == 3
    salida = capsys.readouterr().out
    assert "[catalogo]" in salida
    assert "eliminados 1 duplicados" in salida
    assert "descartadas 1 filas con NaN" in salida

    # Ordenado por fecha, con anio correcto.
    assert list(df["fecha_hora_utc"]) == ["1999-06-01 10:00:00", "2000-01-01 10:00:00", "2020-05-01 12:00:00"]
    assert list(df["anio"]) == [1999, 2000, 2020]

    # event_id = f"{catalogo}_{i:07d}" y único.
    assert df["event_id"].is_unique
    assert df["event_id"].iloc[0] == "earthworm_seisan_0000000"
    assert df["event_id"].iloc[2] == "seiscomp3_0000002"

    # Esquema final sin es_atipico, que agrega marcar_atipicos.
    assert list(df.columns) == [c for c in catalogo.COLUMNAS_EVENTOS if c != "es_atipico"]


def test_marcar_atipicos_regla_rms_y_error_profundidad():
    eventos = pd.DataFrame({
        "rms_s":       [2.5, 2.0, 0.5, np.nan, 0.5, 2.1],
        "err_prof_km": [1.0, 1.0, 60.0, 1.0, np.nan, 50.0],
    })
    res = catalogo.marcar_atipicos(eventos)

    # 2.5 > RMS_MAX -> atípico; 2.0 no es > 2.0 -> no; error 60 > 50 -> atípico;
    # NaN no cuenta como atípico; 50.0 no es > 50 -> no, pero 2.1 > 2.0 -> atípico.
    assert res["es_atipico"].tolist() == [True, False, True, False, False, True]
    assert res["es_atipico"].dtype == bool
    assert "es_atipico" not in eventos.columns  # no modifica la entrada


def test_construir_eventos_datos_reales():
    eventos = catalogo.construir_eventos()

    assert len(eventos) > 200_000
    assert eventos["anio"].between(1993, 2026).all()
    for col in ["lat", "lon", "prof_km", "mag"]:
        assert eventos[col].notna().all(), f"NaN en {col}"
    assert eventos["event_id"].is_unique

    # Columnas de CONTRATOS §2 menos cod_mpio y dist_cabecera_km.
    esperadas = [
        "event_id", "fecha_hora_utc", "anio", "lat", "lon", "prof_km", "mag", "tipo_mag",
        "mw", "fases", "rms_s", "gap_grados", "err_lat_km", "err_lon_km", "err_prof_km",
        "municipio_sgc", "catalogo", "periodo_red", "es_atipico",
    ]
    assert list(eventos.columns) == esperadas
