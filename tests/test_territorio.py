"""Pruebas del módulo territorio (T1.2). Contrato: docs/CONTRATOS_DATOS.md §3."""
import numpy as np
import pandas as pd
import pytest

from sismo import territorio, config


COLUMNAS_CONTRATO = [
    "cod_mpio", "nom_mpio", "cod_dpto", "dpto", "lat", "lon", "es_santander", "dist_nido_km",
    "poblacion_total", "pct_rural", "n_emergencias_total", "emergencias_prom_anual",
    "mov_masa_total", "viviendas_afectadas_total", "personas_afectadas_total",
]


@pytest.fixture(scope="module")
def municipios():
    return territorio.construir_municipios()


def test_distancia_bucaramanga_los_santos():
    d = territorio.distancia_km(7.119, -73.122, 6.755, -73.105)
    assert isinstance(d, np.ndarray)
    assert 37.0 <= float(d) <= 43.0


def test_distancia_vectorizada_y_cero():
    lat = np.array([7.119, 6.755])
    lon = np.array([-73.122, -73.105])
    d = territorio.distancia_km(lat, lon, lat, lon)
    assert d.shape == (2,)
    assert np.allclose(d, 0.0, atol=1e-9)


def test_cargar_divipola_codigos_y_decimales():
    div = territorio.cargar_divipola(config.RAW_DIVIPOLA)
    assert len(div) == 1122
    assert div["cod_mpio"].str.len().eq(5).all()
    assert div["cod_dpto"].str.len().eq(2).all()
    assert div["lat"].dtype == float and div["lon"].dtype == float
    # Coma decimal convertida a punto; San Andrés (~-81,7) cae fuera de -80.
    assert div["lon"].between(-82, -66).all()


def test_municipios_en_caja_176_y_87_santander():
    div = territorio.cargar_divipola(config.RAW_DIVIPOLA)
    caja = territorio.municipios_en_caja(div)
    assert len(caja) == 176
    assert (caja["cod_dpto"] == "68").sum() == 87


def test_construir_municipios_forma_y_columnas(municipios):
    assert len(municipios) == 176
    assert list(municipios.columns) == COLUMNAS_CONTRATO
    assert int(municipios["es_santander"].sum()) == 87


def test_cod_mpio_unico_y_de_5_caracteres(municipios):
    assert municipios["cod_mpio"].str.len().eq(5).all()
    assert municipios["cod_mpio"].is_unique
    assert municipios["cod_mpio"].is_monotonic_increasing  # ordenado por cod_mpio


def test_poblacion_no_nula_en_al_menos_99_por_ciento(municipios):
    assert municipios["poblacion_total"].notna().mean() >= 0.99
    assert municipios["pct_rural"].dropna().between(0, 1).all()


def test_emergencias_sin_nulos_y_consistentes(municipios):
    cols = ["n_emergencias_total", "mov_masa_total", "viviendas_afectadas_total", "personas_afectadas_total"]
    assert municipios[cols].notna().all().all()
    assert (municipios[cols] >= 0).all().all()
    assert (municipios["mov_masa_total"] <= municipios["n_emergencias_total"]).all()


def test_asignar_municipio_tres_cabeceras(municipios):
    idx = municipios.set_index("cod_mpio")
    codigos = ["68001", "68081", "68679"]  # Bucaramanga, Barrancabermeja, San Gil
    eventos = pd.DataFrame(
        {
            "lat": idx.loc[codigos, "lat"].to_numpy(),
            "lon": idx.loc[codigos, "lon"].to_numpy(),
        }
    )
    salida = territorio.asignar_municipio(eventos, municipios)
    assert list(salida["cod_mpio"]) == codigos
    assert (salida["dist_cabecera_km"] < 0.01).all()
    assert "dist_cabecera_km" in salida.columns
    assert len(eventos.columns) == 2  # la entrada no se modifica


def test_asignar_municipio_coordenadas_faltantes(municipios):
    eventos = pd.DataFrame({"lat": [np.nan, 6.8], "lon": [np.nan, -73.1]})
    salida = territorio.asignar_municipio(eventos, municipios)
    assert pd.isna(salida["cod_mpio"].iloc[0])
    assert np.isnan(salida["dist_cabecera_km"].iloc[0])
    assert isinstance(salida["cod_mpio"].iloc[1], str)
