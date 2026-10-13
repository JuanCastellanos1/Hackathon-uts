"""Tests del componente no supervisado (T2.2). Usan los datos reales del proyecto."""
import pandas as pd
import pytest

from sismo import config, no_supervisado


@pytest.fixture(scope="module")
def datos():
    df = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    eventos = pd.read_csv(config.EVENTOS_CSV, dtype={"cod_mpio": str})
    perfil = no_supervisado.perfil_municipal(df, (config.ANIO_INICIO, config.ANIO_FIN), eventos)
    return df, eventos, perfil


def test_perfil_176_filas_sin_nan(datos):
    _, _, perfil = datos
    assert len(perfil) == 176
    assert perfil["cod_mpio"].nunique() == 176
    assert perfil[config.FEATURES_PERFIL].notna().all().all()


def test_etiquetas_en_rango(datos):
    _, _, perfil = datos
    k = 4
    ajustado, _ = no_supervisado.ajustar_clusters(perfil, k)
    assert ajustado["cluster"].between(0, k - 1).all()
    assert ajustado["cluster"].nunique() == k


def test_nombrar_clusters_nombres_unicos(datos):
    _, _, perfil = datos
    k = 4
    ajustado, _ = no_supervisado.ajustar_clusters(perfil, k)
    nombres = no_supervisado.nombrar_clusters(ajustado)
    assert set(nombres.keys()) == set(range(k))
    assert len(set(nombres.values())) == k


def test_dbscan_devuelve_al_menos_una_fuente(datos):
    _, eventos, _ = datos
    fuentes = no_supervisado.fuentes_dbscan(eventos)
    assert len(fuentes) >= 1
    assert fuentes["n"].is_monotonic_decreasing
