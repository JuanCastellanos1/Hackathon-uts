"""Pruebas del módulo features (T1.3). Contrato: docs/CONTRATOS_DATOS.md §4."""
import numpy as np
import pandas as pd
import pytest

from sismo import features, config


def test_b_value_gutenberg_richter_b1():
    rng = np.random.default_rng(0)
    mc = 2.5
    mags = mc + rng.exponential(1 / np.log(10), 5000)   # G-R con b = 1
    b = features.b_value_aki(mags, mc)
    assert abs(b - 1.0) <= 0.15


def test_b_value_nan_con_menos_de_dos_eventos():
    assert np.isnan(features.b_value_aki([3.0], 2.5))
    assert np.isnan(features.b_value_aki([], 2.5))


def test_b_value_ignora_magnitudes_bajo_mc():
    limpio = features.b_value_aki([2.6, 3.0, 3.4, 2.9], 2.5)
    con_ruido = features.b_value_aki([2.6, 3.0, 3.4, 2.9, 1.0, 0.5], 2.5)
    assert limpio == pytest.approx(con_ruido)


def test_nivel_actividad_umbrales():
    res = features.nivel_actividad(pd.Series([0, 1, 4, 5, 20]))
    assert res.tolist() == ["bajo", "medio", "medio", "alto", "alto"]


def test_nivel_actividad_conserva_nan():
    res = features.nivel_actividad(pd.Series([np.nan, 0.0, 7.0]))
    assert pd.isna(res.iloc[0])
    assert res.iloc[1] == "bajo" and res.iloc[2] == "alto"


def _panel_sintetico(anios=range(2000, 2004)):
    """Dos municipios con n_m25 distintos por año (controlado para verificar el desplazamiento)."""
    filas = []
    for m, extra in [("68001", 0), ("68002", 1)]:
        for anio in anios:
            filas.append({"cod_mpio": m, "anio": anio, "n_m25": anio * 10 + extra,
                          "log_energia_j": float(anio - 2000)})
    return pd.DataFrame(filas)


def test_objetivo_es_n_m25_del_anio_siguiente_y_nan_en_ultimo():
    panel = features.agregar_objetivo(_panel_sintetico())
    for m in ["68001", "68002"]:
        for anio in [2000, 2001, 2002]:
            t = panel.loc[(panel.cod_mpio == m) & (panel.anio == anio)]
            t1 = panel.loc[(panel.cod_mpio == m) & (panel.anio == anio + 1)]
            assert t["n_m25_t1"].item() == t1["n_m25"].item()
            assert t["log_energia_t1"].item() == t1["log_energia_j"].item()
    ultimo = panel.loc[panel.anio == 2003]
    assert ultimo["n_m25_t1"].isna().all()
    assert ultimo["log_energia_t1"].isna().all()
    assert ultimo["nivel_actividad_t1"].isna().all()


def test_objetivo_clase_coincide_con_umbrales():
    panel = features.agregar_objetivo(_panel_sintetico())
    # Para 68001 en 2001 el n_m25 de 2002 es 20 -> alto; en 2000 el de 2001 es 10 -> alto.
    fila = panel.loc[(panel.cod_mpio == "68001") & (panel.anio == 2000)]
    assert fila["nivel_actividad_t1"].item() == "alto"


def test_rezagos_usan_solo_anios_anteriores():
    base = pd.DataFrame({"cod_mpio": ["68001"] * 6, "anio": range(2015, 2021),
                         "n_m25": [1, 2, 3, 4, 5, 6]})
    p = features.agregar_rezagos(base)
    # En 2018 la media de 2015..2017 = (1+2+3)/3 = 2.0; la tendencia = 4 / (2 + 1).
    fila = p.loc[p.anio == 2018].iloc[0]
    assert fila["n_m25_prom_3a"] == pytest.approx(2.0)
    assert fila["tendencia"] == pytest.approx(4 / 3)
    assert p.loc[p.anio == 2017, "periodo_red"].item() == 0
    assert p.loc[p.anio == 2018, "periodo_red"].item() == 1


def test_ninguna_feature_de_entrada_termina_en_t1():
    assert not any(c.endswith("_t1") for c in config.FEATURES_SUP)


def test_diccionario_una_fila_por_variable_con_justificacion():
    dic = features.diccionario_datos()
    assert len(dic) == 39
    assert dic["variable"].is_unique
    assert set(dic["rol"]) <= {"llave", "sup", "territorial", "exposicion",
                               "vulnerabilidad", "objetivo", "auxiliar_futuro"}
    assert set(dic["fuente"]) <= {"SGC catálogo RSNC", "DANE DIVIPOLA", "DANE proyecciones población 2020",
                                  "UNGRD emergencias (datos.gov.co)", "Derivada"}
    assert dic["justificacion"].str.strip().str.len().gt(0).all()
    assert set(config.FEATURES_SUP) <= set(dic["variable"])


@pytest.mark.skipif(not config.DATASET_CSV.exists(),
                    reason="dataset no generado; ejecutar src/03_tabla.py")
def test_dataset_final_cumple_contrato():
    df = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    assert len(df) == 4224
    assert df["cod_mpio"].str.len().eq(5).all()
    assert not df.duplicated(subset=["cod_mpio", "anio"]).any()
    assert df[config.FEATURES_SUP].notna().all().all()
    assert (df[config.OBJETIVO].isna() == (df["anio"] == config.ANIO_FIN)).all()
