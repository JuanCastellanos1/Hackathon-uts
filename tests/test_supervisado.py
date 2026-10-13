"""Pruebas del módulo supervisado (CONTRATOS §4 y §6)."""
import json

import numpy as np
import pandas as pd
import pytest

from sismo import config, supervisado

CLAVES_JSON = {
    "modelo_elegido", "params", "val", "test", "lineas_base_test",
    "bootstrap_f1_test", "importancias", "distribucion_clases",
}


def _df_sintetico(n_mpios=12, seed=0):
    """Tabla pequeña con la forma de CONTRATOS §4 (2002–2025, objetivo NaN en 2025)."""
    rng = np.random.default_rng(seed)
    filas = []
    for m in range(n_mpios):
        for anio in range(config.ANIO_INICIO, config.ANIO_FIN + 1):
            fila = {"cod_mpio": f"68{m:03d}", "anio": anio}
            for col in config.FEATURES_SUP:
                fila[col] = float(rng.normal())
            fila["n_m25"] = int(rng.poisson(2))
            fila[config.OBJETIVO] = (
                str(rng.choice(config.CLASES, p=[0.6, 0.3, 0.1]))
                if anio < config.ANIO_FIN else np.nan
            )
            filas.append(fila)
    return pd.DataFrame(filas)


PARAMS_RAPIDOS = {"modelo": supervisado.HGB, "params": {"learning_rate": 0.1, "max_depth": None}}


def test_dividir_temporal_sin_solape():
    partes = supervisado.dividir_temporal(_df_sintetico())
    assert set(partes) == {"train", "val", "test", "futuro"}
    anios = {k: set(v["anio"]) for k, v in partes.items()}
    assert anios["train"] == set(range(config.TRAIN[0], config.TRAIN[1] + 1))
    assert anios["val"] == set(range(config.VAL[0], config.VAL[1] + 1))
    assert anios["test"] == set(range(config.TEST[0], config.TEST[1] + 1))
    assert anios["futuro"] == {config.ANIO_FIN}
    assert not (anios["train"] & anios["val"])
    assert not (anios["val"] & anios["test"])
    assert not (anios["train"] & anios["test"])


def test_walkforward_entrena_solo_con_anios_anteriores():
    df = _df_sintetico()
    for t in range(config.WALKFORWARD_INICIO, config.ANIO_FIN + 1):
        entreno = supervisado._filas_entrenamiento_walkforward(df, t)
        assert not entreno.empty
        assert entreno["anio"].max() < t
        assert entreno[config.OBJETIVO].notna().all()


def test_walkforward_probabilidades_suman_uno_y_cubren_anios():
    df = _df_sintetico()
    wf = supervisado.predecir_walkforward(df, PARAMS_RAPIDOS)
    assert set(wf["anio"]) == set(range(config.WALKFORWARD_INICIO, config.ANIO_FIN + 1))
    suma = wf[["p_bajo", "p_medio", "p_alto"]].sum(axis=1)
    assert np.allclose(suma, 1.0, atol=1e-6)
    assert set(wf["pred"]) <= set(config.CLASES)
    assert wf.loc[wf["anio"] == config.ANIO_FIN, "y_true"].isna().all()


def test_persistencia_con_umbrales_del_contrato():
    nivel = supervisado._nivel_persistencia(np.array([0, 1, 4, 5, 12]))
    assert list(nivel) == ["bajo", "medio", "medio", "alto", "alto"]


def test_lineas_base_devuelve_metricas():
    df = _df_sintetico()
    bases = supervisado.lineas_base(df)
    assert set(bases) == {"mayoritaria", "persistencia"}
    for clave in ("mayoritaria", "persistencia"):
        assert 0.0 <= bases[clave]["f1_macro"] <= 1.0
        assert 0.0 <= bases[clave]["balanced_accuracy"] <= 1.0


def test_json_tiene_claves_del_contrato_si_existe():
    ruta = config.REPORTS / "metricas_supervisado.json"
    if not ruta.exists():
        pytest.skip("metricas_supervisado.json aún no generado")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert CLAVES_JSON <= set(datos)
    assert {"f1_macro", "balanced_accuracy", "recall_alto", "matriz_confusion", "reporte"} <= set(datos["test"])
    assert {"media", "p2_5", "p97_5"} == set(datos["bootstrap_f1_test"])
    assert set(datos["distribucion_clases"]) == {"train", "test"}
