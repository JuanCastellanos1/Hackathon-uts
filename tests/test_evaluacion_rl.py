"""Pruebas de evaluacion_rl con datos sintéticos (solo para probar la mecánica)."""
import numpy as np
import pandas as pd
import pytest

from sismo import config, evaluacion_rl
from sismo.agente_rl import AgentePoliticaLineal
from sismo.entorno_rl import PriorizacionEnv

N_MUNICIPIOS = 87
ANIOS_ESTADO = list(range(2006, 2026))  # 2006–2025, como en el pipeline real
K_CLUSTERS = 5


def _datos_sinteticos(seed=0):
    """Panel, predicciones y clusters con la forma del pipeline real (87 municipios de Santander)."""
    rng = np.random.default_rng(seed)
    cods = [f"68{i:03d}" for i in range(N_MUNICIPIOS)]
    filas_df, filas_pred = [], []
    for anio in range(2002, 2026):
        for i, cod in enumerate(cods):
            lt1 = rng.uniform(0, 12) if anio < 2025 else np.nan
            filas_df.append({
                "cod_mpio": cod, "nom_mpio": f"Municipio {i}", "anio": anio, "es_santander": 1,
                "lat": 6.5 + 0.01 * i, "lon": -73.0 - 0.01 * i,
                "n_m25": int(rng.integers(0, 6)), "n_m25_prom_3a": rng.uniform(0, 20),
                "mag_max_historica": rng.uniform(3, 6), "dist_nido_km": rng.uniform(0, 150),
                "poblacion_total": rng.uniform(1e3, 5e5), "pct_rural": rng.uniform(0, 1),
                "emergencias_prom_anual": rng.uniform(0, 15), "mov_masa_total": rng.uniform(0, 20),
                "viviendas_afectadas_total": rng.uniform(0, 3000), "log_energia_t1": lt1,
            })
            if anio >= 2006:
                p_alto = rng.uniform(0, 1)
                p_medio = rng.uniform(0, 1 - p_alto)
                filas_pred.append({"cod_mpio": cod, "anio": anio, "p_bajo": 1 - p_alto - p_medio,
                                   "p_medio": p_medio, "p_alto": p_alto, "pred": 0, "y_true": np.nan})
    df = pd.DataFrame(filas_df)
    pred = pd.DataFrame(filas_pred)
    clusters = pd.DataFrame({
        "cod_mpio": cods,
        "cluster_rl": [i % K_CLUSTERS for i in range(N_MUNICIPIOS)],
        "cluster_nombre": [f"c{i % K_CLUSTERS}" for i in range(N_MUNICIPIOS)],
    })
    return df, pred, clusters


@pytest.fixture(scope="module")
def estados_sinteticos():
    df, pred, clusters = _datos_sinteticos()
    return evaluacion_rl.construir_estados(df, pred, clusters, ANIOS_ESTADO)


def test_construir_estados_87_filas_y_sin_variables_t1(estados_sinteticos):
    estados, recompensas, nombres, extra = estados_sinteticos
    assert set(estados) == set(ANIOS_ESTADO)
    for anio in ANIOS_ESTADO:
        assert estados[anio].shape == (N_MUNICIPIOS, len(nombres))
        assert recompensas[anio].shape == (N_MUNICIPIOS,)
    assert not any(n.endswith("_t1") for n in nombres)
    assert len(extra["media"]) == len(nombres) == len(extra["std"])
    assert np.all(extra["std"] > 0)
    assert len([n for n in nombres if n.startswith("cluster_rl_")]) == K_CLUSTERS


def test_estado_estandarizado_con_media_cero_en_train(estados_sinteticos):
    estados, _, _, extra = estados_sinteticos
    apilado = np.vstack([estados[a] for a in extra["anios_train"]])
    assert np.allclose(apilado.mean(axis=0), 0.0, atol=1e-8)


def test_recompensas_finitas_y_no_negativas_con_objetivo(estados_sinteticos):
    _, recompensas, _, _ = estados_sinteticos
    for anio in range(2006, 2025):  # 2025 no tiene t+1
        r = recompensas[anio]
        assert np.all(np.isfinite(r))
        assert np.all(r >= 0)
        assert np.all(r <= 1)
    assert np.all(np.isnan(recompensas[2025]))


def test_politica_oraculo_da_pct_uno(estados_sinteticos):
    estados, recompensas, _, _ = estados_sinteticos
    anios = list(range(2006, 2016))
    env = PriorizacionEnv(estados, recompensas, anios=anios, k=config.K_PRESUPUESTO)
    resultado = evaluacion_rl.evaluar_politica(
        env, evaluacion_rl.politica_oraculo(recompensas), anios
    )
    assert resultado["pct_oraculo"] == pytest.approx(1.0)
    assert resultado["precision_at_k"] == pytest.approx(1.0)


def test_entrenar_devuelve_una_fila_por_epoca(estados_sinteticos):
    estados, recompensas, nombres, _ = estados_sinteticos
    anios_train = list(range(2006, 2016))
    anios_val = list(range(2016, 2020))
    env = PriorizacionEnv(estados, recompensas, anios=anios_train, k=config.K_PRESUPUESTO)
    agente = AgentePoliticaLineal(n_feat=len(nombres), lr=0.05, entropia=0.01, seed=0)
    curva = evaluacion_rl.entrenar(env, agente, 10, anios_train, anios_val, np.random.default_rng(0))
    assert list(curva.columns) == ["epoca", "recompensa_train", "pct_oraculo_val"]
    assert len(curva) == 10
    evaluadas = curva.dropna(subset=["pct_oraculo_val"])
    assert evaluadas["epoca"].tolist() == [10]
    assert np.all(np.isfinite(curva["recompensa_train"]))


def test_priorizar_87_filas_con_rangos_unicos(estados_sinteticos):
    estados, _, nombres, _ = estados_sinteticos
    df, pred, clusters = _datos_sinteticos()
    x = estados[2025]
    info = pd.DataFrame({
        "cod_mpio": clusters["cod_mpio"],
        "nom_mpio": [f"Municipio {i}" for i in range(N_MUNICIPIOS)],
        "lat": np.linspace(6.5, 7.0, N_MUNICIPIOS),
        "lon": np.linspace(-73.5, -73.0, N_MUNICIPIOS),
        "p_alto": pred[pred["anio"] == 2025]["p_alto"].to_numpy(),
        "cluster_nombre": clusters["cluster_nombre"],
        "poblacion_total": np.full(N_MUNICIPIOS, 1000.0),
        "n_m25_2025": np.zeros(N_MUNICIPIOS, dtype=int),
    })
    agentes = []
    for semilla in range(3):
        agente = AgentePoliticaLineal(n_feat=len(nombres), seed=semilla)
        agente.w = np.random.default_rng(semilla).normal(size=len(nombres))
        agentes.append(agente)

    tabla = evaluacion_rl.priorizar(agentes, x, info, nombres)
    assert len(tabla) == N_MUNICIPIOS
    assert sorted(tabla["rango"].tolist()) == list(range(1, N_MUNICIPIOS + 1))
    assert set(tabla["confianza"]) <= {"alta", "media", "baja"}
    assert tabla["prob_top_k"].between(0, 1).all()


def test_simulador_decisor_reentrena_con_otros_pesos():
    """El simulador del decisor entrena rápido y devuelve una priorización completa."""
    from sismo import evaluacion_rl
    r = evaluacion_rl.simular_decisor(peso_exposicion=0.3, k=5, semillas=(0,), epocas=20)
    pri = r["priorizacion"]
    assert len(pri) == 87 and sorted(pri["rango"]) == list(range(1, 88))
    assert 0.0 <= r["pct_oraculo_agente"] <= 1.0 and 0.0 <= r["pct_oraculo_indice"] <= 1.0
    assert r["k"] == 5
