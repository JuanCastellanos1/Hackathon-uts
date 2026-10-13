"""Pruebas de PriorizacionEnv con datos sintéticos (solo para probar la mecánica)."""
import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sismo.entorno_rl import PriorizacionEnv

N_MPIOS = 40
N_FEAT = 5
K = 4
ANIOS = [2010, 2011, 2012]


def _env_sintetico(seed=0):
    rng = np.random.default_rng(seed)
    estados = {a: rng.normal(size=(N_MPIOS, N_FEAT)) for a in ANIOS}
    recompensas = {a: np.maximum(estados[a][:, 0], 0.0) for a in ANIOS}
    return PriorizacionEnv(estados, recompensas, ANIOS, k=K)


def test_check_env_pasa():
    """La API de Gymnasium se valida con su propio verificador.

    Advertencias esperadas (no son fallos): el Box de observación tiene límites
    infinitos según la especificación, y no hay spec de registro, así que no se
    prueban modos de render alternativos.
    """
    env = _env_sintetico()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Box observation space.*")
        warnings.filterwarnings("ignore", message=".*alternative render modes.*")
        check_env(env)


def test_forma_de_observacion_y_presupuesto():
    env = _env_sintetico()
    obs, info = env.reset(seed=1)
    assert obs.shape == (N_MPIOS, N_FEAT + 2)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    # Columna de elegidos a 0 y presupuesto restante a 1 al inicio
    assert np.all(obs[:, N_FEAT] == 0)
    assert np.allclose(obs[:, N_FEAT + 1], 1.0)
    assert info["seleccion"] == []
    assert info["mascara"].dtype == bool and info["mascara"].all()


def test_episodio_con_acciones_validas_termina_en_k_pasos():
    env = _env_sintetico()
    env.reset(seed=2)
    terminado = False
    pasos = 0
    while not terminado:
        disponibles = np.flatnonzero(env.mascara())
        _, _, terminado, truncado, info = env.step(int(disponibles[0]))
        pasos += 1
        assert not truncado
        if pasos < K:
            assert not terminado
    assert pasos == K
    assert terminado
    assert len(info["seleccion"]) == K
    assert len(set(info["seleccion"])) == K  # sin repetidos


def test_accion_repetida_da_menos_uno_y_no_cambia_presupuesto():
    env = _env_sintetico()
    env.reset(seed=3, options={"anio": ANIOS[0]})
    obs1, r1, term1, _, info1 = env.step(5)
    assert r1 == pytest.approx(env.recompensas[ANIOS[0]][5])
    assert not term1
    presupuesto_tras_1 = obs1[0, -1]

    obs2, r2, term2, _, info2 = env.step(5)  # misma acción otra vez
    assert r2 == -1.0
    assert not term2
    assert obs2[0, -1] == pytest.approx(presupuesto_tras_1)
    assert obs2[5, N_FEAT] == 1.0  # sigue marcado como elegido
    assert info2["seleccion"] == [5]  # no se añadió de nuevo
    assert not info2["mascara"][5]


def test_oraculo_es_suma_del_top_k():
    env = _env_sintetico()
    for anio in ANIOS:
        r = env.recompensas[anio]
        esperado = np.sort(r)[-K:].sum()
        assert env.oraculo(anio) == pytest.approx(esperado)
        top = env.topk_oraculo(anio)
        assert len(top) == K
        assert set(top.tolist()) == set(np.argsort(-r)[:K].tolist())
        assert r[top].sum() == pytest.approx(esperado)


def test_reset_con_options_fija_el_anio():
    env = _env_sintetico()
    for seed in range(5):
        for anio in ANIOS:
            _, info = env.reset(seed=seed, options={"anio": anio})
            assert info["anio"] == anio


def test_reset_sin_options_sortea_un_anio_valido():
    env = _env_sintetico()
    anios_vistos = {env.reset(seed=s)[1]["anio"] for s in range(30)}
    assert anios_vistos <= set(ANIOS)
    assert len(anios_vistos) > 1


def test_episodio_termina_y_no_se_puede_seguir_sin_reset():
    env = _env_sintetico()
    env.reset(seed=4, options={"anio": ANIOS[1]})
    for a in range(K):
        env.step(a)
    with pytest.raises(RuntimeError):
        env.step(K)
