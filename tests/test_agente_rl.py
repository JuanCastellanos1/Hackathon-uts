"""Pruebas de AgentePoliticaLineal con datos sintéticos (solo para probar la mecánica)."""
import time

import numpy as np
import pytest

from sismo.agente_rl import AgentePoliticaLineal
from sismo.entorno_rl import PriorizacionEnv

N_MPIOS = 40
N_FEAT = 5
K = 10
ANIOS = list(range(2000, 2008))  # 8 años


def _entorno_sintetico(seed=0):
    """Recompensa = max(x[:, 0], 0): solo la primera característica importa."""
    rng = np.random.default_rng(seed)
    estados = {a: rng.normal(size=(N_MPIOS, N_FEAT)) for a in ANIOS}
    recompensas = {a: np.maximum(estados[a][:, 0], 0.0) for a in ANIOS}
    return PriorizacionEnv(estados, recompensas, ANIOS, k=K)


def test_probabilidades_suman_uno_y_enmascarados_son_cero():
    rng = np.random.default_rng(10)
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=0)
    agente.w = rng.normal(size=N_FEAT)
    x = rng.normal(size=(N_MPIOS, N_FEAT))
    mascara = rng.random(N_MPIOS) > 0.4
    mascara[0] = True  # al menos un disponible

    probs = agente.politica(x, mascara)
    assert probs.shape == (N_MPIOS,)
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs[~mascara] == 0.0)
    assert np.all(probs[mascara] > 0.0)


def test_politica_estable_con_puntajes_grandes():
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=0)
    agente.w = np.full(N_FEAT, 500.0)
    x = np.ones((N_MPIOS, N_FEAT))
    x[3] = 2.0  # puntaje enorme para un municipio
    probs = agente.politica(x, np.ones(N_MPIOS, dtype=bool))
    assert np.all(np.isfinite(probs))
    assert probs.sum() == pytest.approx(1.0)
    assert np.argmax(probs) == 3


def test_actuar_nunca_elige_enmascarados():
    rng = np.random.default_rng(11)
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=1)
    x = rng.normal(size=(N_MPIOS, N_FEAT))
    mascara = np.zeros(N_MPIOS, dtype=bool)
    mascara[[4, 17, 33]] = True
    for _ in range(200):
        assert mascara[agente.actuar(x, mascara)]
    assert agente.actuar(x, mascara, greedy=True) in (4, 17, 33)


def test_contribuciones_forma_y_valor():
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=0)
    agente.w = np.array([1.0, -2.0, 0.0, 0.5, 3.0])
    x = np.arange(N_MPIOS * N_FEAT, dtype=float).reshape(N_MPIOS, N_FEAT)
    contrib = agente.contribuciones(x)
    assert contrib.shape == (N_MPIOS, N_FEAT)
    assert np.allclose(contrib, x * agente.w)
    assert np.allclose(contrib.sum(axis=1), x @ agente.w)


def test_actualizar_devuelve_recompensa_total():
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=0)
    rng = np.random.default_rng(12)
    trayectoria = []
    for r in [1.0, 0.5, -1.0]:
        x = rng.normal(size=(N_MPIOS, N_FEAT))
        trayectoria.append((x, np.ones(N_MPIOS, dtype=bool), 0, r))
    assert agente.actualizar(trayectoria) == pytest.approx(0.5)


def test_aprendizaje_recupera_la_caracteristica_util():
    """Con recompensa = max(x0, 0), 300 episodios deben dar w[0] > 0 y > 90 % del oráculo."""
    inicio = time.time()
    env = _entorno_sintetico(seed=0)
    agente = AgentePoliticaLineal(n_feat=N_FEAT, seed=0)

    for _ in range(300):
        obs, info = env.reset()
        trayectoria = []
        terminado = False
        while not terminado:
            x = obs[:, :N_FEAT]
            mascara = info["mascara"]
            a = agente.actuar(x, mascara, greedy=False)
            obs_sig, r, terminado, _, info_sig = env.step(a)
            trayectoria.append((x.copy(), mascara.copy(), a, r))
            obs, info = obs_sig, info_sig
        agente.actualizar(trayectoria)

    # Evaluación greedy sobre los 8 años
    porcentajes = []
    for anio in ANIOS:
        obs, info = env.reset(options={"anio": anio})
        total = 0.0
        terminado = False
        while not terminado:
            a = agente.actuar(obs[:, :N_FEAT], info["mascara"], greedy=True)
            obs, r, terminado, _, info = env.step(a)
            total += r
        porcentajes.append(total / env.oraculo(anio))
    media = float(np.mean(porcentajes))
    duracion = time.time() - inicio

    print(f"\n[aprendizaje] w={np.round(agente.w, 3)} %oraculo_greedy={media:.3f} t={duracion:.1f}s")
    assert agente.w[0] > 0
    assert media > 0.9
    assert duracion < 60


def test_guardar_y_cargar_ida_y_vuelta(tmp_path):
    agente = AgentePoliticaLineal(n_feat=N_FEAT, lr=0.123, entropia=0.007, seed=3)
    rng = np.random.default_rng(13)
    agente.w = rng.normal(size=N_FEAT)
    agente._baseline[:5] = rng.normal(size=5)

    ruta = tmp_path / "agente.npz"
    agente.guardar(ruta)
    cargado = AgentePoliticaLineal.cargar(ruta)

    assert np.array_equal(cargado.w, agente.w)
    assert cargado.lr == agente.lr
    assert cargado.entropia == agente.entropia
    assert np.array_equal(cargado._baseline, agente._baseline)
    x = rng.normal(size=(N_MPIOS, N_FEAT))
    assert np.allclose(cargado.puntajes(x), agente.puntajes(x))
