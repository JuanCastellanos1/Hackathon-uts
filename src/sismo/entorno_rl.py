"""Entorno de priorización sísmica como problema de decisión secuencial.

Contrato: docs/uml/03_clases.md ("Semántica clave de las clases RL") y
docs/CONTRATOS_DATOS.md §6.

Cada episodio corresponde a un año. El agente elige K municipios distintos
sin reemplazo. La recompensa de cada elección es el riesgo realizado en t+1,
que el agente nunca observa en el estado.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sismo import config


class PriorizacionEnv(gym.Env):
    """Entorno de presupuesto K sobre los municipios de un año.

    Parámetros
    ----------
    estados : dict[int, ndarray (n_mpios, n_feat)]
        Estado de cada año, ya estandarizado.
    recompensas : dict[int, ndarray (n_mpios,)]
        Riesgo realizado en t+1 por municipio y año (no observable por el agente).
    anios : list[int]
        Años entre los que se sortea el episodio cuando reset no fija uno.
    k : int
        Presupuesto: número de municipios que se deben elegir.

    Observación: ndarray (n_mpios, n_feat + 2) =
    [estado | elegido (0/1) | presupuesto_restante / k].
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        estados: dict[int, np.ndarray],
        recompensas: dict[int, np.ndarray],
        anios: list[int],
        k: int = config.K_PRESUPUESTO,
    ):
        super().__init__()
        self.estados = {int(a): np.asarray(e) for a, e in estados.items()}
        self.recompensas = {int(a): np.asarray(r, dtype=np.float64) for a, r in recompensas.items()}
        self.anios = [int(a) for a in anios]
        self.k = int(k)

        if not self.anios:
            raise ValueError("anios no puede estar vacío")
        primer_anio = self.anios[0]
        self.n_mpios, self.n_feat = self.estados[primer_anio].shape
        for a in self.anios:
            if self.estados[a].shape != (self.n_mpios, self.n_feat):
                raise ValueError(f"estados[{a}] tiene forma distinta a la del año {primer_anio}")
            if self.recompensas[a].shape != (self.n_mpios,):
                raise ValueError(f"recompensas[{a}] debe tener forma ({self.n_mpios},)")
        if not 1 <= self.k <= self.n_mpios:
            raise ValueError(f"k debe estar entre 1 y n_mpios={self.n_mpios}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_mpios, self.n_feat + 2), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_mpios)

        self._anio: int = primer_anio
        self._elegidos = np.zeros(self.n_mpios, dtype=bool)
        self._seleccion: list[int] = []

    # ------------------------------------------------------------------ API Gymnasium
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Inicia un episodio. Con options={"anio": X} fija el año; si no, sortea uno."""
        super().reset(seed=seed)
        if options and "anio" in options:
            anio = int(options["anio"])
            if anio not in self.estados:
                raise ValueError(f"el año {anio} no está en estados")
        else:
            anio = int(self.np_random.choice(self.anios))
        self._anio = anio
        self._elegidos = np.zeros(self.n_mpios, dtype=bool)
        self._seleccion = []
        return self._obs(), self._info()

    def step(self, accion: int):
        """Elige un municipio. Acción repetida: recompensa -1 y sin cambios."""
        accion = int(accion)
        if not 0 <= accion < self.n_mpios:
            raise ValueError(f"acción {accion} fuera de rango [0, {self.n_mpios})")
        if self._elegidos.sum() >= self.k:
            raise RuntimeError("el episodio ya terminó; llame a reset()")

        if self._elegidos[accion]:
            recompensa = -1.0
        else:
            recompensa = float(self.recompensas[self._anio][accion])
            self._elegidos[accion] = True
            self._seleccion.append(accion)

        terminated = bool(self._elegidos.sum() == self.k)
        return self._obs(), recompensa, terminated, False, self._info()

    # ------------------------------------------------------------------ auxiliares
    def mascara(self) -> np.ndarray:
        """Máscara bool: True = municipio disponible (no elegido todavía)."""
        return ~self._elegidos

    def oraculo(self, anio: int) -> float:
        """Techo teórico: suma de las k recompensas más altas del año."""
        r = self.recompensas[int(anio)]
        return float(np.sort(r)[-self.k:].sum())

    def topk_oraculo(self, anio: int) -> np.ndarray:
        """Índices de los k municipios con mayor recompensa, de mayor a menor."""
        r = self.recompensas[int(anio)]
        return np.argsort(-r, kind="stable")[: self.k]

    # ------------------------------------------------------------------ internos
    def _obs(self) -> np.ndarray:
        estado = self.estados[self._anio].astype(np.float32)
        elegido = self._elegidos.astype(np.float32)[:, None]
        presupuesto = (self.k - int(self._elegidos.sum())) / self.k
        columna_presupuesto = np.full((self.n_mpios, 1), presupuesto, dtype=np.float32)
        return np.concatenate([estado, elegido, columna_presupuesto], axis=1)

    def _info(self) -> dict:
        return {
            "anio": self._anio,
            "mascara": self.mascara(),
            "seleccion": list(self._seleccion),
        }
