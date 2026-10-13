"""Agente de política lineal con REINFORCE, línea base y bonificación de entropía.

Contrato: docs/uml/03_clases.md ("Semántica clave de las clases RL").

Política: pi(i | s) = softmax(w · x_i) sobre los municipios disponibles.
Actualización: REINFORCE con retorno G_t (gamma = 1), línea base por paso t
(media móvil exponencial de G_t) y bonificación de entropía.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sismo import config

_K_MAX_INICIAL = 200     # tamaño inicial de la línea base (pasos por episodio)
_ALPHA_BASELINE = 0.05   # factor de la media móvil exponencial de la línea base
_NORMA_MAX = 10.0        # recorte de la norma del gradiente acumulado


class AgentePoliticaLineal:
    """Política softmax lineal sobre las características de cada municipio."""

    def __init__(
        self,
        n_feat: int,
        lr: float = config.RL_LR,
        entropia: float = config.RL_ENTROPIA,
        seed: int = 0,
    ):
        self.n_feat = int(n_feat)
        self.w = np.zeros(self.n_feat, dtype=np.float64)
        self.lr = float(lr)
        self.entropia = float(entropia)
        self.rng = np.random.default_rng(seed)
        self._baseline = np.zeros(_K_MAX_INICIAL, dtype=np.float64)

    # ------------------------------------------------------------------ política
    def puntajes(self, x: np.ndarray) -> np.ndarray:
        """Puntaje lineal w · x_i de cada municipio."""
        return np.asarray(x, dtype=np.float64) @ self.w

    def politica(self, x: np.ndarray, mascara: np.ndarray) -> np.ndarray:
        """Probabilidades softmax sobre los disponibles. Los enmascarados valen 0.

        Se resta el máximo antes de exponenciar para evitar desbordamientos.
        """
        mascara = np.asarray(mascara, dtype=bool)
        if not mascara.any():
            raise ValueError("no hay municipios disponibles para elegir")
        puntajes = self.puntajes(x)
        probs = np.zeros_like(puntajes)
        disponibles = puntajes[mascara]
        e = np.exp(disponibles - disponibles.max())
        probs[mascara] = e / e.sum()
        return probs

    def actuar(self, x: np.ndarray, mascara: np.ndarray, greedy: bool = False) -> int:
        """Elige un municipio: argmax si greedy, muestreo según la política si no."""
        probs = self.politica(x, mascara)
        if greedy:
            return int(np.argmax(probs))
        return int(self.rng.choice(len(probs), p=probs))

    # ------------------------------------------------------------------ aprendizaje
    def actualizar(self, trayectoria: list[tuple]) -> float:
        """Aplica un paso REINFORCE con línea base sobre un episodio completo.

        trayectoria: lista de (x, mascara, accion, recompensa).
        Devuelve la recompensa total del episodio.
        """
        if not trayectoria:
            return 0.0
        recompensas = np.array([tr[3] for tr in trayectoria], dtype=np.float64)
        # G_t = suma de recompensas desde t hasta el final (gamma = 1)
        retornos = np.cumsum(recompensas[::-1])[::-1]

        if len(trayectoria) > len(self._baseline):
            # Episodios más largos que k_max: se amplía la línea base con ceros.
            extra = np.zeros(len(trayectoria) - len(self._baseline), dtype=np.float64)
            self._baseline = np.concatenate([self._baseline, extra])

        gradiente = np.zeros(self.n_feat, dtype=np.float64)
        for t, (x, mascara, accion, _) in enumerate(trayectoria):
            x = np.asarray(x, dtype=np.float64)
            mascara = np.asarray(mascara, dtype=bool)
            probs = self.politica(x, mascara)
            xbar = probs @ x  # sum_j pi_j x_j (los enmascarados tienen pi = 0)

            # grad log pi(a) = x_a - sum_j pi_j x_j
            grad_logpi = x[accion] - xbar

            # grad H = -sum_j pi_j log(pi_j) (x_j - xbar), solo sobre disponibles
            p = probs[mascara]
            xd = x[mascara]
            with np.errstate(divide="ignore", invalid="ignore"):
                plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
            grad_entropia = -np.sum(plogp[:, None] * (xd - xbar), axis=0)

            ventaja = retornos[t] - self._baseline[t]
            gradiente += ventaja * grad_logpi + self.entropia * grad_entropia

            # Actualización de la línea base del paso t (media móvil exponencial)
            self._baseline[t] += _ALPHA_BASELINE * (retornos[t] - self._baseline[t])

        norma = np.linalg.norm(gradiente)
        if norma > _NORMA_MAX:
            gradiente *= _NORMA_MAX / norma

        self.w += self.lr * gradiente
        return float(recompensas.sum())

    # ------------------------------------------------------------------ explicación
    def contribuciones(self, x: np.ndarray) -> np.ndarray:
        """Contribución de cada criterio: x_ij * w_j, forma (n_mpios, n_feat)."""
        return np.asarray(x, dtype=np.float64) * self.w

    # ------------------------------------------------------------------ persistencia
    def guardar(self, path: Path) -> None:
        """Guarda pesos, hiperparámetros y línea base en un .npz.

        Se abre el archivo explícitamente para que np.savez no añada la extensión .npz.
        """
        with open(path, "wb") as f:
            np.savez(f, w=self.w, lr=self.lr, entropia=self.entropia, baseline=self._baseline)

    @classmethod
    def cargar(cls, path: Path) -> "AgentePoliticaLineal":
        """Reconstruye un agente desde un .npz generado por guardar."""
        with np.load(path) as datos:
            w = datos["w"].astype(np.float64)
            agente = cls(
                n_feat=len(w),
                lr=float(datos["lr"]),
                entropia=float(datos["entropia"]),
            )
            agente.w = w.copy()
            agente._baseline = datos["baseline"].astype(np.float64).copy()
        return agente
