"""Paso 06: agente de aprendizaje por refuerzo para priorización sísmica (CONTRATOS §6).

Ejecutar desde la raíz del proyecto (con el entorno virtual activado): python src/06_agente_rl.py
Llama a sismo.evaluacion_rl.ejecutar(), que guarda models/agente_rl.npz,
reports/metricas_rl.json, reports/curva_aprendizaje_rl.csv, data/processed/priorizacion_2026.csv
y 4 figuras en reports/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from sismo import config, evaluacion_rl  # noqa: E402

PREFIJO = "[rl]"
ORDEN_TABLA = [
    ("agente_rl", "Agente RL"),
    ("indice_pesos_iguales", "Índice pesos iguales"),
    ("ranking_supervisado", "Ranking supervisado"),
    ("mayor_actividad_pasada", "Mayor actividad pasada"),
    ("mayor_poblacion", "Mayor población"),
    ("azar", "Azar (20 rep.)"),
    ("oraculo", "Oráculo"),
]


def main():
    r = evaluacion_rl.ejecutar()
    m, test = r["metricas"], r["metricas"]["test"]

    print(f"{PREFIJO} ==== TEST {config.TEST_RL[0]}–{config.TEST_RL[1]} (greedy, K={config.K_PRESUPUESTO}) ====")
    print(f"{PREFIJO} {'política':<22}{'recompensa media':>18}{'% oráculo':>12}{'precision@K':>13}")
    for clave, etiqueta in ORDEN_TABLA:
        fila = test[clave]
        pct = f"{fila['pct_oraculo'] * 100:>10.2f} %" if "pct_oraculo" in fila else f"{'100.00 %':>12}"
        prec = f"{fila['precision_at_k']:>13.3f}" if "precision_at_k" in fila else f"{'-':>13}"
        print(f"{PREFIJO} {etiqueta:<22}{fila['recompensa_media']:>18.4f}{pct}{prec}")
    print(f"{PREFIJO} agente RL: std entre semillas del %oráculo = "
          f"{test['agente_rl']['std_semillas'] * 100:.2f} puntos")

    print(f"{PREFIJO} ==== Pesos de la política (escala estandarizada, media ± std) ====")
    pesos = m["pesos_politica"]
    for nombre, v in sorted(pesos.items(), key=lambda kv: -abs(kv[1]["media"])):
        signo = "+" if v["media"] >= 0 else "-"
        print(f"{PREFIJO}   {signo} {nombre:<24} {v['media']:>+8.3f} ± {v['std']:.3f}")

    prio = r["priorizacion"]
    print(f"{PREFIJO} ==== Top-10 priorización 2026 ({len(prio)} municipios en total) ====")
    print(f"{PREFIJO} {'rango':>5} {'municipio':<18}{'puntaje':>9}{'prob_topK':>11}{'confianza':>11}"
          f"{'amenaza':>9}{'exposic.':>9}{'vulnerab.':>10}")
    for _, f in prio.head(10).iterrows():
        print(f"{PREFIJO} {int(f['rango']):>5} {str(f['nom_mpio']).title():<18}{f['puntaje']:>9.3f}"
              f"{f['prob_top_k']:>11.2f}{f['confianza']:>11}{f['contrib_amenaza']:>9.2f}"
              f"{f['contrib_exposicion']:>9.2f}{f['contrib_vulnerabilidad']:>10.2f}")

    print(f"{PREFIJO} tiempo total de ejecución: {r['tiempo_s']:.1f} s")


if __name__ == "__main__":
    main()
