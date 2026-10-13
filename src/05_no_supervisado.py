"""Componente no supervisado (T2.2, CONTRATOS §5).

Ejecutar desde la raíz del proyecto: .venv/bin/python src/05_no_supervisado.py
Llama a sismo.no_supervisado.ejecutar(), que guarda data/processed/clusters_municipios.csv,
reports/metricas_no_supervisado.json y 4 figuras en reports/figures/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from sismo import config, no_supervisado  # noqa: E402

PREFIJO = "[no_supervisado]"
COD_LOS_SANTOS = "68418"


def main():
    m = no_supervisado.ejecutar()

    print(f"{PREFIJO} selección de k (2..8):")
    print(f"{PREFIJO}   {'k':>2} {'silueta':>8} {'DB':>8} {'inercia':>12} {'gmm_bic':>14}")
    for fila in m["seleccion_k"]:
        print(f"{PREFIJO}   {fila['k']:>2} {fila['silueta']:>8.3f} {fila['davies_bouldin']:>8.3f} "
              f"{fila['inercia']:>12.1f} {fila['gmm_bic']:>14.1f}")
    print(f"{PREFIJO} k elegido = {m['k_elegido']} | silueta final = {m['silueta_final']:.3f}")

    clusters = pd.read_csv(config.CLUSTERS_CSV, dtype={"cod_mpio": str})
    conteo = clusters["cluster_nombre"].value_counts()
    print(f"{PREFIJO} centroides (valores reales, medias del cluster):")
    for nombre, valores in m["centroides"].items():
        n = int(conteo.get(nombre, 0))
        resumen = ", ".join(f"{f}={v:.3g}" for f, v in valores.items())
        print(f"{PREFIJO}   {nombre} (n={n}): {resumen}")

    print(f"{PREFIJO} comparación territorial (dpto x cluster):")
    comp = pd.DataFrame(m["comparacion_territorial"]).T.fillna(0).astype(int)
    print(comp.to_string())
    sant = comp.loc["SANTANDER"] if "SANTANDER" in comp.index else None
    vec = comp.loc[no_supervisado.ETIQUETA_VECINOS]
    if sant is not None:
        print(f"{PREFIJO} Santander ({int(sant.sum())} mpios): "
              + ", ".join(f"{c}={int(v)} ({100*v/sant.sum():.0f} %)" for c, v in sant.items()))
    print(f"{PREFIJO} Vecinos ({int(vec.sum())} mpios): "
          + ", ".join(f"{c}={int(v)} ({100*v/vec.sum():.0f} %)" for c, v in vec.items()))

    print(f"{PREFIJO} top-5 importancias (árbol sustituto, profundidad 4):")
    for feat, imp in sorted(m["importancia_variables"].items(), key=lambda t: -t[1])[:5]:
        print(f"{PREFIJO}   {feat:<22} {imp:.3f}")

    print(f"{PREFIJO} top-3 fuentes DBSCAN (M >= 3,0, no atípicos):")
    for f in m["dbscan_fuentes"][:3]:
        print(f"{PREFIJO}   fuente {f['fuente']}: n={f['n']} lat={f['lat']:.3f} lon={f['lon']:.3f} "
              f"prof_media={f['prof_media_km']:.1f} km mag_max={f['mag_max']:.1f}")

    fila_ls = clusters[clusters["cod_mpio"] == COD_LOS_SANTOS]
    if len(fila_ls):
        ls = fila_ls.iloc[0]
        marca = "OK" if ls["cluster_nombre"] == "Nido / sismicidad intermedia" else "AVISO"
        print(f"{PREFIJO} Los Santos ({COD_LOS_SANTOS}) cae en: '{ls['cluster_nombre']}' "
              f"(pct_intermedio={ls['pct_intermedio']:.2f}) [{marca}]")
    else:
        print(f"{PREFIJO} AVISO: Los Santos ({COD_LOS_SANTOS}) no aparece en clusters_municipios.csv")


if __name__ == "__main__":
    main()
