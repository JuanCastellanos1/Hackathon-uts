"""Construye la tabla única municipio × año (T1.3, CONTRATOS §4).

Ejecutar desde la raíz del proyecto (con el entorno virtual activado): python src/03_tabla.py
Llama a sismo.features.construir_dataset(), que guarda los artefactos en data/ y docs/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sismo import config, features  # noqa: E402

PREFIJO = "[tabla]"


def main():
    df = features.construir_dataset()
    print(f"{PREFIJO} shape: {df.shape[0]} filas x {df.shape[1]} columnas "
          f"({df['cod_mpio'].nunique()} municipios, años {df['anio'].min()}–{df['anio'].max()})")

    dic = features.diccionario_datos()
    print(f"{PREFIJO} variables por rol:")
    for rol, n in dic["rol"].value_counts().reindex(
            ["llave", "sup", "territorial", "exposicion", "vulnerabilidad", "objetivo", "auxiliar_futuro"]).items():
        print(f"{PREFIJO}   {rol:<16} {int(n)}")

    print(f"{PREFIJO} distribución de {config.OBJETIVO} por split:")
    for nombre, (ini, fin) in [("TRAIN", config.TRAIN), ("VAL", config.VAL), ("TEST", config.TEST)]:
        serie = df.loc[df["anio"].between(ini, fin), config.OBJETIVO]
        conteo = serie.value_counts().reindex(config.CLASES, fill_value=0)
        pct = 100.0 * conteo / conteo.sum()
        detalle = ", ".join(f"{c}={int(conteo[c])} ({pct[c]:.1f} %)" for c in config.CLASES)
        print(f"{PREFIJO}   {nombre} {ini}–{fin} (n={int(conteo.sum())}): {detalle}")

    top = (df.groupby(["cod_mpio", "nom_mpio"])["n_m25"].sum()
           .sort_values(ascending=False).head(5))
    print(f"{PREFIJO} top-5 municipios por n_m25 total ({config.ANIO_INICIO}–{config.ANIO_FIN}):")
    for (cod, nombre), total in top.items():
        print(f"{PREFIJO}   {cod} {nombre}: {int(total)}")


if __name__ == "__main__":
    main()
