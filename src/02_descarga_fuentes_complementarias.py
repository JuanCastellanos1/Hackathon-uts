"""Paso 2 — Descarga de fuentes oficiales complementarias (integración).

  * DANE · DIVIPOLA (códigos y coordenadas de cabeceras municipales)
      https://www.datos.gov.co/resource/gdxc-w37w
  * DANE · Proyecciones de población municipal 2020-2035 (post-COVID, base CNPV 2018)
      https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/
  * UNGRD · Consolidado de emergencias por municipio (datos.gov.co)
      https://www.datos.gov.co/resource/wwkg-r6te   (2019-2024)
      https://www.datos.gov.co/resource/2343-nuqp   (2025 en adelante)

Uso:  python src/02_descarga_fuentes_complementarias.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOCRATA = "https://www.datos.gov.co/resource/{}.csv?$limit=1000000"
FUENTES = {
    "dane_divipola.csv": SOCRATA.format("gdxc-w37w"),
    "ungrd_emergencias_2019_2024.csv": SOCRATA.format("wwkg-r6te"),
    "ungrd_emergencias_2025.csv": SOCRATA.format("2343-nuqp"),
    "dane_proyeccion_poblacion_municipal_2020_2035.xlsx": (
        "https://www.dane.gov.co/files/censo2018/proyecciones-de-poblacion/Municipal/"
        "DCD-area-proypoblacion-Mun-2020-2035-ActPostCOVID-19.xlsx"
    ),
}


def main() -> None:
    meta = {}
    for nombre, url in FUENTES.items():
        destino = RAW / nombre
        if not destino.exists():
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            destino.write_bytes(r.content)
        meta[nombre] = {
            "url": url,
            "bytes": destino.stat().st_size,
            "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if nombre.endswith(".csv"):
            meta[nombre]["filas"] = len(pd.read_csv(destino, low_memory=False))
        print(nombre, meta[nombre])
    (RAW / "metadata_fuentes_complementarias.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
