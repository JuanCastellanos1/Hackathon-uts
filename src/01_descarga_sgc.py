"""Paso 1 — Descarga del Catálogo Sísmico de la Red Sismológica Nacional (SGC).

Fuente: http://bdrsnc.sgc.gov.co/paginas1/catalogo/
  * Consulta 1 (Earthworm/SEISAN): 1 jun 1993 – 28 feb 2018  -> Consulta_Experta/
  * Consulta 2 (SeisComP3):        1 mar 2018 – a la fecha   -> Consulta_Experta_Seiscomp/

Filtros aplicados (iguales en ambas consultas):
  * Cuadrante: latitud 5.7°–8.2° N, longitud -74.6° – -72.4° (Santander + borde de
    departamentos vecinos, para poder comparar territorios).
  * Magnitud 0–9, profundidad 0–700 km, RMS 0–10 s, GAP 0–360°, errores 0–999 km
    (es decir, sin filtrar por calidad: la limpieza se hace en el paso 2).

Los archivos crudos se guardan sin modificar en data/raw/ para trazabilidad.
Uso:  python src/01_descarga_sgc.py
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

BASE = "https://bdrsnc.sgc.gov.co/paginas1/catalogo"
BOX = dict(latitudStart=5.7, latitudEnd=8.2, longitudStart=-74.6, longitudEnd=-72.4)
HOY = date.today()


# ---------------------------------------------------------------- consulta 1993-2018
def _leer_xlsx(contenido: bytes) -> pd.DataFrame:
    """El xlsx del SGC tiene un styles.xml inválido; se lee la hoja directamente."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    z = zipfile.ZipFile(io.BytesIO(contenido))
    sst = [
        "".join(t.text or "" for t in si.iter(f"{{{ns['m']}}}t"))
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)
    ]
    filas = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{ns['m']}}}row"):
        fila = {}
        for c in row.findall("m:c", ns):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            v = c.find("m:v", ns)
            if v is None:
                isv = c.find("m:is/m:t", ns)
                val = isv.text if isv is not None else None
            else:
                val = sst[int(v.text)] if c.get("t") == "s" else v.text
            fila[col] = val.strip() if isinstance(val, str) else val
        filas.append(fila)
    df = pd.DataFrame(filas)
    df.columns = df.iloc[0]
    return df.iloc[1:].reset_index(drop=True)


def descargar_antiguo(anio: int) -> pd.DataFrame:
    ini = date(1993, 6, 1) if anio == 1993 else date(anio, 1, 1)
    fin = date(2018, 2, 28) if anio == 2018 else date(anio, 12, 31)
    params = dict(
        **BOX, lat="", magnitudStart=0, magnitudEnd=9, magnitudmwStart=0, magnitudmwEnd=9,
        depthStart=0, depthEnd=700, rmsStart=0, rmsEnd=10,
        inicial=ini.strftime("%d/%m/%Y"), final=fin.strftime("%d/%m/%Y"),
        contipo="cuadrante", longcentral="", radio="", registro1=35, departamento="",
        municipio="", gapinicio=0, gapfinal=360, eprofmin=0, elongmin=0, elatmin=0,
        eprofmax=999, elongmax=999, elatmax=999, tregistros=1_000_000,
    )
    archivo = RAW / f"sgc_1993_2018_{anio}.xlsx"
    if not archivo.exists():
        r = requests.get(f"{BASE}/Consulta_Experta/descargar_exel_experta.php",
                         params=params, timeout=600)
        r.raise_for_status()
        archivo.write_bytes(r.content)
    df = _leer_xlsx(archivo.read_bytes())
    print(f"  1993-2018 | {anio}: {len(df):6d} eventos", flush=True)
    return df


# ---------------------------------------------------------------- consulta 2018-hoy
COLS_NUEVO = ["fecha_hora_utc", "lat", "lon", "prof_km", "mag", "tipo_mag", "fases",
              "rms_s", "gap_grados", "err_lat_km", "err_lon_km", "err_prof_km", "region",
              "quakeml", "fases_link", "mapa", "estado"]


def descargar_nuevo(anio: int, mes: int) -> pd.DataFrame:
    ini = date(anio, mes, 1)
    fin = (date(anio + (mes == 12), mes % 12 + 1, 1) - pd.Timedelta(days=1))
    fin = min(fin.date() if hasattr(fin, "date") else fin, HOY)
    archivo = RAW / f"sgc_2018_hoy_{anio}_{mes:02d}.html"
    if not archivo.exists():
        data = dict(
            **BOX, departamento="", municipio="", ubi="cuadrante",
            inicial=ini.strftime("%d/%m/%Y"), final=fin.strftime("%d/%m/%Y"),
            magnitudStart=0, magnitudEnd=9, depthStart=0, depthEnd=700, rmsStart=0,
            rmsEnd=10, gapStart=0, gapEnd=360, eprofmin=0, eprofmax=999, elongmin=0,
            elongmax=999, elatmin=0, elatmax=999, Submit="Consultar",
        )
        r = requests.post(f"{BASE}/Consulta_Experta_Seiscomp/consulta_sismo.php",
                          data=data, timeout=600)
        r.raise_for_status()
        archivo.write_bytes(r.content)
    h = archivo.read_text(encoding="utf-8", errors="replace")
    total = int(re.search(r"Total de registros:.*?<td colspan=2>(\d+)", h, re.S).group(1))
    cuerpo = h[h.find("<tbody"):]
    filas = []
    for tr in re.findall(r"<tr>(.*?)</tr>", cuerpo, re.S):
        celdas = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                  for c in re.split(r"<td[^>]*>", tr)[1:]]
        if celdas and re.match(r"\d{4}-\d\d-\d\d", celdas[0]):
            filas.append(celdas[: len(COLS_NUEVO)])
    df = pd.DataFrame(filas, columns=COLS_NUEVO)
    assert len(df) == total, f"{anio}-{mes}: {len(df)} filas vs total {total}"
    print(f"  2018-hoy  | {anio}-{mes:02d}: {len(df):6d} eventos", flush=True)
    return df


def main() -> None:
    print("Descargando consulta 1 (1993-2018)…")
    with ThreadPoolExecutor(4) as ex:
        antiguos = list(ex.map(descargar_antiguo, range(1993, 2019)))
    viejo = pd.concat(antiguos, ignore_index=True)
    viejo.to_csv(RAW / "sgc_1993_2018.csv", index=False)

    print("Descargando consulta 2 (2018-hoy)…")
    meses = [(a, m) for a in range(2018, HOY.year + 1) for m in range(1, 13)
             if date(2018, 3, 1) <= date(a, m, 1) <= HOY]
    with ThreadPoolExecutor(6) as ex:
        nuevos = list(ex.map(lambda am: descargar_nuevo(*am), meses))
    nuevo = pd.concat(nuevos, ignore_index=True)
    nuevo.to_csv(RAW / "sgc_2018_hoy.csv", index=False)

    meta = {
        "fuente": "Servicio Geológico Colombiano - Red Sismológica Nacional, Catálogo de Sismicidad",
        "url": "http://bdrsnc.sgc.gov.co/paginas1/catalogo/",
        "consultas": {
            "1993-2018": f"{BASE}/Consulta_Experta/consultaexperta_2.php",
            "2018-hoy": f"{BASE}/Consulta_Experta_Seiscomp/consultaexperta.php",
        },
        "fecha_descarga_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filtros": {**BOX, "magnitud": "0-9", "profundidad_km": "0-700", "rms_s": "0-10",
                    "gap_grados": "0-360", "errores_km": "0-999",
                    "periodo": f"1993-06-01 a {HOY.isoformat()}"},
        "eventos_1993_2018": len(viejo),
        "eventos_2018_hoy": len(nuevo),
    }
    (RAW / "metadata_descarga_sgc.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
