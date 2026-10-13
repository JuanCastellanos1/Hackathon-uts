"""Módulo territorial: municipios (DIVIPOLA), población (DANE) y emergencias (UNGRD).

Contrato: docs/CONTRATOS_DATOS.md §3 y docs/uml/03_clases.md (módulo `territorio`).
Este módulo NO guarda archivos a disco; eso lo hace T1.3.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from sismo import config

RADIO_TIERRA_KM = 6371.0

# Nombres de columnas de UNGRD. Los dos archivos usan nombres distintos para lo mismo.
_COLUMNAS_UNGRD = {
    "divipola": ["divipola", "codificaci_n_segun_divipola"],
    "viv_destruidas": ["viviendas_destruidas", "viv_destru"],
    "viv_averiadas": ["viviendas_averiadas", "viv_aver"],
    "personas": ["personas"],
    "evento": ["evento"],
    "fecha": ["fecha"],
}


def distancia_km(lat1, lon1, lat2, lon2):
    """Distancia haversine en km (vectorizada). Devuelve un ndarray."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return np.asarray(2.0 * RADIO_TIERRA_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _norm_texto(valor) -> str:
    """Normaliza un nombre para cruzar: sin tildes, mayúsculas y espacios simples."""
    txt = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return " ".join(txt.upper().split())


def _a_codigo_mpio(serie):
    """Convierte códigos numéricos o texto a str de 5 dígitos. Los NaN quedan como None."""
    num = pd.to_numeric(serie, errors="coerce")
    return num.astype("Int64").astype(str).str.zfill(5).where(num.notna(), None)


def cargar_divipola(path: Path) -> pd.DataFrame:
    """Lee DIVIPOLA. Devuelve todas las cabeceras con cod_mpio y cod_dpto como str."""
    raw = pd.read_csv(path, dtype=str)
    df = pd.DataFrame(
        {
            "cod_mpio": _a_codigo_mpio(raw["cod_mpio"]),
            "cod_dpto": _a_codigo_mpio(raw["cod_dpto"]).str[-2:],
            "dpto": raw["dpto"].str.strip(),
            "nom_mpio": raw["nom_mpio"].str.strip(),
            "lat": raw["latitud"].str.replace(",", ".", regex=False).astype(float),
            "lon": raw["longitud"].str.replace(",", ".", regex=False).astype(float),
        }
    )
    print(f"[territorio] DIVIPOLA: {len(df)} cabeceras leídas")
    return df


def municipios_en_caja(divipola: pd.DataFrame) -> pd.DataFrame:
    """Filtra las cabeceras cuyas coordenadas caen dentro de config.BOX (esperado: 176)."""
    b = config.BOX
    mascara = (
        divipola["lat"].between(b["lat_min"], b["lat_max"])
        & divipola["lon"].between(b["lon_min"], b["lon_max"])
    )
    df = divipola.loc[mascara].reset_index(drop=True)
    print(
        f"[territorio] municipios en caja: {len(df)} "
        f"(Santander, cod_dpto '{config.COD_DPTO_SANTANDER}': "
        f"{(df['cod_dpto'] == config.COD_DPTO_SANTANDER).sum()})"
    )
    return df


def cargar_poblacion(path: Path) -> pd.DataFrame:
    """Población total y porcentaje rural 2020 (DANE). Devuelve cod_mpio, poblacion_total, pct_rural."""
    raw = pd.read_excel(path, header=8)
    raw = raw[raw["MPIO"].notna() & (raw["AÑO"] == 2020)]  # quita pie de página y otros años
    area = raw["ÁREA GEOGRÁFICA"].str.strip()
    raw = raw.assign(area=area)
    cod = raw["MPIO"].astype(int).astype(str).str.zfill(5)
    raw = raw.assign(cod_mpio=cod)

    total = raw[raw["area"] == "Total"].set_index("cod_mpio")["Población"]
    cabecera = raw[raw["area"] == "Cabecera Municipal"].set_index("cod_mpio")["Población"]
    if total.index.duplicated().any():
        raise ValueError("[territorio] población: hay códigos duplicados en el área 'Total' de 2020")

    df = pd.DataFrame({"poblacion_total": total})
    df["pct_rural"] = 1.0 - cabecera.reindex(df.index) / df["poblacion_total"]
    df["poblacion_total"] = df["poblacion_total"].round().astype("Int64")
    df = df.reset_index().rename(columns={"index": "cod_mpio"})
    return df[["cod_mpio", "poblacion_total", "pct_rural"]]


def _leer_ungrd(path: Path) -> pd.DataFrame:
    """Lee un archivo UNGRD con nombres estándar: cod_raw, clave_nombre, anio, evento, personas, viviendas."""
    raw = pd.read_csv(path, low_memory=False)
    cols = {}
    for estandar, posibles in _COLUMNAS_UNGRD.items():
        encontrada = next((c for c in posibles if c in raw.columns), None)
        if encontrada is None:
            raise KeyError(f"[territorio] {path.name}: falta la columna para '{estandar}' (buscadas {posibles})")
        cols[estandar] = raw[encontrada]

    clave_nombre = raw["departamento"].map(_norm_texto) + "|" + raw["municipio"].map(_norm_texto)
    df = pd.DataFrame(
        {
            "cod_raw": _a_codigo_mpio(cols["divipola"]),
            "clave_nombre": clave_nombre,
            "anio": pd.to_datetime(cols["fecha"], errors="coerce").dt.year,
            "evento": cols["evento"].astype(str),
            "personas": pd.to_numeric(cols["personas"], errors="coerce").fillna(0),
            "viviendas": (
                pd.to_numeric(cols["viv_destruidas"], errors="coerce").fillna(0)
                + pd.to_numeric(cols["viv_averiadas"], errors="coerce").fillna(0)
            ),
        }
    )
    print(f"[territorio] UNGRD {path.name}: {len(df)} registros")
    return df


def _resolver_municipio(eventos: pd.DataFrame) -> pd.DataFrame:
    """Asigna cod_mpio a cada registro UNGRD: primero por nombre (dpto|municipio), luego por código.

    Motivo: en el archivo 2025 la columna `codificaci_n_segun_divipola` trae secuencias que no son
    DIVIPOLA (p. ej. 20046 = CHIRIGUANÁ, cuyo código real es 20178) y códigos válidos asignados a otro
    municipio. El nombre sí es confiable y único por departamento en DIVIPOLA. El código solo se usa
    cuando el nombre no cruza. Los registros que no cruzan por ninguno de los dos quedan con None.
    """
    divipola = cargar_divipola(config.RAW_DIVIPOLA)
    clave_div = (divipola["dpto"].map(_norm_texto) + "|" + divipola["nom_mpio"].map(_norm_texto))
    por_nombre_map = pd.Series(divipola["cod_mpio"].to_numpy(), index=clave_div.to_numpy())
    por_nombre_map = por_nombre_map[~por_nombre_map.index.duplicated(keep=False)]  # descarta claves ambiguas

    codigos_validos = set(divipola["cod_mpio"])
    por_nombre = eventos["clave_nombre"].map(por_nombre_map)
    por_codigo = eventos["cod_raw"].where(eventos["cod_raw"].isin(codigos_validos))

    salida = eventos.assign(cod_mpio=por_nombre.fillna(por_codigo))
    n_tot = len(salida)
    n_nombre = int(por_nombre.notna().sum())
    n_codigo = int((por_nombre.isna() & por_codigo.notna()).sum())
    n_ok = int(salida["cod_mpio"].notna().sum())
    print(
        f"[territorio] UNGRD resolución: {n_ok}/{n_tot} registros ({100 * n_ok / n_tot:.1f} %) "
        f"a municipio DIVIPOLA; por nombre={n_nombre}, por código de respaldo={n_codigo}, "
        f"sin resolver={n_tot - n_ok}"
    )
    return salida


def cargar_emergencias(paths) -> pd.DataFrame:
    """Agrega los registros UNGRD por municipio.

    Columnas: cod_mpio, n_emergencias_total, emergencias_prom_anual (n / años distintos en los datos),
    mov_masa_total, viviendas_afectadas_total, personas_afectadas_total.
    """
    frames = [_leer_ungrd(Path(p)) for p in paths]
    eventos = pd.concat(frames, ignore_index=True)
    eventos = _resolver_municipio(eventos)

    sin_codigo = eventos["cod_mpio"].isna()
    eventos = eventos.loc[~sin_codigo]

    n_anios = int(eventos["anio"].dropna().nunique())
    print(f"[territorio] UNGRD: años distintos presentes en los datos = {n_anios}")

    eventos = eventos.assign(
        es_mov_masa=eventos["evento"].str.upper().str.contains("MOVIMIENTO EN MASA", regex=False),
    )
    agregado = eventos.groupby("cod_mpio").agg(
        n_emergencias_total=("evento", "size"),
        mov_masa_total=("es_mov_masa", "sum"),
        viviendas_afectadas_total=("viviendas", "sum"),
        personas_afectadas_total=("personas", "sum"),
    )
    agregado["emergencias_prom_anual"] = agregado["n_emergencias_total"] / n_anios
    for col in ["n_emergencias_total", "mov_masa_total", "viviendas_afectadas_total", "personas_afectadas_total"]:
        agregado[col] = agregado[col].round().astype("int64")
    return agregado.reset_index()[
        [
            "cod_mpio",
            "n_emergencias_total",
            "emergencias_prom_anual",
            "mov_masa_total",
            "viviendas_afectadas_total",
            "personas_afectadas_total",
        ]
    ]


def asignar_municipio(eventos: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Asigna a cada evento la cabecera más cercana (haversine) y la distancia en km.

    Devuelve una copia de `eventos` con las columnas `cod_mpio` y `dist_cabecera_km`.
    Los eventos sin coordenadas válidas quedan con cod_mpio NaN y distancia NaN.
    """
    salida = eventos.copy()
    cabeceras = np.radians(municipios[["lat", "lon"]].to_numpy(dtype=float))
    arbol = BallTree(cabeceras, metric="haversine")

    coords = salida[["lat", "lon"]].to_numpy(dtype=float)
    valido = np.isfinite(coords).all(axis=1)
    cod = np.full(len(salida), None, dtype=object)
    dist = np.full(len(salida), np.nan)
    if valido.any():
        d_rad, idx = arbol.query(np.radians(coords[valido]), k=1)
        cod[valido] = municipios["cod_mpio"].to_numpy()[idx[:, 0]]
        dist[valido] = d_rad[:, 0] * RADIO_TIERRA_KM

    salida["cod_mpio"] = cod
    salida["dist_cabecera_km"] = dist
    return salida


def construir_municipios() -> pd.DataFrame:
    """Construye la tabla de municipios de CONTRATOS §3 (176 filas, ordenada por cod_mpio). No guarda a disco."""
    divipola = cargar_divipola(config.RAW_DIVIPOLA)
    municipios = municipios_en_caja(divipola)

    poblacion = cargar_poblacion(config.RAW_POBLACION)
    emergencias = cargar_emergencias(config.RAW_UNGRD)

    df = municipios.copy()
    df["es_santander"] = (df["cod_dpto"] == config.COD_DPTO_SANTANDER).astype(int)
    df["dist_nido_km"] = distancia_km(df["lat"], df["lon"], config.NIDO_LAT, config.NIDO_LON)

    df = df.merge(poblacion, on="cod_mpio", how="left")
    df = df.merge(emergencias, on="cod_mpio", how="left")
    for col in ["n_emergencias_total", "mov_masa_total", "viviendas_afectadas_total", "personas_afectadas_total"]:
        df[col] = df[col].fillna(0).astype("int64")
    df["emergencias_prom_anual"] = df["emergencias_prom_anual"].fillna(0.0).astype(float)

    df = df.sort_values("cod_mpio").reset_index(drop=True)
    columnas = [
        "cod_mpio", "nom_mpio", "cod_dpto", "dpto", "lat", "lon", "es_santander", "dist_nido_km",
        "poblacion_total", "pct_rural", "n_emergencias_total", "emergencias_prom_anual",
        "mov_masa_total", "viviendas_afectadas_total", "personas_afectadas_total",
    ]
    df = df[columnas]

    # Reportes de cruce
    n = len(df)
    con_pob = int(df["poblacion_total"].notna().sum())
    print(f"[territorio] cruce DANE: {con_pob}/{n} municipios con población 2020 ({100 * con_pob / n:.1f} %)")
    if con_pob < n:
        faltan = df.loc[df["poblacion_total"].isna(), "cod_mpio"].tolist()
        print(f"[territorio] DANE: sin población -> {faltan}")

    con_emerg = int((df["n_emergencias_total"] > 0).sum())
    print(f"[territorio] cruce UNGRD: {con_emerg}/{n} municipios de la caja con al menos una emergencia")
    return df


if __name__ == "__main__":
    print(construir_municipios().head())
