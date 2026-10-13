"""Demo en vivo: priorización sísmica de Santander (Hackathon CIAI UTSmart 2026 · Reto Avanzado).

La app SOLO lee artefactos generados por los scripts de src/. No entrena modelos.
Si falta un artefacto, la pestaña correspondiente muestra el comando para generarlo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sismo import config  # noqa: E402

st.set_page_config(layout="wide", page_title="Priorización sísmica Santander", page_icon="🌎")

# Estilo visual Windows Vista / Aero Glass (solo apariencia; ver app/estilo_vista.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estilo_vista  # noqa: E402

TITULO_VENTANA = "Priorización sísmica de Santander — Reto Avanzado"
estilo_vista.plantilla_plotly()
estilo_vista.aplicar_estilo(TITULO_VENTANA)
st.markdown(estilo_vista.barra_titulo(TITULO_VENTANA), unsafe_allow_html=True)

# ----------------------------------------------------------------- rutas y comandos
RUTA_META_SGC = config.RAW / "metadata_descarga_sgc.json"
RUTA_META_COMP = config.RAW / "metadata_fuentes_complementarias.json"
RUTA_MET_SUP = config.REPORTS / "metricas_supervisado.json"
RUTA_MET_NOSUP = config.REPORTS / "metricas_no_supervisado.json"
RUTA_MET_RL = config.REPORTS / "metricas_rl.json"
RUTA_CURVA_RL = config.REPORTS / "curva_aprendizaje_rl.csv"

CMD_TABLA = "python src/03_tabla.py"
CMD_SUP = "python src/04_supervisado.py"
CMD_NOSUP = "python src/05_no_supervisado.py"
CMD_RL = "python src/06_agente_rl.py"
CMD_DESCARGA = "python src/01_descarga_sgc.py  y  python src/02_descarga_fuentes_complementarias.py"

# ----------------------------------------------------------------- estilo y etiquetas
PALETA = ["#0F766E", "#D97706", "#7C3AED", "#2563EB", "#DB2777", "#65A30D", "#0891B2"]
COLOR_CONF = {"alta": "#0F766E", "media": "#D97706", "baja": "#9CA3AF"}
COLOR_CRITERIO = {"Amenaza": "#EA580C", "Exposición": "#2563EB", "Vulnerabilidad": "#7C3AED"}
COLOR_NIDO = "#B91C1C"
CENTRO_SANTANDER = {"lat": 7.0, "lon": -73.3}

ETIQUETAS = {
    "tasa_m25_anual": "Tasa M≥2,5 por año",
    "tasa_m30_anual": "Tasa M≥3,0 por año",
    "tasa_m35_anual": "Tasa M≥3,5 por año",
    "mag_max": "Magnitud máxima",
    "mag_media": "Magnitud media",
    "b_value": "Valor b",
    "prof_media_km": "Profundidad media (km)",
    "pct_superficial": "% superficial (<30 km)",
    "pct_intermedio": "% intermedio (70–300 km)",
    "log_energia_anual": "Energía anual (log10 J)",
    "fraccion_anios_activos": "Fracción de años activos",
    "p_alto": "Probabilidad de actividad alta",
    "p_medio": "Probabilidad de actividad media",
    "n_m25_prom_3a_log": "Sismicidad reciente (log)",
    "mag_max_historica": "Magnitud máxima histórica",
    "dist_nido_km": "Distancia al Nido (km)",
    "log_poblacion": "Población (log)",
    "pct_rural": "% rural",
    "emergencias_log": "Emergencias UNGRD (log)",
    "mov_masa_log": "Movimientos en masa (log)",
    "viviendas_log": "Viviendas afectadas (log)",
    "n_m25_prom_3a": "Sismos M≥2,5 (prom. 3 años)",
    "n_m25": "Sismos M≥2,5",
    "lat": "Latitud",
    "lon": "Longitud",
    "periodo_red": "Periodo de la red (2018+)",
}


def etiqueta(clave: str) -> str:
    """Nombre legible para jurado no experto."""
    if clave in ETIQUETAS:
        return ETIQUETAS[clave]
    if clave.startswith("cluster_rl_"):
        return f"Grupo {clave.split('_')[-1]} (no supervisado)"
    return clave


# ----------------------------------------------------------------- utilidades de formato
def num(x: float, d: int = 2) -> str:
    """Número con formato español: 1.234,5"""
    try:
        return f"{float(x):,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (TypeError, ValueError):
        return "—"


def signo(x: float, d: int = 2) -> str:
    """Diferencia con signo explícito: +0,03 / -0,10"""
    return f"{'+' if x >= 0 else '-'}{num(abs(x), d)}"


def pct(x: float, d: int = 0) -> str:
    return f"{num(float(x) * 100, d)} %"


def entero(x) -> str:
    try:
        return f"{int(x):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def fecha_utc(s: str) -> str:
    return str(s)[:16].replace("T", " ") + " UTC"


def km_desde_nido(lat: pd.Series, lon: pd.Series) -> pd.Series:
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2, lo2 = np.radians(config.NIDO_LAT), np.radians(config.NIDO_LON)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


# ----------------------------------------------------------------- carga de artefactos (solo lectura)
def falta(ruta: Path, comando: str, nivel: str = "error") -> None:
    msg = f"Falta el artefacto `{ruta.relative_to(config.ROOT)}`. Genéralo con: `{comando}`"
    (st.warning if nivel == "warning" else st.error)(msg)


@st.cache_data(show_spinner=False)
def _leer_csv(ruta: str, mtime: float, con_cod: bool) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"cod_mpio": str} if con_cod else None)


@st.cache_data(show_spinner=False)
def _leer_json(ruta: str, mtime: float) -> dict:
    with open(ruta, encoding="utf-8") as fh:
        return json.load(fh)


def obtener_csv(ruta: Path, comando: str, con_cod: bool = True, nivel: str = "error") -> pd.DataFrame | None:
    if not ruta.exists():
        falta(ruta, comando, nivel)
        return None
    return _leer_csv(str(ruta), ruta.stat().st_mtime, con_cod)


def obtener_json(ruta: Path, comando: str, nivel: str = "error") -> dict | None:
    if not ruta.exists():
        falta(ruta, comando, nivel)
        return None
    return _leer_json(str(ruta), ruta.stat().st_mtime)


# ----------------------------------------------------------------- barra lateral
with st.sidebar:
    st.title("Priorización sísmica Santander")
    st.caption("Hackathon CIAI UTSmart 2026 · Reto Avanzado")
    st.markdown(
        "- Estimamos la **amenaza sísmica** de cada municipio con el catálogo del SGC (2002–2025).\n"
        "- Combinamos amenaza, **exposición** (población) y **vulnerabilidad** (emergencias) en un puntaje de riesgo.\n"
        "- Un **agente de refuerzo** aprende a elegir K municipios prioritarios. Es una herramienta exploratoria."
    )
    st.markdown("**Fuentes oficiales**")
    st.markdown(
        "- SGC · Catálogo de sismicidad (RSNC)\n"
        "- DANE · DIVIPOLA\n"
        "- DANE · Proyección de población 2020–2035\n"
        "- UNGRD · Emergencias 2019–2025"
    )
    st.toggle(
        "Mapa compatible (sin internet)", key="mapa_compatible",
        help="Dibuja los mapas sin WebGL ni teselas de internet. Úsalo si el mapa sale en blanco "
             "(por ejemplo en Brave con Shields activos o sin conexión).",
    )


# ----------------------------------------------------------------- pestaña 1: problema y datos
NOMBRES_FUENTE = {
    "dane_divipola.csv": "DANE · DIVIPOLA (códigos y cabeceras)",
    "ungrd_emergencias_2019_2024.csv": "UNGRD · Emergencias 2019–2024",
    "ungrd_emergencias_2025.csv": "UNGRD · Emergencias 2025",
    "dane_proyeccion_poblacion_municipal_2020_2035.xlsx": "DANE · Proyección de población 2020–2035",
}


def tabla_fuentes() -> pd.DataFrame | None:
    filas = []
    meta_sgc = obtener_json(RUTA_META_SGC, CMD_DESCARGA, nivel="warning")
    if meta_sgc:
        f = meta_sgc.get("filtros", {})
        filtros = (
            f"Recuadro lat {f.get('latitudStart')} a {f.get('latitudEnd')}, "
            f"lon {f.get('longitudStart')} a {f.get('longitudEnd')}; "
            f"magnitud {f.get('magnitud')}; profundidad {f.get('profundidad_km')} km; "
            f"RMS {f.get('rms_s')} s; errores {f.get('errores_km')} km"
        ).replace(".", ",")
        descarga = fecha_utc(meta_sgc.get("fecha_descarga_utc", ""))
        for periodo, url in meta_sgc.get("consultas", {}).items():
            eventos = meta_sgc.get(f"eventos_{periodo.replace('-', '_')}")
            filas.append({
                "Fuente": f"SGC · Catálogo RSNC {periodo}",
                "Eventos": entero(eventos),
                "URL": url,
                "Descargada": descarga,
                "Filtros": filtros,
            })
    meta_comp = obtener_json(RUTA_META_COMP, CMD_DESCARGA, nivel="warning")
    if meta_comp:
        for archivo, info in meta_comp.items():
            filas.append({
                "Fuente": NOMBRES_FUENTE.get(archivo, archivo),
                "Eventos": entero(info.get("filas")) if info.get("filas") else "—",
                "URL": info.get("url", ""),
                "Descargada": fecha_utc(info.get("fecha_descarga_utc", "")),
                "Filtros": "Descarga completa del recurso (sin filtros de contenido)",
            })
    return pd.DataFrame(filas) if filas else None


def render_problema() -> None:
    st.subheader("El problema: un nido sísmico bajo Santander")
    col_txt, col_marco = st.columns([3, 2], gap="large")
    with col_txt:
        st.markdown(
            "Santander es una de las zonas de **mayor actividad sísmica de Colombia**. Buena parte de esa "
            "actividad se concentra en el **Nido sísmico de Bucaramanga**, bajo la **Mesa de Los Santos** "
            f"(≈ {num(config.NIDO_LAT, 1)}° N, {num(abs(config.NIDO_LON), 1)}° O). Allí ocurren sismos de "
            "**profundidad intermedia (70–300 km)** que se sienten en amplias zonas del departamento."
        )
        st.markdown(
            "La pregunta de este proyecto no es predecir cuándo ocurrirá un sismo. Es más práctica: "
            "**¿en qué municipios conviene priorizar la preparación con un presupuesto limitado?**"
        )
    with col_marco:
        st.markdown("#### Marco de riesgo")
        st.markdown("### Riesgo = Amenaza × Exposición × Vulnerabilidad")
        st.markdown(
            "- **Amenaza:** actividad sísmica del catálogo del SGC.\n"
            "- **Exposición:** personas que viven en el municipio (DANE).\n"
            "- **Vulnerabilidad:** emergencias históricas registradas por la UNGRD."
        )

    ds = obtener_csv(config.DATASET_CSV, CMD_TABLA)
    meta_sgc = obtener_json(RUTA_META_SGC, CMD_DESCARGA, nivel="warning")
    st.divider()
    st.markdown("#### Datos del proyecto")
    c1, c2, c3, c4 = st.columns(4)
    if meta_sgc:
        total_ev = int(meta_sgc.get("eventos_1993_2018", 0)) + int(meta_sgc.get("eventos_2018_hoy", 0))
        c1.metric("Eventos sísmicos SGC", entero(total_ev), help="Catálogo 1993–2018 más 2018–hoy, antes de limpiar")
    else:
        c1.metric("Eventos sísmicos SGC", "—")
    if ds is not None:
        n_mpio = ds["cod_mpio"].nunique()
        n_sant = ds.loc[ds["es_santander"] == 1, "cod_mpio"].nunique()
        c2.metric("Municipios analizados", n_mpio, delta=f"{n_sant} en Santander", delta_color="off")
        c3.metric("Filas municipio × año", entero(len(ds)), delta="2002–2025", delta_color="off")
        c4.metric("Variables del dataset", len(ds.columns), help="Incluye llaves, variables sísmicas, exposición, vulnerabilidad y objetivo")

        serie = ds[ds["es_santander"] == 1].groupby("anio", as_index=False)["n_m25"].sum()
        fig = px.line(
            serie, x="anio", y="n_m25", markers=True,
            labels={"anio": "Año", "n_m25": "Sismos M ≥ 2,5"},
            title="Sismos M ≥ 2,5 por año en Santander (suma de sus municipios)",
        )
        fig.update_traces(line_color=PALETA[0])
        fig.add_vline(x=2018, line_dash="dot", line_color="#6B7280",
                      annotation_text="Cambio de red SGC", annotation_position="top left")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Desde 2002 se usan solo sismos M ≥ 2,5 porque el catálogo es completo a partir de esa magnitud. "
            "El salto de 2018 refleja el cambio de sistema de la red, no un cambio real de sismicidad."
        )

    st.markdown("#### Fuentes, fechas de descarga y filtros")
    fuentes = tabla_fuentes()
    if fuentes is not None:
        st.dataframe(
            fuentes, hide_index=True, width="stretch",
            column_config={"URL": st.column_config.LinkColumn("URL", width="medium")},
        )

    st.markdown("#### Diccionario de datos")
    dic = obtener_csv(config.DICCIONARIO_CSV, CMD_TABLA, con_cod=False)
    if dic is not None:
        st.dataframe(dic, hide_index=True, width="stretch", height=420)


# ----------------------------------------------------------------- pestaña 2: supervisado
def render_supervisado() -> None:
    st.subheader("Modelo supervisado: ¿habrá mucha, poca o ninguna actividad el año siguiente?")
    met = obtener_json(RUTA_MET_SUP, CMD_SUP)
    if met is None:
        return
    t = met["test"]
    lb = met["lineas_base_test"]
    bs = met["bootstrap_f1_test"]
    f1 = t["f1_macro"]
    f1_pers = lb["persistencia"]["f1_macro"]
    f1_may = lb["mayoritaria"]["f1_macro"]
    mat = np.array(t["matriz_confusion"])
    tp_alto, total_alto = int(mat[2, 2]), int(mat[2].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "F1-macro del modelo (test)", num(f1),
        delta=f"{signo(f1 - f1_pers)} frente a persistencia",
        delta_color="normal",
        help="F1 promedio de las tres clases (bajo, medio, alto). Pesa igual las clases poco frecuentes.",
    )
    c2.metric(
        "Línea base: persistencia", num(f1_pers),
        delta=f"{signo(f1_pers - f1_may)} frente a mayoritaria",
        delta_color="off",
        help="Predice que el año siguiente tendrá la misma clase que el año actual.",
    )
    c3.metric(
        "Línea base: mayoritaria", num(f1_may),
        help="Siempre predice la clase más frecuente (bajo).",
    )
    c4.metric(
        "Intervalo de confianza 95 % (bootstrap)",
        f"[{num(bs['p2_5'])} ; {num(bs['p97_5'])}]",
        help="Rango plausible del F1-macro en test, estimado remuestreando los municipios-año.",
    )
    st.caption(
        f"Recall de la clase alta: el modelo detecta **{tp_alto} de {total_alto}** casos de actividad alta en test "
        f"({pct(t['recall_alto'])}). En alerta temprana importa más no dejar pasar un caso alto que evitar falsas alarmas."
    )

    col_m, col_i = st.columns(2, gap="large")
    with col_m:
        etiquetas = ["Bajo", "Medio", "Alto"]
        fig = px.imshow(
            mat, x=etiquetas, y=etiquetas, text_auto=True, aspect="auto",
            color_continuous_scale=["#F0FDFA", PALETA[0]],
            labels=dict(x="Clase predicha", y="Clase real", color="Casos"),
            title="Matriz de confusión (test 2020–2024)",
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with col_i:
        imp = pd.Series(met["importancias"]).sort_values(ascending=False).head(15)
        df_imp = imp.rename_axis("variable").reset_index(name="importancia")
        df_imp["variable"] = df_imp["variable"].map(etiqueta)
        fig = px.bar(df_imp, x="importancia", y="variable", orientation="h",
                     title="Importancia de las 15 variables principales",
                     labels={"importancia": "Importancia (modelo)", "variable": ""})
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_traces(marker_color=PALETA[0])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Cómo se evalúa, sin trampas")
    st.markdown(
        f"- **Validación temporal:** entrenamiento {config.TRAIN[0]}–{config.TRAIN[1]}, validación "
        f"{config.VAL[0]}–{config.VAL[1]} (para elegir el modelo) y prueba {config.TEST[0]}–{config.TEST[1]} "
        "(solo se usa una vez, al final). Nunca se mezclan años futuros con pasados.\n"
        "- **Objetivo:** el nivel de actividad del año siguiente. *Bajo* = 0 sismos M ≥ 2,5; *medio* = 1 a 4; *alto* = 5 o más.\n"
        "- **Sin circularidad:** todas las variables de entrada se calculan con información hasta el año t. "
        "El año siguiente solo se usa como respuesta correcta."
    )
    st.caption(
        f"Modelo elegido: {met.get('modelo_elegido', '—')} con parámetros {met.get('params', {})}. "
        "Exposición y vulnerabilidad no entran en este modelo: solo mide la amenaza."
    )


# ----------------------------------------------------------------- mapa compatible
def modo_compatible() -> bool:
    return bool(st.session_state.get("mapa_compatible", False))


def mapa_plano(capas: list[dict], alto: int) -> go.Figure:
    """Mapa de respaldo en SVG (longitud × latitud): no usa WebGL ni teselas, funciona en
    cualquier navegador y sin internet."""
    trazas = []
    for c in capas:
        c = dict(c)
        c["x"], c["y"] = c.pop("lon"), c.pop("lat")
        trazas.append(go.Scatter(**c))
    fig = go.Figure(trazas)
    fig.update_xaxes(title="Longitud", range=[-74.65, -72.35], showgrid=True, gridcolor="#E3ECF4")
    fig.update_yaxes(title="Latitud", range=[5.65, 8.25], showgrid=True, gridcolor="#E3ECF4",
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(height=alto, plot_bgcolor="#F4F8FB")
    return fig


# ----------------------------------------------------------------- pestaña 3: no supervisado
def figura_mapa_clusters(df: pd.DataFrame) -> go.Figure:
    nido = dict(lat=[config.NIDO_LAT], lon=[config.NIDO_LON])
    if modo_compatible():
        capas = [
            dict(lat=g["lat"].tolist(), lon=g["lon"].tolist(), mode="markers", name=nombre,
                 marker=dict(size=9, color=PALETA[i % len(PALETA)]),
                 hovertext=(g["nom_mpio"] + " (" + g["dpto"] + ")").tolist(), hoverinfo="text")
            for i, (nombre, g) in enumerate(df.groupby("cluster_nombre"))
        ]
        capas.append(dict(**nido, mode="markers", marker=dict(size=18, color=COLOR_NIDO, symbol="star"),
                          name="Nido sísmico de Bucaramanga", hovertext=["Nido sísmico de Bucaramanga"],
                          hoverinfo="text"))
        fig = mapa_plano(capas, 560)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.12))
        return fig
    try:
        fig = px.scatter_map(
            df, lat="lat", lon="lon", color="cluster_nombre", hover_name="nom_mpio",
            hover_data={"dpto": True, "lat": False, "lon": False, "cluster_nombre": False},
            color_discrete_sequence=PALETA, map_style="carto-positron",
            center=CENTRO_SANTANDER, zoom=6.5, height=560,
        )
        fig.add_trace(go.Scattermap(
            **nido, mode="markers", marker=dict(size=18, color=COLOR_NIDO),
            name="Nido sísmico de Bucaramanga", hovertext=["Nido sísmico de Bucaramanga"], hoverinfo="text",
        ))
    except Exception:  # noqa: BLE001 - respaldo si el mapa base no está disponible
        fig = px.scatter_geo(
            df, lat="lat", lon="lon", color="cluster_nombre", hover_name="nom_mpio",
            color_discrete_sequence=PALETA, height=560,
        )
        fig.update_geos(
            lataxis_range=[5.6, 8.4], lonaxis_range=[-74.4, -72.4], projection_type="mercator",
            showland=True, landcolor="#F1F5F4", showcountries=True, countrycolor="#CBD5E1",
        )
        fig.add_trace(go.Scattergeo(
            **nido, mode="markers", marker=dict(size=18, color=COLOR_NIDO),
            name="Nido sísmico de Bucaramanga", hovertext=["Nido sísmico de Bucaramanga"], hoverinfo="text",
        ))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02))
    return fig


def render_nosupervisado() -> None:
    st.subheader("Perfiles sísmicos: grupos de municipios con comportamiento parecido")
    st.caption(
        "Agrupamos los 176 municipios del recuadro según su historia sísmica (K-Means sobre 2002–2025). "
        "No usa la etiqueta de riesgo; solo describe cómo tiembla cada municipio."
    )
    clusters = obtener_csv(config.CLUSTERS_CSV, CMD_NOSUP)
    met = obtener_json(RUTA_MET_NOSUP, CMD_NOSUP)

    if met is not None:
        c1, c2 = st.columns(2)
        c1.metric("Número de grupos elegido (k)", met.get("k_elegido", "—"),
                  help="Se eligió k comparando silueta, Davies-Bouldin, inercia y BIC")
        c2.metric("Silueta final", num(met.get("silueta_final", float("nan")), 2),
                  help="Qué tan bien separados están los grupos: entre -1 y 1; más alto es mejor")

    if clusters is not None:
        st.plotly_chart(figura_mapa_clusters(clusters), width="stretch")

    col_c, col_t = st.columns([3, 2], gap="large")
    with col_c:
        st.markdown("#### Comparación territorial: Santander frente a sus vecinos")
        if met is not None:
            comp = met.get("comparacion_territorial", {})
            filas = []
            for clave, nombre in [("SANTANDER", "Santander"), ("VECINOS (otros dptos)", "Vecinos (otros dptos)")]:
                if clave in comp:
                    total = sum(comp[clave].values())
                    for grupo, n in comp[clave].items():
                        filas.append({"Territorio": f"{nombre} ({total} municipios)", "Perfil": grupo, "Municipios": n})
            if filas:
                df_comp = pd.DataFrame(filas)
                df_comp["% de municipios"] = df_comp["Municipios"] / df_comp.groupby("Territorio")["Municipios"].transform("sum") * 100
                fig = px.bar(df_comp, x="Territorio", y="% de municipios", color="Perfil",
                             color_discrete_sequence=PALETA, hover_data={"Municipios": True},
                             labels={"% de municipios": "% de municipios"})
                fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), barmode="stack", yaxis_title="% de municipios")
                st.plotly_chart(fig, width="stretch")
                if "SANTANDER" in comp and "VECINOS (otros dptos)" in comp:
                    dom_s = max(comp["SANTANDER"], key=comp["SANTANDER"].get)
                    dom_v = max(comp["VECINOS (otros dptos)"], key=comp["VECINOS (otros dptos)"].get)
                    st.caption(
                        f"Perfil más frecuente en Santander: **{dom_s}** ({comp['SANTANDER'][dom_s]} municipios). "
                        f"En los vecinos: **{dom_v}** ({comp['VECINOS (otros dptos)'][dom_v]} municipios)."
                    )

    with col_t:
        st.markdown("#### Fuentes de sismicidad (DBSCAN)")
        if met is not None and met.get("dbscan_fuentes"):
            df_f = pd.DataFrame(met["dbscan_fuentes"])
            df_f["dist"] = km_desde_nido(df_f["lat"], df_f["lon"])
            df_f["Identificación"] = np.where(df_f["dist"] <= 40, "Nido de Bucaramanga", "")
            df_f = df_f.sort_values("n", ascending=False)
            df_f = df_f.rename(columns={
                "fuente": "Fuente", "n": "Eventos", "lat": "Lat", "lon": "Lon",
                "prof_media_km": "Prof. media (km)", "mag_max": "Mag. máx.", "dist": "Dist. al Nido (km)",
            })
            df_f = df_f[["Fuente", "Eventos", "Lat", "Lon", "Prof. media (km)", "Mag. máx.", "Dist. al Nido (km)", "Identificación"]]
            es_nido = df_f["Identificación"].ne("")

            def resaltar(fila):
                return ["background-color: #FDE2E2; font-weight: 600"] * len(fila) if es_nido.loc[fila.name] else [""] * len(fila)

            st.dataframe(
                df_f.style.apply(resaltar, axis=1).format({"Lat": "{:.3f}", "Lon": "{:.3f}",
                                                           "Prof. media (km)": "{:.0f}", "Mag. máx.": "{:.1f}",
                                                           "Dist. al Nido (km)": "{:.0f}"}),
                hide_index=True, width="stretch",
            )
            st.caption("La fila resaltada es el Nido: la fuente más activa, a pocos km del epicentro de referencia.")

    st.markdown("#### Centroides de los grupos (valores reales)")
    if met is not None and met.get("centroides"):
        cent = pd.DataFrame(met["centroides"]).T
        cent = cent.rename(columns=etiqueta).round(2)
        cent.index.name = "Grupo"
        st.dataframe(cent, width="stretch")
        st.caption("Cada fila es el perfil promedio de un grupo. Por ejemplo, el valor de profundidad permite ver "
                   "qué grupos tienen sismos superficiales y cuáles de profundidad intermedia.")


# ----------------------------------------------------------------- pestaña 4: agente RL
ORDEN_LINEAS = [
    ("agente_rl", "Agente RL"),
    ("ranking_supervisado", "Ranking supervisado"),
    ("indice_pesos_iguales", "Índice pesos iguales"),
    ("mayor_actividad_pasada", "Mayor actividad pasada"),
    ("mayor_poblacion", "Mayor población"),
    ("azar", "Azar"),
    ("oraculo", "Oráculo (techo teórico)"),
]


def render_rl() -> None:
    st.subheader("Agente de aprendizaje por refuerzo: aprender a repartir un presupuesto limitado")
    st.markdown(
        "Un agente debe elegir **K = 10 de los 87 municipios de Santander** cada año. Recibe una recompensa según "
        "qué tan alto sea el riesgo de los municipios elegidos, y aprende qué combinación de variables le conviene."
    )
    formulacion = pd.DataFrame([
        ("Entorno", "Cada año es un episodio. Entrenamiento 2006–2015, prueba 2020–2024. Son los 87 municipios de Santander."),
        ("Estado", f"Por municipio: {len(config.FEATURES_RL)} variables estandarizadas (amenaza, exposición, vulnerabilidad) y su grupo no supervisado."),
        ("Acción", f"Elegir K = {config.K_PRESUPUESTO} municipios distintos. La política asigna una probabilidad a cada municipio."),
        ("Recompensa", "Amenaza del año siguiente × (½ exposición + ½ vulnerabilidad). La energía futura solo se usa para calificar, nunca como entrada."),
        ("Algoritmo", f"REINFORCE con política softmax lineal · tasa de aprendizaje {config.RL_LR} · entropía {config.RL_ENTROPIA} · {config.RL_EPOCAS} épocas · {len(config.SEMILLAS_RL)} semillas."),
        ("Medida de éxito", "Porcentaje del oráculo (el mejor presupuesto posible con datos reales) y precisión@K frente al oráculo."),
    ], columns=["Componente", "Definición en este proyecto"])
    st.dataframe(formulacion, hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Curva de aprendizaje")
    cur = obtener_csv(RUTA_CURVA_RL, CMD_RL, con_cod=False, nivel="warning")
    if cur is not None:
        g = cur.groupby("epoca").agg(
            rec_m=("recompensa_train", "mean"), rec_s=("recompensa_train", "std"),
            or_m=("pct_oraculo_val", "mean"), or_s=("pct_oraculo_val", "std"),
        ).reset_index().fillna(0)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Recompensa en entrenamiento", "% del oráculo en validación"))
        for col, (m, s, color) in enumerate([("rec_m", "rec_s", PALETA[0]), ("or_m", "or_s", PALETA[1])], start=1):
            x = g["epoca"]
            fig.add_trace(go.Scatter(x=x, y=g[m] + g[s], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=col)
            fig.add_trace(go.Scatter(x=x, y=g[m] - g[s], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(15,118,110,0.18)" if col == 1 else "rgba(217,119,6,0.18)",
                                     showlegend=False, hoverinfo="skip"), row=1, col=col)
            fig.add_trace(go.Scatter(x=x, y=g[m], mode="lines", line=dict(color=color, width=2.5),
                                     name="media entre semillas", showlegend=(col == 1)), row=1, col=col)
        fig.update_xaxes(title_text="Época")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Línea: media entre semillas. Banda: ± 1 desviación estándar. Si la curva se aplana, el agente ya aprendió.")

    st.markdown("#### Resultados en test (2020–2024) frente a las líneas base")
    met = obtener_json(RUTA_MET_RL, CMD_RL, nivel="warning")
    if met is not None:
        test = met.get("test", {})
        ag = test.get("agente_rl", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Recompensa media del agente", num(ag.get("recompensa_media", float("nan")), 3),
                  delta=f"± {num(ag.get('std_semillas', float('nan')), 3)} entre semillas", delta_color="off")
        c2.metric("Porcentaje del oráculo", f"{num(ag.get('pct_oraculo', float('nan')), 1)} %")
        c3.metric("Precisión@K frente al oráculo", pct(ag.get("precision_at_k", float("nan")), 0),
                  help="Cuántos de los K municipios elegidos coinciden con los K mejores del oráculo, promediado por año")

        filas = []
        for clave, nombre in ORDEN_LINEAS:
            d = test.get(clave)
            if not d:
                continue
            pct_or = d.get("pct_oraculo", 100.0 if clave == "oraculo" else np.nan)
            prec = d.get("precision_at_k", 1.0 if clave == "oraculo" else np.nan)
            filas.append({"Estrategia": nombre, "% del oráculo": float(pct_or),
                          "Precisión@K (%)": float(prec) * 100 if pd.notna(prec) else np.nan})
        if filas:
            df_c = pd.DataFrame(filas)
            largo = df_c.melt(id_vars="Estrategia", var_name="Métrica", value_name="Valor")
            fig = px.bar(largo, x="Estrategia", y="Valor", color="Métrica", barmode="group",
                         color_discrete_sequence=[PALETA[0], PALETA[1]], text_auto=".0f")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="%", xaxis_title="")
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("Los resultados aparecerán cuando el agente termine de entrenarse.")

    st.markdown("#### Pesos de la política aprendida")
    if met is not None and met.get("pesos_politica"):
        pesos = met["pesos_politica"]
        pdf = pd.DataFrame([{"Variable": etiqueta(k), "media": v["media"], "std": v["std"]} for k, v in pesos.items()])
        pdf = pdf.iloc[pdf["media"].abs().argsort()]
        pdf["Efecto"] = np.where(pdf["media"] >= 0, "Aumenta la prioridad", "Reduce la prioridad")
        fig = px.bar(pdf, x="media", y="Variable", orientation="h", error_x="std", color="Efecto",
                     color_discrete_map={"Aumenta la prioridad": PALETA[0], "Reduce la prioridad": PALETA[1]},
                     labels={"media": "Peso medio (± std entre semillas)", "Variable": ""})
        fig.update_layout(height=max(360, 26 * len(pdf)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Pesos de variables estandarizadas: un peso positivo sube la probabilidad de que el municipio sea elegido.")


# ----------------------------------------------------------------- pestaña 5: priorización 2026
def mapa_prioridad(df: pd.DataFrame, k: int) -> go.Figure:
    top = df[df["rango"] <= k]
    capas = [dict(
        lat=top["lat"].tolist(), lon=top["lon"].tolist(), mode="markers",
        marker=dict(size=(top["tamano"] + 9).tolist(), color="#111827"),
        name=f"Top-{k} (contorno negro)", hoverinfo="skip",
    )]
    for conf in ["alta", "media", "baja"]:
        sub = df[df["confianza"] == conf]
        if sub.empty:
            continue
        textos = [f"#{r.rango} {r.nom_mpio}<br>Puntaje {num(r.puntaje, 2)} · confianza {r.confianza}"
                  for r in sub.itertuples()]
        capas.append(dict(
            lat=sub["lat"].tolist(), lon=sub["lon"].tolist(), mode="markers",
            marker=dict(size=sub["tamano"].tolist(), color=COLOR_CONF[conf]),
            name=f"Confianza {conf}", hovertext=textos, hoverinfo="text",
        ))
    capas.append(dict(
        lat=[config.NIDO_LAT], lon=[config.NIDO_LON], mode="markers",
        marker=dict(size=18, color=COLOR_NIDO), name="Nido sísmico de Bucaramanga",
        hovertext=["Nido sísmico de Bucaramanga"], hoverinfo="text",
    ))
    if modo_compatible():
        fig = mapa_plano(capas, 580)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.12))
        return fig
    try:
        fig = go.Figure([go.Scattermap(**c) for c in capas])
        fig.update_layout(map=dict(style="carto-positron", center=CENTRO_SANTANDER, zoom=6.5))
    except Exception:  # noqa: BLE001 - respaldo si el mapa base no está disponible
        fig = go.Figure([go.Scattergeo(**c) for c in capas])
        fig.update_geos(
            lataxis_range=[5.6, 8.4], lonaxis_range=[-74.4, -72.4], projection_type="mercator",
            showland=True, landcolor="#F1F5F4", showcountries=True, countrycolor="#CBD5E1",
        )
    fig.update_layout(height=580, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.02))
    return fig


def render_priorizacion() -> None:
    st.subheader("Priorización 2026: dónde mirar primero")
    st.caption("Puntaje de riesgo del agente para el año 2026 (2026 está incompleto; se usa solo para priorizar).")
    df = obtener_csv(config.PRIORIZACION_CSV, CMD_RL, nivel="warning")
    if df is None:
        return
    df = df.sort_values("rango").reset_index(drop=True)
    pmin, pmax = float(df["puntaje"].min()), float(df["puntaje"].max())
    norm = (df["puntaje"] - pmin) / (pmax - pmin) if pmax > pmin else 0.0
    df["tamano"] = 7 + 15 * norm

    k = st.slider("Presupuesto K: municipios a intervenir", min_value=5, max_value=20,
                  value=config.K_PRESUPUESTO, step=1)
    top = df[df["rango"] <= k]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Municipios en el top-{k}", len(top))
    c2.metric("Confianza alta en el top-K", int((top["confianza"] == "alta").sum()))
    c3.metric("Población cubierta por el top-K", entero(top["poblacion_total"].sum()),
              help="Suma de la población DANE 2020 de los municipios seleccionados")

    st.plotly_chart(mapa_prioridad(df, k), width="stretch")
    st.caption(
        "Tamaño del punto ∝ puntaje (normalizado). Color = confianza de la posición en el ranking "
        "(alta ≥ 0,8 · media ≥ 0,5 · baja). El contorno negro marca el top-K. La estrella roja es el Nido."
    )

    st.markdown(f"#### Top-{k}")
    tabla = top[["rango", "nom_mpio", "puntaje", "prob_top_k", "confianza", "contrib_amenaza",
                 "contrib_exposicion", "contrib_vulnerabilidad", "p_alto", "cluster_nombre", "poblacion_total"]]
    st.dataframe(
        tabla, hide_index=True, width="stretch",
        column_config={
            "rango": st.column_config.NumberColumn("Rango", format="%d"),
            "nom_mpio": "Municipio",
            "puntaje": st.column_config.NumberColumn("Puntaje", format="%.2f"),
            "prob_top_k": st.column_config.ProgressColumn("Prob. en top-K", format="%.2f", min_value=0.0, max_value=1.0),
            "confianza": "Confianza",
            "contrib_amenaza": st.column_config.NumberColumn("Contrib. amenaza", format="%.2f"),
            "contrib_exposicion": st.column_config.NumberColumn("Contrib. exposición", format="%.2f"),
            "contrib_vulnerabilidad": st.column_config.NumberColumn("Contrib. vulnerabilidad", format="%.2f"),
            "p_alto": st.column_config.ProgressColumn("P(actividad alta)", format="%.2f", min_value=0.0, max_value=1.0),
            "cluster_nombre": "Perfil sísmico",
            "poblacion_total": st.column_config.NumberColumn("Población", format="%d"),
        },
    )
    st.caption("Probabilidad en top-K: fracción de simulaciones del agente en que el municipio entra al top-10. "
               "Contribuciones: aporte de cada criterio al puntaje (positivo = empuja a priorizar).")

    st.divider()
    st.markdown("#### Ficha del municipio")
    opciones = df["cod_mpio"].tolist()
    nombres = dict(zip(df["cod_mpio"], df["nom_mpio"]))
    rangos = dict(zip(df["cod_mpio"], df["rango"]))
    cod = st.selectbox("Municipio", opciones, format_func=lambda c: f"#{rangos[c]} · {nombres[c]}")
    fila = df[df["cod_mpio"] == cod].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rango", int(fila["rango"]))
    m2.metric("Confianza", str(fila["confianza"]).capitalize())
    m3.metric("P(actividad alta el año siguiente)", pct(fila["p_alto"]))
    m4.metric("Población (DANE 2020)", entero(fila["poblacion_total"]))
    m5.metric("Sismos M ≥ 2,5 en 2025", entero(fila["n_m25_2025"]))

    izq, der = st.columns(2, gap="large")
    with izq:
        contr = pd.DataFrame({
            "Criterio": ["Amenaza", "Exposición", "Vulnerabilidad"],
            "Aporte": [fila["contrib_amenaza"], fila["contrib_exposicion"], fila["contrib_vulnerabilidad"]],
        })
        fig = px.bar(contr, x="Aporte", y="Criterio", orientation="h", color="Criterio",
                     color_discrete_map=COLOR_CRITERIO, title="Por qué este municipio tiene este puntaje")
        fig.update_layout(height=330, showlegend=False, margin=dict(l=10, r=10, t=50, b=10),
                          yaxis_title="", xaxis_title="Aporte al puntaje")
        st.plotly_chart(fig, width="stretch")
    with der:
        ds = obtener_csv(config.DATASET_CSV, CMD_TABLA)
        if ds is not None:
            hist = ds[ds["cod_mpio"] == cod].sort_values("anio")
            fig = px.bar(hist, x="anio", y="n_m25", title="Historia sísmica: sismos M ≥ 2,5 por año",
                         labels={"anio": "Año", "n_m25": "Sismos M ≥ 2,5"})
            fig.update_traces(marker_color=PALETA[0])
            fig.add_vline(x=2018, line_dash="dot", line_color="#6B7280")
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
            if not hist.empty:
                emerg = hist["emergencias_prom_anual"].iloc[-1]
                st.caption(f"Emergencias UNGRD promedio por año: {num(emerg, 1)}. "
                           f"Perfil sísmico: {fila['cluster_nombre']}.")

    st.warning(
        "**Limitaciones.** Esta es una priorización exploratoria: no predice sismos ni dónde ocurrirán daños. "
        "El catálogo cambió de red en 2018, la población es una proyección de 2020 y las emergencias históricas "
        "no capturan todos los daños. Sirve para decidir qué revisar primero, no como base única de inversión."
    )


# ----------------------------------------------------------------- estructura principal
# Navegación que solo dibuja la sección activa: con st.tabs los mapas de pestañas ocultas
# se inicializan con tamaño 0 y quedan en blanco al abrirlas.
SECCIONES = {
    "Problema y datos": render_problema,
    "Supervisado": render_supervisado,
    "No supervisado": render_nosupervisado,
    "Agente RL": render_rl,
    "Priorización 2026": render_priorizacion,
}
seccion = st.segmented_control(
    "Sección", list(SECCIONES), default="Problema y datos", key="seccion",
    label_visibility="collapsed",
) or "Problema y datos"
SECCIONES[seccion]()
