"""Componente no supervisado: perfiles municipales y agrupamiento (CONTRATOS §5).

Diagrama de clases: docs/uml/03_clases.md (módulo `no_supervisado`).

Flujo de `ejecutar()`:
  perfil 2002–2025 -> elegir_k (k=2..8) -> k = argmax silueta con k>=3 -> ajustar_clusters ->
  nombrar_clusters -> comparacion_territorial -> importancia_variables -> fuentes_dbscan ->
  cluster_rl (perfil 2002–2015, mismo k) -> CSV, JSON y 4 figuras.

Este componente es SOLO no supervisado: no usa etiquetas (nivel_actividad_t1 ni *_t1).
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import davies_bouldin_score, silhouette_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from sismo import config  # noqa: E402

PREFIJO = "[no_supervisado]"

FEATURES_PERFIL = config.FEATURES_PERFIL
# Variables a las que se aplica log1p antes de estandarizar (CONTRATOS §5).
FEATURES_LOG1P = ["tasa_m25_anual", "tasa_m30_anual", "tasa_m35_anual", "log_energia_anual"]
MAG_DBSCAN = 3.0
EPS_INICIO, EPS_PASO, EPS_MAX = 0.3, 0.1, 0.6
MIN_MUESTRAS_DBSCAN = 30
MIN_EVENTOS_B = 30              # por debajo de este n de eventos, b = b regional
ETIQUETA_VECINOS = "VECINOS (otros dptos)"
CONTEXTO = ["nom_mpio", "dpto", "es_santander", "lat", "lon"]


def _es_atipico(serie):
    """Máscara booleana de es_atipico, tolerante a bool o texto 'True'/'False'."""
    return serie.astype(str).str.lower().eq("true")


def _filtrar_eventos(eventos, anios):
    """Eventos M >= MAG_MIN, no atípicos, dentro del periodo y con cod_mpio."""
    ini, fin = anios
    ev = eventos[eventos["anio"].between(ini, fin)]
    ev = ev[(ev["mag"] >= config.MAG_MIN) & (~_es_atipico(ev["es_atipico"]))]
    return ev[ev["cod_mpio"].notna()]


def perfil_municipal(df, anios=(config.ANIO_INICIO, config.ANIO_FIN), eventos=None):
    """Perfil de cada municipio en un periodo (176 filas, sin NaN).

    - Tasas anuales y fracción de años activos: desde el panel `df` (conteos por año).
    - Magnitud, profundidad, b-value y energía: desde los eventos individuales (M >= MAG_MIN,
      no atípicos), no promediando promedios.
    - Municipios sin eventos: mag/prof/pct -> 0; b-value -> b regional del periodo.
    """
    ini, fin = anios
    n_anios = fin - ini + 1
    if eventos is None:
        eventos = pd.read_csv(config.EVENTOS_CSV, dtype={"cod_mpio": str})

    codigos = pd.Series(sorted(df["cod_mpio"].unique()), name="cod_mpio")
    d = df[df["anio"].between(ini, fin)]
    ev = _filtrar_eventos(eventos, anios)
    ev = ev.assign(energia=10.0 ** (1.5 * ev["mag"] + 4.8))
    prof = ev["prof_km"]

    # Tasas anuales (media de conteos por año del panel).
    tasas = d.groupby("cod_mpio")[["n_m25", "n_m30", "n_m35"]].mean()
    tasas.columns = ["tasa_m25_anual", "tasa_m30_anual", "tasa_m35_anual"]
    activos = d.assign(activo=d["n_m25"] > 0).groupby("cod_mpio")["activo"].mean()
    activos.name = "fraccion_anios_activos"

    g = ev.groupby("cod_mpio")
    mag_max = g["mag"].max().rename("mag_max")
    mag_media = g["mag"].mean().rename("mag_media")
    prof_media = g["prof_km"].mean().rename("prof_media_km")
    pct_sup = (prof < 30.0).groupby(ev["cod_mpio"]).mean().rename("pct_superficial")
    pct_int = ((prof >= 70.0) & (prof <= 300.0)).groupby(ev["cod_mpio"]).mean().rename("pct_intermedio")
    log_energia = np.log10(g["energia"].sum() / n_anios).rename("log_energia_anual")

    # b-value de Aki por municipio (n >= 30) y b regional del periodo como respaldo.
    f_corr = config.MAG_MIN - 0.05
    b_reg = np.log10(np.e) / (ev["mag"].mean() - f_corr) if len(ev) else 0.0
    n_ev = g["mag"].size()
    media_m = g["mag"].mean()
    b_mpio = np.log10(np.e) / (media_m - f_corr)
    b_val = b_mpio.where(n_ev >= MIN_EVENTOS_B, b_reg).rename("b_value")

    out = pd.DataFrame({"cod_mpio": codigos})
    out = out.set_index("cod_mpio")
    for serie in [tasas, activos, mag_max, mag_media, b_val, prof_media, pct_sup, pct_int, log_energia]:
        out = out.join(serie, how="left")
    out["b_value"] = out["b_value"].fillna(b_reg)
    out = out.fillna(0.0)
    out = out[FEATURES_PERFIL].reset_index()

    ctx = df.groupby("cod_mpio")[CONTEXTO].first().reset_index()
    ctx["es_santander"] = ctx["es_santander"].astype(int)
    perfil = ctx.merge(out, on="cod_mpio", how="inner")
    perfil = perfil[["cod_mpio", *CONTEXTO, *FEATURES_PERFIL]]
    assert perfil[FEATURES_PERFIL].notna().all().all(), "perfil con NaN"
    return perfil.reset_index(drop=True)


def _matriz(perfil):
    """Matriz de clustering: log1p en tasas y energía, luego StandardScaler."""
    X = perfil[FEATURES_PERFIL].astype(float).copy()
    for c in FEATURES_LOG1P:
        X[c] = np.log1p(X[c].clip(lower=0))
    esc = StandardScaler()
    return esc.fit_transform(X.to_numpy()), esc


def elegir_k(X, ks=range(2, 9)):
    """Métricas de calidad para cada k: silueta, Davies-Bouldin, inercia y BIC de GMM."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=config.SEMILLA).fit(X)
        gmm = GaussianMixture(n_components=k, random_state=config.SEMILLA).fit(X)
        filas.append({
            "k": int(k),
            "silueta": float(silhouette_score(X, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
            "inercia": float(km.inertia_),
            "gmm_bic": float(gmm.bic(X)),
        })
    return pd.DataFrame(filas)


def ajustar_clusters(perfil, k):
    """K-Means (k) sobre el perfil, con PCA(2) para visualizar.

    Devuelve (perfil con cluster, pca1, pca2; info con modelo, escalador, pca y centroides
    como medias por cluster en unidades reales).
    """
    X, esc = _matriz(perfil)
    km = KMeans(n_clusters=k, n_init=20, random_state=config.SEMILLA).fit(X)
    pca = PCA(n_components=2)
    xy = pca.fit_transform(X)
    out = perfil.copy()
    out["cluster"] = km.labels_.astype(int)
    out["pca1"] = xy[:, 0]
    out["pca2"] = xy[:, 1]
    centroides = out.groupby("cluster")[FEATURES_PERFIL].mean()
    info = {"modelo": km, "escalador": esc, "pca": pca, "centroides": centroides}
    return out, info


def nombrar_clusters(perfil):
    """Nombra cada cluster con reglas sobre sus centroides en unidades reales.

    Orden de las reglas (corregido tras la revisión del orquestador):
    1) mayor pct_intermedio (desempate: mayor tasa_m25) -> Nido / sismicidad intermedia
    2) de los restantes, menor tasa_m25 -> Baja actividad
    3) de los restantes, mayor pct_superficial -> Cortical / superficial
    4) restantes con pct_intermedio >= 0,5, por tasa_m25 desc: el primero ->
       Intermedia activa (periferia del Nido), el siguiente -> Intermedia moderada
    5) cualquier otro -> Moderada
    """
    cent = perfil.groupby("cluster")[FEATURES_PERFIL].mean()
    nombres = {}
    # 1) Nido: mayor pct_intermedio; desempate por mayor tasa_m25.
    orden_nido = cent.sort_values(["pct_intermedio", "tasa_m25_anual"], ascending=False)
    nido = int(orden_nido.index[0])
    nombres[nido] = "Nido / sismicidad intermedia"
    restantes = cent.drop(index=nido)
    # 2) Baja actividad: menor tasa_m25 entre los restantes.
    if len(restantes):
        baja = int(restantes["tasa_m25_anual"].idxmin())
        nombres[baja] = "Baja actividad"
        restantes = restantes.drop(index=baja)
    # 3) Cortical / superficial: mayor pct_superficial entre los restantes.
    if len(restantes):
        cortical = int(restantes["pct_superficial"].idxmax())
        nombres[cortical] = "Cortical / superficial"
        restantes = restantes.drop(index=cortical)
    # 4) Intermedias: restantes con pct_intermedio >= 0,5, por tasa_m25 descendente.
    intermedias = restantes[restantes["pct_intermedio"] >= 0.5].sort_values(
        "tasa_m25_anual", ascending=False)
    for c, nombre in zip(intermedias.index, ["Intermedia activa (periferia del Nido)",
                                             "Intermedia moderada"]):
        nombres[int(c)] = nombre
    restantes = restantes.drop(index=intermedias.index[:2], errors="ignore")
    # 5) Cualquier otro: Moderada.
    for c in restantes.index:
        nombres[int(c)] = "Moderada"
    # Desduplicar con sufijo numérico (solo si hace falta).
    vistos, final = {}, {}
    for c in sorted(nombres):
        base = nombres[c]
        vistos[base] = vistos.get(base, 0) + 1
        final[c] = base if vistos[base] == 1 else f"{base} {vistos[base]}"
    return final


def comparacion_territorial(perfil):
    """Conteos dpto × cluster_nombre, más una fila de resumen Santander vs. vecinos."""
    if "cluster_nombre" not in perfil.columns:
        raise ValueError("comparacion_territorial requiere la columna cluster_nombre")
    ct = pd.crosstab(perfil["dpto"], perfil["cluster_nombre"]).sort_index()
    vecinos = perfil.loc[perfil["es_santander"] == 0, "cluster_nombre"].value_counts()
    fila_vec = vecinos.reindex(ct.columns, fill_value=0).astype(int)
    ct.loc[ETIQUETA_VECINOS] = fila_vec
    return ct.astype(int)


def importancia_variables(perfil):
    """Árbol de decisión (profundidad 4) sustituto que predice el cluster desde FEATURES_PERFIL."""
    X = StandardScaler().fit_transform(perfil[FEATURES_PERFIL].astype(float).to_numpy())
    arbol = DecisionTreeClassifier(max_depth=4, random_state=config.SEMILLA)
    arbol.fit(X, perfil["cluster"].to_numpy())
    imp = pd.DataFrame({"feature": FEATURES_PERFIL, "importancia": arbol.feature_importances_})
    return imp.sort_values("importancia", ascending=False).reset_index(drop=True)


def fuentes_dbscan(eventos):
    """Top-5 fuentes sísmicas: DBSCAN en (x, y, profundidad) de eventos M >= 3,0 no atípicos."""
    mascara = (
        (eventos["mag"] >= MAG_DBSCAN)
        & (~_es_atipico(eventos["es_atipico"]))
        & eventos["anio"].between(config.ANIO_INICIO, config.ANIO_FIN)
    )
    ev = eventos.loc[mascara].dropna(subset=["lat", "lon", "prof_km"])
    x = (ev["lon"] - config.NIDO_LON) * 111.0 * np.cos(np.radians(config.NIDO_LAT))
    y = (ev["lat"] - config.NIDO_LAT) * 111.0
    z = ev["prof_km"]
    P = StandardScaler().fit_transform(np.column_stack([x, y, z]))

    eps_usado, etiquetas = None, None
    for eps in np.round(np.arange(EPS_INICIO, EPS_MAX + 1e-9, EPS_PASO), 2):
        etiquetas = DBSCAN(eps=float(eps), min_samples=MIN_MUESTRAS_DBSCAN).fit_predict(P)
        eps_usado = float(eps)
        if len(set(etiquetas) - {-1}) >= 1:
            break
    print(f"{PREFIJO} DBSCAN: eps={eps_usado}, min_samples={MIN_MUESTRAS_DBSCAN}, "
          f"clusters={len(set(etiquetas) - {-1})}, ruido={int((etiquetas == -1).sum())}")

    ev = ev.assign(_lab=etiquetas)
    filas = []
    for lab, grupo in ev[ev["_lab"] >= 0].groupby("_lab"):
        filas.append({
            "n": int(len(grupo)),
            "lat": float(grupo["lat"].mean()),
            "lon": float(grupo["lon"].mean()),
            "prof_media_km": float(grupo["prof_km"].mean()),
            "mag_max": float(grupo["mag"].max()),
        })
    if not filas:
        return pd.DataFrame(columns=["fuente", "n", "lat", "lon", "prof_media_km", "mag_max"])
    tabla = pd.DataFrame(filas).sort_values("n", ascending=False).head(5).reset_index(drop=True)
    tabla.insert(0, "fuente", np.arange(1, len(tabla) + 1))
    tabla.attrs["eps"] = eps_usado
    return tabla


def _alinear_etiquetas(ref, otro):
    """Renombra las etiquetas de `otro` para que coincidan lo más posible con las de `ref`.

    ref y otro: DataFrames con cod_mpio y cluster. Devuelve dict etiqueta_otro -> etiqueta_ref
    (asignación óptima por Hungarian sobre la tabla de contingencia).
    """
    m = ref[["cod_mpio", "cluster"]].merge(otro[["cod_mpio", "cluster"]], on="cod_mpio",
                                          suffixes=("_ref", "_otro"))
    ct = pd.crosstab(m["cluster_otro"], m["cluster_ref"])
    fila, col = linear_sum_assignment(-ct.to_numpy())
    return {int(ct.index[i]): int(ct.columns[j]) for i, j in zip(fila, col)}


# ----------------------------------------------------------------- figuras
def _figura_seleccion_k(sel, k, ruta):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(sel["k"], sel["silueta"], marker="o", color="#1f77b4")
    a1.axvline(k, color="grey", ls="--", lw=1)
    a1.set_title("Silueta por k (mayor = mejor)")
    a1.set_xlabel("k")
    a1.set_ylabel("Silueta")
    a2.plot(sel["k"], sel["davies_bouldin"], marker="o", color="#d62728")
    a2.axvline(k, color="grey", ls="--", lw=1)
    a2.set_title("Davies-Bouldin por k (menor = mejor)")
    a2.set_xlabel("k")
    a2.set_ylabel("Davies-Bouldin")
    fig.suptitle(f"Selección de k (k elegido = {k})")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def _figura_mapa(perfil, ruta):
    fig, ax = plt.subplots(figsize=(7, 8))
    nombres = sorted(perfil["cluster_nombre"].unique())
    cmap = plt.get_cmap("tab10")
    for i, nombre in enumerate(nombres):
        sub = perfil[perfil["cluster_nombre"] == nombre]
        sant = sub[sub["es_santander"] == 1]
        otros = sub[sub["es_santander"] == 0]
        ax.scatter(otros["lon"], otros["lat"], s=35, color=cmap(i), label=nombre, alpha=0.85)
        ax.scatter(sant["lon"], sant["lat"], s=45, color=cmap(i), edgecolors="black", linewidths=1.4)
    ax.scatter([config.NIDO_LON], [config.NIDO_LAT], marker="*", s=300, color="red",
               edgecolors="black", label="Nido de Bucaramanga", zorder=5)
    ax.set_title("Clusters de municipios (borde negro = Santander)")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_aspect(1.0 / np.cos(np.radians(config.NIDO_LAT)))
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def _figura_pca(perfil, ruta):
    fig, ax = plt.subplots(figsize=(8, 6))
    nombres = sorted(perfil["cluster_nombre"].unique())
    cmap = plt.get_cmap("tab10")
    for i, nombre in enumerate(nombres):
        sub = perfil[perfil["cluster_nombre"] == nombre]
        ax.scatter(sub["pca1"], sub["pca2"], s=35, color=cmap(i), label=nombre, alpha=0.85)
    top5 = perfil.nlargest(5, "tasa_m25_anual")
    for _, fila in top5.iterrows():
        ax.annotate(fila["nom_mpio"], (fila["pca1"], fila["pca2"]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_title("Municipios en espacio PCA (5 más activos etiquetados)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def _figura_comparacion(comp, ruta):
    tabla = comp.drop(index=ETIQUETA_VECINOS, errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 5))
    tabla.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", width=0.7)
    ax.set_title("Municipios por departamento y cluster")
    ax.set_xlabel("Departamento")
    ax.set_ylabel("Número de municipios")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    ax.legend(title="Cluster", fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- orquestación
def ejecutar():
    """Corre el componente completo, guarda CSV/JSON/figuras y devuelve el dict de métricas."""
    print(f"{PREFIJO} cargando dataset y eventos...")
    df = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    eventos = pd.read_csv(config.EVENTOS_CSV, dtype={"cod_mpio": str})
    anios = (config.ANIO_INICIO, config.ANIO_FIN)

    perfil = perfil_municipal(df, anios, eventos)
    print(f"{PREFIJO} perfil completo: {len(perfil)} municipios x {len(FEATURES_PERFIL)} variables")

    X, _ = _matriz(perfil)
    sel = elegir_k(X, range(2, 9))
    candidatos = sel[sel["k"] >= 3]
    k = int(candidatos.loc[candidatos["silueta"].idxmax(), "k"])
    print(f"{PREFIJO} k elegido = {k} (silueta {float(sel.loc[sel['k'] == k, 'silueta'].iloc[0]):.3f})")

    perfil_c, info = ajustar_clusters(perfil, k)
    nombres = nombrar_clusters(perfil_c)
    perfil_c["cluster_nombre"] = perfil_c["cluster"].map(nombres)
    comp = comparacion_territorial(perfil_c)
    imp = importancia_variables(perfil_c)
    fuentes = fuentes_dbscan(eventos)
    silueta_final = float(silhouette_score(X, perfil_c["cluster"].to_numpy()))

    # Versión para el agente RL: mismo k, ajustado solo con TRAIN (2002–2015).
    perfil_rl = perfil_municipal(df, config.TRAIN, eventos)
    rl, _ = ajustar_clusters(perfil_rl, k)
    mapa = _alinear_etiquetas(perfil_c, rl)
    rl_map = dict(zip(rl["cod_mpio"], rl["cluster"].map(mapa).astype(int)))
    perfil_c["cluster_rl"] = perfil_c["cod_mpio"].map(rl_map).astype(int)

    # Salidas
    columnas = ["cod_mpio", "nom_mpio", "dpto", "es_santander", "lat", "lon", "cluster",
                "cluster_nombre", "pca1", "pca2", "cluster_rl", *FEATURES_PERFIL]
    perfil_c[columnas].to_csv(config.CLUSTERS_CSV, index=False)

    cent = info["centroides"]
    metricas = {
        "k_elegido": k,
        "seleccion_k": [
            {kk: (int(v) if kk == "k" else float(v)) for kk, v in fila.items()}
            for fila in sel.to_dict("records")
        ],
        "silueta_final": silueta_final,
        "centroides": {
            nombres[int(c)]: {f: float(v) for f, v in fila.items()}
            for c, fila in cent.iterrows()
        },
        "comparacion_territorial": {
            str(dpto): {str(col): int(v) for col, v in fila.items()}
            for dpto, fila in comp.iterrows()
        },
        "importancia_variables": {r["feature"]: float(r["importancia"]) for _, r in imp.iterrows()},
        "dbscan_fuentes": [
            {"fuente": int(r["fuente"]), "n": int(r["n"]), "lat": float(r["lat"]),
             "lon": float(r["lon"]), "prof_media_km": float(r["prof_media_km"]),
             "mag_max": float(r["mag_max"])}
            for _, r in fuentes.iterrows()
        ],
    }
    with open(config.REPORTS / "metricas_no_supervisado.json", "w", encoding="utf-8") as fh:
        json.dump(metricas, fh, ensure_ascii=False, indent=2)

    _figura_seleccion_k(sel, k, config.FIGURES / "nosup_seleccion_k.png")
    _figura_mapa(perfil_c, config.FIGURES / "nosup_mapa_clusters.png")
    _figura_pca(perfil_c, config.FIGURES / "nosup_pca.png")
    _figura_comparacion(comp, config.FIGURES / "nosup_comparacion_territorial.png")

    print(f"{PREFIJO} archivos: {config.CLUSTERS_CSV.name}, metricas_no_supervisado.json, 4 figuras")
    return metricas
