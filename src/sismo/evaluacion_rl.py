"""Evaluación del agente RL de priorización sísmica (CONTRATOS §6, T3.2).

Diagrama de clases: docs/uml/03_clases.md (módulo `evaluacion_rl`).
Secuencia de entrenamiento: docs/uml/04_secuencias.md §4.2.

Flujo de `ejecutar()`:
  dataset + predicciones walk-forward + cluster_rl -> construir_estados (solo Santander,
  estandarización con TRAIN_RL, recompensa de t+1) -> entrenar un agente por semilla ->
  evaluar greedy en TEST_RL frente a líneas base y oráculo -> priorizar sobre 2025 ->
  CSV, JSON, npz y 4 figuras.

Sin fuga: la recompensa de t usa solo log_energia_t1 (riesgo realizado en t+1) y nunca entra
al estado. Las predicciones del supervisado son walk-forward. cluster_rl se ajusta con 2002–2015.
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sismo import config  # noqa: E402
from sismo.agente_rl import AgentePoliticaLineal  # noqa: E402
from sismo.entorno_rl import PriorizacionEnv  # noqa: E402

PREFIJO = "[rl]"
REPETICIONES_AZAR = 20
UMBRAL_ALTA_CONFIANZA = 0.8
UMBRAL_MEDIA_CONFIANZA = 0.5
COLORES_GRUPO = {"amenaza": "#c0392b", "exposicion": "#2471a3", "vulnerabilidad": "#1e8449"}


# ============================================================ estados y recompensas
def _minmax(v: np.ndarray) -> np.ndarray:
    """Escala a [0, 1] sobre el vector; si no hay rango devuelve ceros."""
    v = np.asarray(v, dtype=np.float64)
    rango = np.nanmax(v) - np.nanmin(v)
    if not np.isfinite(rango) or rango <= 0:
        return np.zeros_like(v)
    return (v - np.nanmin(v)) / rango


def construir_estados(df, pred, clusters, anios):
    """Construye estados estandarizados y recompensas de los municipios de Santander.

    Devuelve (estados, recompensas, nombres_feat, extra):
      estados[anio]     ndarray (n_mpios, n_feat), estandarizado con los años TRAIN_RL.
      recompensas[anio] ndarray (n_mpios,), riesgo realizado en t+1. NaN si no hay t+1 (2025).
      nombres_feat      orden fijo de las columnas (CONTRATOS §6).
      extra             {"media", "std", "cod_mpio", "anios_train", "k_clusters"}.
    """
    anios = sorted(int(a) for a in anios)
    sant = df[df["es_santander"] == 1].copy()
    sant["cod_mpio"] = sant["cod_mpio"].astype(str)
    cods = sorted(sant["cod_mpio"].unique())
    k_clusters = int(clusters["cluster_rl"].max()) + 1

    pred = pred.copy()
    pred["cod_mpio"] = pred["cod_mpio"].astype(str)
    cl = clusters.copy()
    cl["cod_mpio"] = cl["cod_mpio"].astype(str)
    cl = cl.drop_duplicates("cod_mpio").set_index("cod_mpio").reindex(cods)
    if cl["cluster_rl"].isna().any():
        raise ValueError("faltan municipios de Santander en clusters_municipios.csv")
    cluster_mpio = cl["cluster_rl"].astype(int).to_numpy()
    onehot = np.eye(k_clusters)[cluster_mpio]  # (n_mpios, k)

    # Referencia de escala de la amenaza: máximo de log_energia_t1 en los años TRAIN_RL.
    en_train = sant["anio"].between(config.TRAIN_RL[0], config.TRAIN_RL[1])
    ref_amenaza = float(sant.loc[en_train, "log_energia_t1"].max())
    if not np.isfinite(ref_amenaza) or ref_amenaza <= 0:
        raise ValueError("máximo de log_energia_t1 en TRAIN_RL no válido")

    nombres_cluster = [f"cluster_rl_{i}" for i in range(k_clusters)]
    nombres = (
        ["p_alto", "p_medio", "n_m25_prom_3a_log", "mag_max_historica", "dist_nido_km"]
        + nombres_cluster
        + ["log_poblacion", "pct_rural", "emergencias_log", "mov_masa_log", "viviendas_log"]
    )
    assert not any(n.endswith("_t1") for n in nombres), "una variable *_t1 entró al estado"

    crudos, recompensas = {}, {}
    for anio in anios:
        sub = sant[sant["anio"] == anio].set_index("cod_mpio").reindex(cods)
        pr = pred[pred["anio"] == anio].set_index("cod_mpio").reindex(cods)
        if sub.isna()[["poblacion_total", "n_m25_prom_3a"]].any().any():
            raise ValueError(f"datos incompletos en el año {anio}")
        if pr[["p_alto", "p_medio"]].isna().any().any():
            raise ValueError(f"faltan predicciones del supervisado en el año {anio}")

        bloque = np.column_stack([
            pr["p_alto"].to_numpy(dtype=np.float64),
            pr["p_medio"].to_numpy(dtype=np.float64),
            np.log1p(sub["n_m25_prom_3a"].to_numpy(dtype=np.float64)),
            sub["mag_max_historica"].to_numpy(dtype=np.float64),
            sub["dist_nido_km"].to_numpy(dtype=np.float64),
            onehot,
            np.log1p(sub["poblacion_total"].to_numpy(dtype=np.float64)),
            sub["pct_rural"].to_numpy(dtype=np.float64),
            np.log1p(sub["emergencias_prom_anual"].to_numpy(dtype=np.float64)),
            np.log1p(sub["mov_masa_total"].to_numpy(dtype=np.float64)),
            np.log1p(sub["viviendas_afectadas_total"].to_numpy(dtype=np.float64)),
        ])
        crudos[anio] = bloque

        # Recompensa: amenaza x (0.5 exposición + 0.5 vulnerabilidad), todo en el año t.
        lt1 = sub["log_energia_t1"].to_numpy(dtype=np.float64)
        amenaza = np.clip(lt1 / ref_amenaza, 0.0, 1.0)
        exposicion = _minmax(np.log1p(sub["poblacion_total"].to_numpy(dtype=np.float64)))
        vulnerab = (
            _minmax(np.log1p(sub["emergencias_prom_anual"].to_numpy(dtype=np.float64)))
            + _minmax(np.log1p(sub["mov_masa_total"].to_numpy(dtype=np.float64)))
            + _minmax(np.log1p(sub["viviendas_afectadas_total"].to_numpy(dtype=np.float64)))
        ) / 3.0
        recompensas[anio] = amenaza * (0.5 * exposicion + 0.5 * vulnerab)

    anios_train = [a for a in anios if config.TRAIN_RL[0] <= a <= config.TRAIN_RL[1]]
    if not anios_train:
        raise ValueError("ninguno de los años pedidos está en TRAIN_RL")
    apilado = np.vstack([crudos[a] for a in anios_train])
    media = apilado.mean(axis=0)
    std = apilado.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)

    estados = {a: (crudos[a] - media) / std for a in anios}
    extra = {
        "media": media,
        "std": std,
        "cod_mpio": cods,
        "anios_train": anios_train,
        "k_clusters": k_clusters,
    }
    return estados, recompensas, nombres, extra


# ============================================================ evaluación
def evaluar_politica(env, politica_fn, anios):
    """Ejecuta un episodio completo por año y promedia recompensa, % oráculo y precision@K.

    politica_fn(x_estandarizado, mascara, anio) -> índice del municipio a elegir.
    """
    k = env.k
    recompensas, pcts, precisiones = [], [], []
    for anio in anios:
        anio = int(anio)
        obs, info = env.reset(options={"anio": anio})
        total, terminado = 0.0, False
        while not terminado:
            x = obs[:, : env.n_feat]
            mascara = info["mascara"]
            a = int(politica_fn(x, mascara, anio))
            if not mascara[a]:
                raise ValueError(f"la política eligió un municipio ya elegido ({a}) en {anio}")
            obs, r, terminado, _, info = env.step(a)
            total += r
        seleccion = set(info["seleccion"])
        oraculo = env.oraculo(anio)
        recompensas.append(total)
        pcts.append(total / oraculo if oraculo > 0 else np.nan)
        precisiones.append(len(seleccion & set(env.topk_oraculo(anio).tolist())) / k)
    return {
        "recompensa_media": float(np.mean(recompensas)),
        "pct_oraculo": float(np.nanmean(pcts)),
        "precision_at_k": float(np.mean(precisiones)),
    }


def entrenar(env, agente, n_epocas, anios_train, anios_val, rng):
    """REINFORCE con episodios estocásticos por año. Cada 10 épocas mide greedy en validación.

    Devuelve DataFrame con columnas epoca, recompensa_train, pct_oraculo_val (NaN si no se evalúa).
    """
    filas = []
    for epoca in range(1, int(n_epocas) + 1):
        orden = rng.permutation(np.asarray(anios_train, dtype=int))
        totales = []
        for anio in orden:
            obs, info = env.reset(options={"anio": int(anio)})
            trayectoria, terminado = [], False
            while not terminado:
                x = np.array(obs[:, : agente.n_feat], dtype=np.float64)
                mascara = np.array(info["mascara"], dtype=bool)
                a = agente.actuar(x, mascara, greedy=False)
                obs, r, terminado, _, info = env.step(a)
                trayectoria.append((x, mascara, a, r))
            totales.append(agente.actualizar(trayectoria))

        pct_val = np.nan
        if epoca % 10 == 0:
            pct_val = evaluar_politica(
                env, lambda x, m, _a: agente.actuar(x, m, greedy=True), anios_val
            )["pct_oraculo"]
        filas.append({
            "epoca": epoca,
            "recompensa_train": float(np.mean(totales)),
            "pct_oraculo_val": float(pct_val),
        })
    return pd.DataFrame(filas)


def lineas_base(nombres_feat, rng):
    """Políticas de referencia. Devuelve dict nombre -> politica_fn(x, mascara, anio)."""
    idx = {n: i for i, n in enumerate(nombres_feat)}

    def _argmax_disponible(puntaje, mascara):
        return int(np.argmax(np.where(mascara, puntaje, -np.inf)))

    def azar(x, mascara, _anio):
        return int(rng.choice(np.flatnonzero(mascara)))

    def mayor_poblacion(x, mascara, _anio):
        return _argmax_disponible(x[:, idx["log_poblacion"]], mascara)

    def mayor_actividad_pasada(x, mascara, _anio):
        return _argmax_disponible(x[:, idx["n_m25_prom_3a_log"]], mascara)

    def ranking_supervisado(x, mascara, _anio):
        return _argmax_disponible(x[:, idx["p_alto"]], mascara)

    def indice_pesos_iguales(x, mascara, _anio):
        puntaje = (x[:, idx["p_alto"]] + x[:, idx["log_poblacion"]] + x[:, idx["emergencias_log"]]) / 3.0
        return _argmax_disponible(puntaje, mascara)

    return {
        "azar": azar,
        "mayor_poblacion": mayor_poblacion,
        "mayor_actividad_pasada": mayor_actividad_pasada,
        "ranking_supervisado": ranking_supervisado,
        "indice_pesos_iguales": indice_pesos_iguales,
    }


def politica_oraculo(recompensas):
    """Elige por recompensa realizada (techo teórico). Sirve de referencia y de prueba."""
    def _politica(x, mascara, anio):
        return int(np.argmax(np.where(mascara, recompensas[int(anio)], -np.inf)))
    return _politica


def _grupos(nombres_feat):
    """Índices de las columnas por criterio: amenaza, exposición y vulnerabilidad."""
    amenaza, exposicion, vulnerabilidad = [], [], []
    for i, nombre in enumerate(nombres_feat):
        if nombre in config.FEATURES_RL_AMENAZA or nombre.startswith("cluster_rl_"):
            amenaza.append(i)
        elif nombre in config.FEATURES_RL_EXPOSICION:
            exposicion.append(i)
        elif nombre in config.FEATURES_RL_VULNERABILIDAD:
            vulnerabilidad.append(i)
        else:
            raise ValueError(f"feature sin grupo: {nombre}")
    return amenaza, exposicion, vulnerabilidad


def priorizar(agentes, x_2025, info_mpios, nombres_feat=None):
    """Tabla de priorización 2026 (CONTRATOS §6) con puntaje, confianza y contribuciones.

    info_mpios: DataFrame con una fila por municipio, en el mismo orden que x_2025, con
    cod_mpio, nom_mpio, lat, lon, p_alto (crudo), cluster_nombre, poblacion_total, n_m25_2025.
    nombres_feat: orden de columnas de x_2025 (necesario para agrupar contribuciones).
    """
    if nombres_feat is None:
        raise ValueError("priorizar necesita nombres_feat para agrupar las contribuciones")
    x = np.asarray(x_2025, dtype=np.float64)
    n = x.shape[0]
    if len(info_mpios) != n:
        raise ValueError("info_mpios debe tener una fila por municipio de x_2025")
    info = info_mpios.reset_index(drop=True)
    amenaza, exposicion, vulnerabilidad = _grupos(nombres_feat)
    k = config.K_PRESUPUESTO

    puntajes = np.zeros((len(agentes), n))
    contrib = np.zeros((len(agentes), n, 3))
    hits = np.zeros(n)
    for s, agente in enumerate(agentes):
        puntajes[s] = agente.puntajes(x)
        c = agente.contribuciones(x)
        contrib[s, :, 0] = c[:, amenaza].sum(axis=1)
        contrib[s, :, 1] = c[:, exposicion].sum(axis=1)
        contrib[s, :, 2] = c[:, vulnerabilidad].sum(axis=1)
        # Top-K greedy: se eligen k municipios con el modo greedy del agente.
        mascara = np.ones(n, dtype=bool)
        for _ in range(k):
            a = agente.actuar(x, mascara, greedy=True)
            hits[a] += 1
            mascara[a] = False

    puntaje = puntajes.mean(axis=0)
    prob_top_k = hits / len(agentes)
    confianza = np.where(
        prob_top_k >= UMBRAL_ALTA_CONFIANZA, "alta",
        np.where(prob_top_k >= UMBRAL_MEDIA_CONFIANZA, "media", "baja"),
    )
    cont = contrib.mean(axis=0)

    tabla = pd.DataFrame({
        "cod_mpio": info["cod_mpio"].astype(str).to_numpy(),
        "nom_mpio": info["nom_mpio"].to_numpy(),
        "lat": info["lat"].to_numpy(),
        "lon": info["lon"].to_numpy(),
        "puntaje": puntaje,
        "prob_top_k": prob_top_k,
        "confianza": confianza,
        "contrib_amenaza": cont[:, 0],
        "contrib_exposicion": cont[:, 1],
        "contrib_vulnerabilidad": cont[:, 2],
        "p_alto": info["p_alto"].to_numpy(),
        "cluster_nombre": info["cluster_nombre"].to_numpy(),
        "poblacion_total": info["poblacion_total"].to_numpy(),
        "n_m25_2025": info["n_m25_2025"].to_numpy(),
    })
    tabla = tabla.sort_values("puntaje", ascending=False, kind="stable").reset_index(drop=True)
    tabla.insert(0, "rango", np.arange(1, n + 1))
    return tabla


# ============================================================ salidas
def _f(v):
    """Convierte a float nativo para JSON."""
    return float(v)


def _figuras(curva, metricas_test, pesos_media, pesos_std, nombres, priorizacion):
    config.FIGURES.mkdir(parents=True, exist_ok=True)

    # 1. Curva de aprendizaje: media ± std entre semillas (solo épocas evaluadas).
    evaluada = curva.dropna(subset=["pct_oraculo_val"])
    resumen = evaluada.groupby("epoca")["pct_oraculo_val"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(resumen["epoca"], resumen["mean"] * 100, marker="o", color="#1f4e79")
    ax.fill_between(resumen["epoca"], (resumen["mean"] - resumen["std"].fillna(0)) * 100,
                    (resumen["mean"] + resumen["std"].fillna(0)) * 100,
                    color="#1f4e79", alpha=0.2)
    ax.set_xlabel("Época")
    ax.set_ylabel("% del oráculo en validación (greedy)")
    ax.set_title("Curva de aprendizaje del agente RL (media ± desviación entre semillas)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "rl_curva_aprendizaje.png", dpi=120)
    plt.close(fig)

    # 2. Comparación en TEST: agente con barra de error entre semillas.
    nombres_bar = ["agente_rl", "indice_pesos_iguales", "ranking_supervisado",
                   "mayor_actividad_pasada", "mayor_poblacion", "azar"]
    etiquetas = ["Agente RL", "Índice pesos\niguales", "Ranking\nsupervisado",
                 "Mayor actividad\npasada", "Mayor\npoblación", "Azar"]
    valores = [metricas_test[n]["pct_oraculo"] * 100 for n in nombres_bar]
    errores = [metricas_test["agente_rl"]["std_semillas"] * 100] + [0.0] * (len(nombres_bar) - 1)
    colores = ["#1f4e79"] + ["#95a5a6"] * (len(nombres_bar) - 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(etiquetas, valores, yerr=errores, color=colores, capsize=5)
    for i, (v, e) in enumerate(zip(valores, errores)):
        ax.text(i, v + e + 1.5, f"{v:.1f} %", ha="center", fontsize=9)
    ax.set_ylabel("% del oráculo (TEST 2020–2024)")
    ax.set_title("Agente RL frente a líneas base en TEST")
    ax.set_ylim(0, max(valores) * 1.15 + 5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "rl_comparacion_lineas_base.png", dpi=120)
    plt.close(fig)

    # 3. Pesos de la política, con signo, coloreados por criterio.
    orden = np.argsort(pesos_media)
    grupo_de = {}
    for n in nombres:
        if n in config.FEATURES_RL_AMENAZA or n.startswith("cluster_rl_"):
            grupo_de[n] = "amenaza"
        elif n in config.FEATURES_RL_EXPOSICION:
            grupo_de[n] = "exposicion"
        else:
            grupo_de[n] = "vulnerabilidad"
    etiq = [nombres[i] for i in orden]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(etiq, pesos_media[orden], xerr=pesos_std[orden],
            color=[COLORES_GRUPO[grupo_de[nombres[i]]] for i in orden], capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Peso w (escala estandarizada), media ± std entre semillas")
    ax.set_title("Pesos aprendidos de la política")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g.capitalize()) for g, c in COLORES_GRUPO.items()],
              loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "rl_pesos_politica.png", dpi=120)
    plt.close(fig)

    # 4. Mapa de priorización 2026.
    p = priorizacion["puntaje"].to_numpy()
    norm = (p - p.min()) / (p.max() - p.min()) if p.max() > p.min() else np.zeros_like(p)
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.scatter(priorizacion["lon"], priorizacion["lat"], s=20 + 280 * norm,
               c="#7fb3d5", edgecolors="#1f4e79", alpha=0.7, label="Municipios")
    top = priorizacion[priorizacion["rango"] <= 10]
    ax.scatter(top["lon"], top["lat"], s=20 + 280 * norm[top.index.to_numpy()],
               c="#c0392b", edgecolors="black", alpha=0.85, label="Top-10 (2026)")
    for _, fila in top.iterrows():
        ax.annotate(f"{int(fila['rango'])}. {fila['nom_mpio'].title()}",
                    (fila["lon"], fila["lat"]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.scatter([config.NIDO_LON], [config.NIDO_LAT], marker="*", s=350, c="#f1c40f",
               edgecolors="black", zorder=5, label="Nido sísmico")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Priorización sísmica 2026 (tamaño ∝ puntaje)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "rl_mapa_priorizacion.png", dpi=120)
    plt.close(fig)


def _info_mpios(df, pred, clusters, cods, anio):
    """Información descriptiva de cada municipio para el año de priorización."""
    sub = df[(df["es_santander"] == 1) & (df["anio"] == anio)].copy()
    sub["cod_mpio"] = sub["cod_mpio"].astype(str)
    sub = sub.set_index("cod_mpio").reindex(cods)
    pr = pred[pred["anio"] == anio].copy()
    pr["cod_mpio"] = pr["cod_mpio"].astype(str)
    pr = pr.set_index("cod_mpio").reindex(cods)
    cl = clusters.copy()
    cl["cod_mpio"] = cl["cod_mpio"].astype(str)
    cl = cl.drop_duplicates("cod_mpio").set_index("cod_mpio").reindex(cods)
    return pd.DataFrame({
        "cod_mpio": cods,
        "nom_mpio": sub["nom_mpio"].to_numpy(),
        "lat": sub["lat"].to_numpy(),
        "lon": sub["lon"].to_numpy(),
        "p_alto": pr["p_alto"].to_numpy(),
        "cluster_nombre": cl["cluster_nombre"].to_numpy(),
        "poblacion_total": sub["poblacion_total"].to_numpy(),
        "n_m25_2025": sub["n_m25"].to_numpy(),
    })


def ejecutar():
    """Corre el componente completo: guarda npz, JSON, curva, priorización y figuras."""
    t0 = time.time()
    print(f"{PREFIJO} cargando dataset, predicciones y clusters...")
    df = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    pred = pd.read_csv(config.PREDICCIONES_CSV, dtype={"cod_mpio": str})
    clusters = pd.read_csv(config.CLUSTERS_CSV, dtype={"cod_mpio": str})

    anio_final = config.ANIO_FIN
    anios_estado = list(range(config.TRAIN_RL[0], anio_final + 1))
    estados, recompensas, nombres, extra = construir_estados(df, pred, clusters, anios_estado)
    anios_train = list(range(config.TRAIN_RL[0], config.TRAIN_RL[1] + 1))
    anios_val = list(range(config.VAL_RL[0], config.VAL_RL[1] + 1))
    anios_test = list(range(config.TEST_RL[0], config.TEST_RL[1] + 1))
    k = config.K_PRESUPUESTO
    n_feat = len(nombres)
    print(f"{PREFIJO} estados: {len(estados)} años x {len(extra['cod_mpio'])} municipios x "
          f"{n_feat} variables (k_clusters={extra['k_clusters']})")
    print(f"{PREFIJO} features: {', '.join(nombres)}")

    env = PriorizacionEnv(estados, recompensas, anios=anios_train, k=k)
    agentes, curvas, res_agente = [], [], []
    pesos = []
    for semilla in config.SEMILLAS_RL:
        agente = AgentePoliticaLineal(n_feat, config.RL_LR, config.RL_ENTROPIA, seed=semilla)
        rng = np.random.default_rng(1000 + semilla)
        curva = entrenar(env, agente, config.RL_EPOCAS, anios_train, anios_val, rng)
        curva.insert(0, "semilla", semilla)
        curvas.append(curva)
        res = evaluar_politica(env, lambda x, m, _a, ag=agente: ag.actuar(x, m, greedy=True), anios_test)
        res_agente.append(res)
        agentes.append(agente)
        pesos.append(agente.w.copy())
        print(f"{PREFIJO} semilla {semilla}: %oraculo val final={curva['pct_oraculo_val'].dropna().iloc[-1]:.3f} "
              f"| TEST %oraculo={res['pct_oraculo']:.3f} | t={time.time() - t0:.0f}s")

    curva_df = pd.concat(curvas, ignore_index=True)
    pesos = np.vstack(pesos)

    # Líneas base y oráculo en TEST.
    bases = lineas_base(nombres, np.random.default_rng(config.SEMILLA))
    test = {}
    pct_agente = np.array([r["pct_oraculo"] for r in res_agente])
    test["agente_rl"] = {
        "recompensa_media": _f(np.mean([r["recompensa_media"] for r in res_agente])),
        "std_semillas": _f(pct_agente.std()),
        "pct_oraculo": _f(pct_agente.mean()),
        "precision_at_k": _f(np.mean([r["precision_at_k"] for r in res_agente])),
    }
    rep_azar = [evaluar_politica(env, bases["azar"], anios_test) for _ in range(REPETICIONES_AZAR)]
    test["azar"] = {
        "recompensa_media": _f(np.mean([r["recompensa_media"] for r in rep_azar])),
        "pct_oraculo": _f(np.mean([r["pct_oraculo"] for r in rep_azar])),
        "precision_at_k": _f(np.mean([r["precision_at_k"] for r in rep_azar])),
    }
    for nombre, fn in bases.items():
        if nombre == "azar":
            continue
        r = evaluar_politica(env, fn, anios_test)
        test[nombre] = {key: _f(v) for key, v in r.items()}
    oraculo = evaluar_politica(env, politica_oraculo(recompensas), anios_test)
    test["oraculo"] = {"recompensa_media": _f(oraculo["recompensa_media"])}
    if abs(oraculo["pct_oraculo"] - 1.0) > 1e-9:
        raise RuntimeError(f"el oráculo no da 100 %: {oraculo['pct_oraculo']}")

    for nombre in ["agente_rl", "azar", "mayor_poblacion", "mayor_actividad_pasada",
                   "ranking_supervisado", "indice_pesos_iguales"]:
        print(f"{PREFIJO} TEST {nombre:<24} recompensa={test[nombre]['recompensa_media']:.4f} "
              f"%oraculo={test[nombre]['pct_oraculo']:.4f} prec@K={test[nombre]['precision_at_k']:.3f}")

    # Pesos de la política (escala estandarizada).
    pesos_media, pesos_std = pesos.mean(axis=0), pesos.std(axis=0)
    pesos_politica = {
        nombre: {"media": _f(pesos_media[i]), "std": _f(pesos_std[i])}
        for i, nombre in enumerate(nombres)
    }

    # Priorización 2026 (estado de 2025).
    x_2025 = estados[anio_final]
    info = _info_mpios(df, pred, clusters, extra["cod_mpio"], anio_final)
    priorizacion = priorizar(agentes, x_2025, info, nombres)
    priorizacion.to_csv(config.PRIORIZACION_CSV, index=False)

    # Artefactos.
    config.MODELS.mkdir(parents=True, exist_ok=True)
    with open(config.MODELS / "agente_rl.npz", "wb") as f:
        np.savez(f, pesos=pesos, features=np.array(nombres), media=extra["media"], std=extra["std"])
    curva_df.to_csv(config.REPORTS / "curva_aprendizaje_rl.csv", index=False)

    metricas = {
        "config": {
            "k": k,
            "lr": config.RL_LR,
            "epocas": config.RL_EPOCAS,
            "semillas": len(config.SEMILLAS_RL),
            "anios_train": anios_train,
            "anios_test": anios_test,
        },
        "test": test,
        "pesos_politica": pesos_politica,
        "curva_aprendizaje": "reports/curva_aprendizaje_rl.csv",
    }
    with open(config.REPORTS / "metricas_rl.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    _figuras(curva_df, test, pesos_media, pesos_std, nombres, priorizacion)

    tiempo = time.time() - t0
    print(f"{PREFIJO} artefactos escritos en {config.MODELS}, {config.REPORTS} y {config.PRIORIZACION_CSV}")
    return {
        "metricas": metricas,
        "priorizacion": priorizacion,
        "curva": curva_df,
        "pesos": pesos,
        "nombres": nombres,
        "tiempo_s": tiempo,
    }
