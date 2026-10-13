"""Modelo supervisado de actividad sísmica municipal (CONTRATOS §4 y §6).

Diagrama de clases: docs/uml/03_clases.md (módulo `supervisado`). Actividad: docs/uml/05_actividad_y_estados.md §5.2.

Flujo de `ejecutar()`:
  dataset -> dividir_temporal -> seleccionar_modelo (train→val) -> reentreno (train+val) ->
  evaluar en TEST -> líneas base -> bootstrap F1 -> importancias -> walk-forward 2006–2025 ->
  guardado (joblib, JSON, CSV, figuras).

Regla anti-fuga: la fila del año t predice el nivel de t+1. El walk-forward entrena solo con
filas anio <= t−1 (cuyo objetivo ya está observado) y predice anio == t.
"""
import json
from itertools import product

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)

from sismo import config  # noqa: E402

PREFIJO = "[supervisado]"
RUTA_MODELO = config.MODELS / "supervisado.joblib"
RUTA_METRICAS = config.REPORTS / "metricas_supervisado.json"
RUTA_FIG_CONFUSION = config.FIGURES / "sup_matriz_confusion.png"
RUTA_FIG_IMPORTANCIAS = config.FIGURES / "sup_importancias.png"

N_BOOTSTRAP = 1000
N_REPEATS_PERMUTACION = 5
TOP_IMPORTANCIAS = 15
RF = "RandomForestClassifier"
HGB = "HistGradientBoostingClassifier"


# ----------------------------------------------------------------- utilidades internas
def _nivel_persistencia(n_m25):
    """Clase por persistencia: n_m25 de t con los umbrales del contrato (0→bajo, 1–4→medio, ≥5→alto)."""
    n = np.asarray(n_m25)
    return np.where(n >= config.UMBRAL_ALTO, "alto",
                    np.where(n >= config.UMBRAL_MEDIO, "medio", "bajo"))


def _f1(y, p):
    return float(f1_score(y, p, labels=config.CLASES, average="macro", zero_division=0))


def _metricas_clase(y, p):
    return {
        "f1_macro": _f1(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


def _crear_estimador(nombre, params):
    """Construye el estimador a partir del nombre del modelo y sus hiperparámetros."""
    if nombre == RF:
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            random_state=config.SEMILLA, n_jobs=-1, **params,
        )
    if nombre == HGB:
        return HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.SEMILLA, **params,
        )
    raise ValueError(f"Modelo desconocido: {nombre}")


def _rejilla():
    """Candidatos (nombre, hiperparámetros) de la selección de modelo."""
    candidatos = []
    for md, msl in product([None, 8, 14], [1, 3, 8]):
        candidatos.append((RF, {"max_depth": md, "min_samples_leaf": msl}))
    for lr, md in product([0.05, 0.1], [None, 6]):
        candidatos.append((HGB, {"learning_rate": lr, "max_depth": md}))
    return candidatos


def _con_objetivo(df):
    return df[df[config.OBJETIVO].notna()]


def _proba_ordenada(estimador, X):
    """Probabilidades en el orden de config.CLASES; clases ausentes en el entrenamiento quedan en 0."""
    P = estimador.predict_proba(X)
    salida = np.zeros((len(X), len(config.CLASES)))
    for j, clase in enumerate(estimador.classes_):
        salida[:, config.CLASES.index(clase)] = P[:, j]
    return salida


def _filas_entrenamiento_walkforward(df, t):
    """Filas usadas para predecir el año t: anio <= t−1 con objetivo observado."""
    return _con_objetivo(df[df["anio"] <= t - 1])


def _nativo(obj):
    """Convierte recursivamente tipos numpy/pandas a tipos nativos de JSON."""
    if isinstance(obj, dict):
        return {str(k): _nativo(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nativo(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_nativo(v) for v in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------- API pública
def dividir_temporal(df):
    """Separa por año según config.TRAIN, VAL, TEST y el año futuro (2025, sin objetivo)."""
    assert config.TRAIN[1] < config.VAL[0] and config.VAL[1] < config.TEST[0], \
        "Los rangos TRAIN/VAL/TEST se solapan"
    anio = df["anio"]

    def rango(lo_hi):
        lo, hi = lo_hi
        return df[(anio >= lo) & (anio <= hi)].copy()

    return {
        "train": rango(config.TRAIN),
        "val": rango(config.VAL),
        "test": rango(config.TEST),
        "futuro": df[anio == config.ANIO_FIN].copy(),
    }


def lineas_base(df_test):
    """Líneas base en TEST: clase mayoritaria de TRAIN y persistencia (n_m25 de t).

    Nota: recibe la tabla completa (o al menos TRAIN+TEST). La mayoritaria se calcula con las
    filas de TRAIN y la evaluación con las de TEST; sin filas de TRAIN no hay clase mayoritaria.
    """
    partes = dividir_temporal(df_test)
    train, test = partes["train"], _con_objetivo(partes["test"])
    if train.empty:
        raise ValueError("lineas_base necesita las filas de TRAIN para la clase mayoritaria")

    clase_mayoritaria = train[config.OBJETIVO].value_counts().idxmax()
    y = test[config.OBJETIVO].to_numpy()
    p_may = np.full(len(y), clase_mayoritaria, dtype=object)
    p_per = _nivel_persistencia(test["n_m25"].to_numpy())

    mayoritaria = {"clase": str(clase_mayoritaria), **_metricas_clase(y, p_may)}
    persistencia = _metricas_clase(y, p_per)
    return {"mayoritaria": mayoritaria, "persistencia": persistencia}


def seleccionar_modelo(train, val):
    """Ajusta cada candidato en TRAIN y elige el de mayor F1-macro en VAL.

    Devuelve (estimador ajustado en TRAIN, {modelo, params, f1_macro_val, resultados}).
    """
    train, val = _con_objetivo(train), _con_objetivo(val)
    X_tr, y_tr = train[config.FEATURES_SUP], train[config.OBJETIVO]
    X_va, y_va = val[config.FEATURES_SUP], val[config.OBJETIVO]

    resultados = []
    mejor = None
    for nombre, params in _rejilla():
        estimador = _crear_estimador(nombre, params)
        estimador.fit(X_tr, y_tr)
        f1 = _f1(y_va, estimador.predict(X_va))
        resultados.append({"modelo": nombre, "params": params, "f1_macro_val": f1})
        if mejor is None or f1 > mejor["f1_macro_val"]:
            mejor = {"modelo": nombre, "params": params, "f1_macro_val": f1,
                     "estimador": estimador}

    estimador = mejor.pop("estimador")
    info = {**mejor, "resultados": resultados}
    return estimador, info


def evaluar(modelo, df):
    """Métricas de clasificación sobre las filas con objetivo observado de `df`."""
    df = _con_objetivo(df)
    y = df[config.OBJETIVO].to_numpy()
    p = modelo.predict(df[config.FEATURES_SUP])
    recall = recall_score(y, p, labels=config.CLASES, average=None, zero_division=0)
    return {
        "f1_macro": _f1(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "recall_alto": float(recall[config.CLASES.index("alto")]),
        "matriz_confusion": confusion_matrix(y, p, labels=config.CLASES).tolist(),
        "reporte": classification_report(y, p, labels=config.CLASES, output_dict=True,
                                         zero_division=0),
    }


def importancias(modelo, df):
    """Importancia por permutación (caída de F1-macro) ordenada de mayor a menor."""
    df = _con_objetivo(df)
    r = permutation_importance(
        modelo, df[config.FEATURES_SUP], df[config.OBJETIVO],
        scoring="f1_macro", n_repeats=N_REPEATS_PERMUTACION, random_state=config.SEMILLA,
    )
    return (pd.DataFrame({"feature": config.FEATURES_SUP, "importancia": r.importances_mean})
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True))


def predecir_walkforward(df, params):
    """Predicción walk-forward para t en WALKFORWARD_INICIO..ANIO_FIN.

    `params` es el dict de seleccionar_modelo: {"modelo": str, "params": dict}.
    Para cada t entrena con anio <= t−1 (objetivo no nulo) y predice anio == t.
    """
    filas = []
    for t in range(config.WALKFORWARD_INICIO, config.ANIO_FIN + 1):
        entreno = _filas_entrenamiento_walkforward(df, t)
        assert entreno["anio"].max() < t, "Fuga temporal en walk-forward"
        prueba = df[df["anio"] == t]

        estimador = _crear_estimador(params["modelo"], params["params"])
        estimador.fit(entreno[config.FEATURES_SUP], entreno[config.OBJETIVO])
        proba = _proba_ordenada(estimador, prueba[config.FEATURES_SUP])

        bloque = pd.DataFrame({
            "cod_mpio": prueba["cod_mpio"].to_numpy(),
            "anio": prueba["anio"].to_numpy(),
            "p_bajo": proba[:, 0],
            "p_medio": proba[:, 1],
            "p_alto": proba[:, 2],
            "pred": np.array(config.CLASES, dtype=object)[proba.argmax(axis=1)],
            "y_true": prueba[config.OBJETIVO].to_numpy(),
        })
        filas.append(bloque)
    return pd.concat(filas, ignore_index=True)


def _distribucion(df):
    if df.empty:
        return {c: 0.0 for c in config.CLASES}
    return {k: float(v) for k, v in
            _con_objetivo(df)[config.OBJETIVO].value_counts(normalize=True)
            .reindex(config.CLASES, fill_value=0).items()}


def _graficar_confusion(matriz, ruta):
    cm = np.asarray(matriz, dtype=float)
    sumas = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, sumas, out=np.zeros_like(cm), where=sumas > 0)

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=120)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(config.CLASES)):
        for j in range(len(config.CLASES)):
            ax.text(j, i, f"{norm[i, j]:.2f}\n(n={int(cm[i, j])})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(config.CLASES)), config.CLASES)
    ax.set_yticks(range(len(config.CLASES)), config.CLASES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real (t+1)")
    ax.set_title("Matriz de confusión en TEST (2020–2024)\nnormalizada por fila")
    fig.colorbar(im, ax=ax, label="Proporción de la clase real")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def _graficar_importancias(imp, ruta):
    top = imp.head(TOP_IMPORTANCIAS).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)
    ax.barh(top["feature"], top["importancia"], color="#2b6cb0")
    ax.set_xlabel("Caída de F1-macro al permutar la variable")
    ax.set_title(f"Top {TOP_IMPORTANCIAS} importancias por permutación (TEST)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def ejecutar():
    """Orquesta el pipeline supervisado, guarda artefactos y devuelve el dict de métricas."""
    df = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    partes = dividir_temporal(df)

    _, info = seleccionar_modelo(partes["train"], partes["val"])
    print(f"{PREFIJO} Selección (TRAIN→VAL):")
    for r in info["resultados"]:
        print(f"{PREFIJO}   {r['modelo']:<30} {str(r['params']):<55} F1 val={r['f1_macro_val']:.4f}")
    print(f"{PREFIJO} Elegido: {info['modelo']} {info['params']} (F1 val={info['f1_macro_val']:.4f})")

    trainval = _con_objetivo(
        pd.concat([partes["train"], partes["val"]], ignore_index=True))
    final = _crear_estimador(info["modelo"], info["params"])
    final.fit(trainval[config.FEATURES_SUP], trainval[config.OBJETIVO])

    test_metricas = evaluar(final, partes["test"])
    bases = lineas_base(df)

    test = _con_objetivo(partes["test"])
    y_test = test[config.OBJETIVO].to_numpy()
    p_test = final.predict(test[config.FEATURES_SUP])
    rng = np.random.default_rng(config.SEMILLA)
    n = len(y_test)
    f1_boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        f1_boot[b] = _f1(y_test[idx], p_test[idx])
    bootstrap = {
        "media": float(f1_boot.mean()),
        "p2_5": float(np.percentile(f1_boot, 2.5)),
        "p97_5": float(np.percentile(f1_boot, 97.5)),
    }

    imp = importancias(final, partes["test"])
    wf = predecir_walkforward(df, info)

    RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "features": list(config.FEATURES_SUP)}, RUTA_MODELO)

    metricas = {
        "modelo_elegido": info["modelo"],
        "params": info["params"],
        "val": {"f1_macro": info["f1_macro_val"]},
        "test": test_metricas,
        "lineas_base_test": bases,
        "bootstrap_f1_test": bootstrap,
        "importancias": {r["feature"]: r["importancia"] for r in imp.to_dict("records")},
        "distribucion_clases": {
            "train": _distribucion(partes["train"]),
            "test": _distribucion(partes["test"]),
        },
    }
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    RUTA_METRICAS.write_text(json.dumps(_nativo(metricas), indent=2, ensure_ascii=False),
                             encoding="utf-8")

    wf.to_csv(config.PREDICCIONES_CSV, index=False)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    _graficar_confusion(test_metricas["matriz_confusion"], RUTA_FIG_CONFUSION)
    _graficar_importancias(imp, RUTA_FIG_IMPORTANCIAS)

    print(f"{PREFIJO} Artefactos guardados: {RUTA_MODELO.name}, {RUTA_METRICAS.name}, "
          f"{config.PREDICCIONES_CSV.name}, figuras sup_*.png")
    return metricas
