"""Paso 04: modelo supervisado de actividad sísmica (CONTRATOS §6).

Ejecutar desde la raíz del proyecto: .venv/bin/python src/04_supervisado.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sismo import supervisado  # noqa: E402

PREFIJO = "[supervisado]"


def main():
    m = supervisado.ejecutar()
    test, val, bases, boot = m["test"], m["val"], m["lineas_base_test"], m["bootstrap_f1_test"]
    may, per = bases["mayoritaria"], bases["persistencia"]

    print(f"{PREFIJO} ==== Resumen ====")
    print(f"{PREFIJO} Modelo elegido : {m['modelo_elegido']} {m['params']}")
    print(f"{PREFIJO} F1-macro VAL   : {val['f1_macro']:.4f}")
    print(f"{PREFIJO} {'métrica':<18}{'modelo':>10}{'mayoritaria':>14}{'persistencia':>14}")
    print(f"{PREFIJO} {'F1-macro TEST':<18}{test['f1_macro']:>10.4f}"
          f"{may['f1_macro']:>14.4f}{per['f1_macro']:>14.4f}")
    print(f"{PREFIJO} {'Balanced acc TEST':<18}{test['balanced_accuracy']:>10.4f}"
          f"{may['balanced_accuracy']:>14.4f}{per['balanced_accuracy']:>14.4f}")
    print(f"{PREFIJO} Recall alto TEST: {test['recall_alto']:.4f}")
    print(f"{PREFIJO} IC 95 % F1-macro TEST (bootstrap): "
          f"[{boot['p2_5']:.4f}, {boot['p97_5']:.4f}] (media {boot['media']:.4f})")
    print(f"{PREFIJO} Clase mayoritaria (TRAIN): {may['clase']}")
    sup_persistencia = test["f1_macro"] - per["f1_macro"]
    veredicto = "SUPERA" if sup_persistencia > 0 else "NO supera"
    print(f"{PREFIJO} Modelo {veredicto} a la persistencia en F1-macro "
          f"(diferencia {sup_persistencia:+.4f})")
    print(f"{PREFIJO} Top-5 importancias (permutación, TEST):")
    top5 = sorted(m["importancias"].items(), key=lambda kv: kv[1], reverse=True)[:5]
    for feat, valor in top5:
        print(f"{PREFIJO}   {feat:<22}{valor:.4f}")


if __name__ == "__main__":
    main()
