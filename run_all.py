"""Reproduce todo el pipeline de la hackathon: de los datos crudos a los reportes.

Uso (desde la raíz del proyecto):
    python run_all.py              # pasos 03 → 07 (datos crudos ya descargados)
    python run_all.py --descargar  # además, 01 y 02 (descarga desde las fuentes oficiales)

Funciona igual en Windows, Linux y macOS (en Windows: .venv\Scripts\activate).

Cada paso se ejecuta como un proceso aparte con el mismo intérprete de Python que lanza
este script. Si un paso falla, el resumen muestra el fallo y el script termina con código 1.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

PASOS_DESCARGA = [
    "src/01_descarga_sgc.py",
    "src/02_descarga_fuentes_complementarias.py",
]
PASOS_PIPELINE = [
    "src/03_tabla.py",
    "src/04_supervisado.py",
    "src/05_no_supervisado.py",
    "src/06_agente_rl.py",
    "src/07_presentacion.py",  # opcional: solo se ejecuta si el archivo existe
]


def ejecutar_paso(ruta_rel):
    """Ejecuta un script del proyecto. Devuelve (ok, segundos)."""
    ruta = RAIZ / ruta_rel
    print(f"\n{'=' * 70}\n[run_all] paso: {ruta_rel}\n{'=' * 70}", flush=True)
    inicio = time.perf_counter()
    try:
        entorno = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}  # Windows
        subprocess.run([sys.executable, str(ruta)], check=True, cwd=str(RAIZ), env=entorno)
        ok = True
    except subprocess.CalledProcessError as error:
        print(f"[run_all] ERROR en {ruta_rel} (código {error.returncode})", flush=True)
        ok = False
    return ok, time.perf_counter() - inicio


def main():
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline completo.")
    parser.add_argument("--descargar", action="store_true",
                        help="ejecuta también la descarga de datos (pasos 01 y 02)")
    args = parser.parse_args()

    pasos = (PASOS_DESCARGA if args.descargar else []) + PASOS_PIPELINE
    resultados = []  # (paso, segundos, ok)
    inicio_total = time.perf_counter()

    for paso in pasos:
        if not (RAIZ / paso).exists():
            print(f"[run_all] omitido (no existe): {paso}", flush=True)
            continue
        ok, segundos = ejecutar_paso(paso)
        resultados.append((paso, segundos, ok))
        if not ok:
            break

    total = time.perf_counter() - inicio_total
    print(f"\n{'=' * 70}\n[run_all] RESUMEN\n{'=' * 70}")
    print(f"{'paso':<40}{'segundos':>12}{'OK':>6}")
    for paso, segundos, ok in resultados:
        print(f"{paso:<40}{segundos:>12.1f}{'OK' if ok else 'FALLO':>6}")
    print(f"{'total':<40}{total:>12.1f}")

    if not all(ok for _, _, ok in resultados):
        sys.exit(1)


if __name__ == "__main__":
    main()
