"""Paso 07: genera la presentación del pitch (docs/presentacion.pptx).

Ejecutar desde la raíz del proyecto: .venv/bin/python src/07_presentacion.py
Todas las cifras y textos con números se leen en tiempo de ejecución de
reports/*.json, data/processed/*.csv y data/raw/metadata_descarga_sgc.json.
No hay cifras escritas a mano en este script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from pptx import Presentation  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402

from sismo import config  # noqa: E402

PREFIJO = "[presentacion]"

# ----------------------------------------------------------------- diseño
AZUL = RGBColor(0x1F, 0x4E, 0x79)
NARANJA = RGBColor(0xE0, 0x7A, 0x1F)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)
TEXTO = RGBColor(0x26, 0x26, 0x26)
GRIS_FONDO = RGBColor(0xF2, 0xF2, 0xF2)
GRIS_LINEA = RGBColor(0x7F, 0x7F, 0x7F)
FUENTE = "Calibri"
ANCHO, ALTO = 13.333, 7.5          # 16:9 en pulgadas
MARGEN = 0.5
ALTURA_BARRA = 1.1
TAM_TITULO = 30                    # título dentro del rango 28–32 pt
TAM_CUERPO = 18                    # cuerpo ≥ 16 pt
TAM_TABLA = 16
TOTAL_DIAPOSITIVAS = 10


# ----------------------------------------------------------------- formato
def fmt(x, dec=2):
    """Número con coma decimal, estilo español."""
    return f"{x:.{dec}f}".replace(".", ",")


def pct(x, dec=1):
    """Fracción (0–1) a porcentaje con coma decimal."""
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def miles(n):
    """Entero con punto como separador de miles."""
    return f"{int(round(n)):,}".replace(",", ".")


def nombre_propio(nombre):
    """Pasa 'EL CARMEN DE CHUCURÍ' a 'El Carmen de Chucurí' (preposiciones en minúscula)."""
    minusculas = {"de", "del", "la", "las", "los", "y"}
    partes = str(nombre).lower().split()
    return " ".join(p if (i > 0 and p in minusculas) else p.capitalize() for i, p in enumerate(partes))


# ----------------------------------------------------------------- carga de datos
def leer_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def cargar_datos():
    """Lee todos los resultados. Si falta un archivo, el script se detiene con error."""
    sup = leer_json(config.REPORTS / "metricas_supervisado.json")
    nosup = leer_json(config.REPORTS / "metricas_no_supervisado.json")
    rl = leer_json(config.REPORTS / "metricas_rl.json")
    meta = leer_json(config.RAW / "metadata_descarga_sgc.json")

    dataset = pd.read_csv(config.DATASET_CSV, dtype={"cod_mpio": str})
    diccionario = pd.read_csv(config.DICCIONARIO_CSV)
    priorizacion = pd.read_csv(config.PRIORIZACION_CSV, dtype={"cod_mpio": str}).sort_values("rango")
    clusters = pd.read_csv(config.CLUSTERS_CSV, dtype={"cod_mpio": str})

    # es_santander puede venir como 0/1 o como True/False según el CSV.
    es_sant_ds = dataset["es_santander"].map(lambda v: str(v).lower() in ("1", "true", "1.0"))
    es_sant_cl = clusters["es_santander"].map(lambda v: str(v).lower() in ("1", "true", "1.0"))

    # Sismos M >= 2,5 por mes en Santander durante el periodo de prueba (promedio mensual).
    ini, fin = config.TEST
    en_prueba = dataset["anio"].between(ini, fin) & es_sant_ds
    meses = (fin - ini + 1) * 12
    sismos_mes = dataset.loc[en_prueba, "n_m25"].sum() / meses

    eventos_total = meta["eventos_1993_2018"] + meta["eventos_2018_hoy"]
    return dict(
        sup=sup, nosup=nosup, rl=rl, meta=meta,
        dataset=dataset, diccionario=diccionario, priorizacion=priorizacion,
        clusters=clusters, es_sant_cl=es_sant_cl,
        n_filas=len(dataset), n_vars=len(diccionario),
        n_mpios=dataset["cod_mpio"].nunique(),
        n_mpios_sant=int(es_sant_cl.sum()),
        n_anios=(dataset["anio"].min(), dataset["anio"].max()),
        sismos_mes=sismos_mes, eventos_total=eventos_total,
    )


# ----------------------------------------------------------------- utilidades de diseño
def estilo(run, tam, negrita=False, color=TEXTO):
    run.font.name = FUENTE
    run.font.size = Pt(tam)
    run.font.bold = negrita
    run.font.color.rgb = color


def cuadro_texto(slide, x, y, w, h, parrafos, tam=TAM_CUERPO, color=TEXTO,
                 negrita=False, alinear=PP_ALIGN.LEFT, anclaje=MSO_ANCHOR.TOP):
    """Cuadro de texto. `parrafos` es una lista de cadenas."""
    caja = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    marco = caja.text_frame
    marco.word_wrap = True
    marco.vertical_anchor = anclaje
    for i, texto in enumerate(parrafos):
        p = marco.paragraphs[0] if i == 0 else marco.add_paragraph()
        p.alignment = alinear
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = texto
        estilo(r, tam, negrita, color)
    return caja


def vinetas(slide, x, y, w, h, lineas, tam=TAM_CUERPO):
    """Viñetas manuales (•) para no depender de la plantilla."""
    return cuadro_texto(slide, x, y, w, h, [f"•  {t}" for t in lineas], tam=tam)


def rectangulo(slide, x, y, w, h, color, texto=None, tam=TAM_CUERPO, color_texto=BLANCO,
               negrita=False, borde=None):
    forma = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    forma.fill.solid()
    forma.fill.fore_color.rgb = color
    if borde is None:
        forma.line.fill.background()
    else:
        forma.line.color.rgb = borde
        forma.line.width = Pt(1.25)
    forma.shadow.inherit = False
    if texto is not None:
        marco = forma.text_frame
        marco.word_wrap = True
        marco.vertical_anchor = MSO_ANCHOR.MIDDLE
        marco.margin_left = marco.margin_right = Inches(0.08)
        for i, linea in enumerate(texto.split("\n")):
            p = marco.paragraphs[0] if i == 0 else marco.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = linea
            estilo(r, tam, negrita if i == 0 else False, color_texto)
    return forma


def flecha(slide, x1, y1, x2, y2):
    """Conector recto con punta de flecha."""
    conector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                          Inches(x2), Inches(y2))
    conector.line.color.rgb = GRIS_LINEA
    conector.line.width = Pt(2)
    ln = conector.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return conector


def imagen_ajustada(slide, nombre, x, y, w, h):
    """Escala la figura manteniendo proporción dentro de la caja (x, y, w, h)."""
    ruta = config.FIGURES / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Falta la figura {ruta}")
    with Image.open(ruta) as im:
        iw, ih = im.size
    escala = min(w / iw, h / ih)
    ancho, alto = iw * escala, ih * escala
    px = x + (w - ancho) / 2
    py = y + (h - alto) / 2
    slide.shapes.add_picture(str(ruta), Inches(px), Inches(py), Inches(ancho), Inches(alto))


def tabla(slide, x, y, w, filas, anchos, encabezado=True, tam=TAM_TABLA, negrita_primera_col=False):
    """Tabla con encabezado azul. `filas` incluye el encabezado si `encabezado` es True."""
    n_f, n_c = len(filas), len(filas[0])
    forma = slide.shapes.add_table(n_f, n_c, Inches(x), Inches(y), Inches(w), Inches(0.4 * n_f))
    t = forma.table
    for j, ancho in enumerate(anchos):
        t.columns[j].width = Inches(ancho)
    for i, fila in enumerate(filas):
        for j, valor in enumerate(fila):
            celda = t.cell(i, j)
            celda.margin_left = celda.margin_right = Inches(0.08)
            celda.margin_top = celda.margin_bottom = Inches(0.04)
            celda.fill.solid()
            es_enc = encabezado and i == 0
            if es_enc:
                celda.fill.fore_color.rgb = AZUL
            else:
                celda.fill.fore_color.rgb = GRIS_FONDO if i % 2 == 0 else BLANCO
            marco = celda.text_frame
            marco.word_wrap = True
            p = marco.paragraphs[0]
            r = p.add_run()
            r.text = str(valor)
            negrita = es_enc or (negrita_primera_col and j == 0)
            estilo(r, tam, negrita, BLANCO if es_enc else TEXTO)
    return forma


def nueva_diapositiva(prs, titulo, numero):
    """Diapositiva en blanco con barra de título azul, filete naranja y número de página."""
    diap = prs.slides.add_slide(prs.slide_layouts[6])
    rectangulo(diap, 0, 0, ANCHO, ALTURA_BARRA, AZUL)
    cuadro_texto(diap, MARGEN, 0, ANCHO - 2 * MARGEN, ALTURA_BARRA, [titulo], tam=TAM_TITULO,
                 color=BLANCO, negrita=True, anclaje=MSO_ANCHOR.MIDDLE)
    rectangulo(diap, 0, ALTURA_BARRA, ANCHO, 0.06, NARANJA)
    cuadro_texto(diap, ANCHO - 1.6, ALTO - 0.5, 1.2, 0.35, [f"{numero} / {TOTAL_DIAPOSITIVAS}"],
                 tam=12, color=GRIS_LINEA, alinear=PP_ALIGN.RIGHT)
    return diap


def notas(diap, texto):
    diap.notes_slide.notes_text_frame.text = texto


# ----------------------------------------------------------------- diapositivas
def diap_portada(prs, d, numero):
    diap = prs.slides.add_slide(prs.slide_layouts[6])
    rectangulo(diap, 0, 0, ANCHO, 3.4, AZUL)
    cuadro_texto(diap, MARGEN + 0.2, 0.7, ANCHO - 1.0, 1.1, ["Santander Inteligente"],
                 tam=32, color=BLANCO, negrita=True, anclaje=MSO_ANCHOR.MIDDLE)
    cuadro_texto(diap, MARGEN + 0.2, 1.8, ANCHO - 1.0, 1.2,
                 ["Priorización exploratoria del riesgo sísmico con aprendizaje por refuerzo"],
                 tam=24, color=BLANCO)
    rectangulo(diap, 0, 3.4, ANCHO, 0.08, NARANJA)
    cuadro_texto(diap, MARGEN + 0.2, 3.9, ANCHO - 1.0, 0.6,
                 ["Hackathon CIAI UTSmart 2026 · Reto Avanzado"], tam=22, color=AZUL, negrita=True)
    cuadro_texto(diap, MARGEN + 0.2, 4.6, ANCHO - 1.0, 0.6, ["Equipo: Santiago Afanador Arguello · Juan Castellanos"], tam=20)
    cuadro_texto(diap, MARGEN + 0.2, 5.4, ANCHO - 1.0, 0.8,
                 [f"Fuentes oficiales: SGC, DANE y UNGRD · {d['n_filas']} municipio-año "
                  f"de {d['n_mpios']} municipios"], tam=18, color=GRIS_LINEA)
    cuadro_texto(diap, ANCHO - 1.6, ALTO - 0.5, 1.2, 0.35, [f"{numero} / {TOTAL_DIAPOSITIVAS}"],
                 tam=12, color=GRIS_LINEA, alinear=PP_ALIGN.RIGHT)
    notas(diap, (
        f"Presentamos Santander Inteligente, un proyecto que prioriza municipios según su riesgo sísmico. "
        f"Usamos datos oficiales del SGC, el DANE y la UNGRD, y un agente de aprendizaje por refuerzo. "
        f"El equipo se presenta al final de la sesión."
    ))
    return diap


def diap_problema(prs, d, numero):
    diap = nueva_diapositiva(prs, "El problema: el Nido sísmico de Bucaramanga", numero)
    vinetas(diap, MARGEN, 1.45, 7.6, 3.6, [
        "Santander registra sismos de forma continua, concentrados en el Nido sísmico de "
        "Bucaramanga (sismicidad intermedia).",
        f"Pregunta de decisión: con presupuesto para {config.K_PRESUPUESTO} estudios o inspecciones "
        "al año, ¿qué municipios priorizar?",
        "Marco conceptual: Riesgo = Amenaza × Exposición × Vulnerabilidad.",
        "Amenaza: actividad sísmica (SGC). Exposición: población (DANE). "
        "Vulnerabilidad: emergencias (UNGRD).",
    ])
    rectangulo(diap, 8.7, 1.8, 4.1, 3.2, GRIS_FONDO, borde=AZUL)
    cuadro_texto(diap, 8.9, 1.95, 3.7, 0.7, ["Sismos M ≥ 2,5 por mes en Santander"],
                 tam=18, color=AZUL, negrita=True, alinear=PP_ALIGN.CENTER)
    cuadro_texto(diap, 8.9, 2.75, 3.7, 1.2, [fmt(d["sismos_mes"], 1)],
                 tam=54, color=NARANJA, negrita=True, alinear=PP_ALIGN.CENTER)
    cuadro_texto(diap, 8.9, 4.0, 3.7, 0.9,
                 [f"Promedio mensual en el periodo de prueba {config.TEST[0]}–{config.TEST[1]}"],
                 tam=16, alinear=PP_ALIGN.CENTER)
    notas(diap, (
        f"Santander concentra una sismicidad constante, sobre todo en el Nido de Bucaramanga. "
        f"En promedio se registran {fmt(d['sismos_mes'], 1)} sismos de magnitud 2,5 o mayor al mes "
        f"en el periodo de prueba. Con presupuesto para {config.K_PRESUPUESTO} estudios al año, la "
        f"pregunta es a qué municipios darles prioridad."
    ))
    return diap


def diap_datos(prs, d, numero):
    diap = nueva_diapositiva(prs, "Datos e integración: cuatro fuentes oficiales", numero)
    vinetas(diap, MARGEN, 1.45, 6.6, 5.2, [
        f"SGC, catálogo RSNC: {miles(d['eventos_total'])} eventos en 2 consultas unificadas "
        f"(1993–2018 y 2018–hoy).",
        f"Tabla única: {miles(d['n_filas'])} filas × {d['n_vars']} variables "
        f"(municipio × año, {d['n_anios'][0]}–{d['n_anios'][1]}, {d['n_mpios']} municipios).",
        "Integración: cada sismo se asigna a la cabecera municipal más cercana.",
        f"Magnitud de completitud: M ≥ {fmt(config.MAG_MIN, 1)} desde {config.ANIO_INICIO}. "
        "El cambio de red de 2018 se modela con la variable periodo_red.",
        f"Calidad: se excluyen eventos atípicos (RMS > {fmt(config.RMS_MAX, 1)} s o error de "
        f"profundidad > {fmt(config.ERR_PROF_MAX, 0)} km).",
    ], tam=TAM_CUERPO)
    filas = [
        ["Fuente", "Aporta"],
        ["SGC · catálogo RSNC", "Eventos: magnitud, profundidad, RMS"],
        ["DANE · DIVIPOLA", "Códigos y coordenadas de cabeceras"],
        ["DANE · población", "Población municipal (2020)"],
        ["UNGRD · emergencias", "Emergencias y afectaciones"],
    ]
    tabla(diap, 7.4, 1.6, 5.4, filas, [2.3, 3.1])
    notas(diap, (
        f"Integramos cuatro fuentes oficiales: el catálogo sísmico del SGC, la división político-administrativa "
        f"del DANE, la proyección de población y las emergencias de la UNGRD. "
        f"El resultado es una tabla única de {miles(d['n_filas'])} filas con {d['n_vars']} variables. "
        f"Cada sismo se asigna a la cabecera municipal más cercana y solo usamos magnitud 2,5 o mayor desde 2002."
    ))
    return diap


def diap_arquitectura(prs, d, numero):
    diap = nueva_diapositiva(prs, "Arquitectura del flujo", numero)
    w, h, sep = 1.8, 0.95, 0.3
    x = [MARGEN + i * (w + sep) for i in range(6)]
    # Columna 0: fuentes oficiales
    fuentes = [("SGC: catálogo sísmico", 1.6), ("DANE: DIVIPOLA y población", 2.7),
               ("UNGRD: emergencias", 3.8)]
    cajas_fuente = []
    for etiqueta, y in fuentes:
        rectangulo(diap, x[0], y, w, h, GRIS_FONDO, texto=etiqueta, tam=16, color_texto=TEXTO,
                   borde=AZUL)
        cajas_fuente.append((y + h / 2))
    # Columna 1: tabla única
    y_tabla, h_tabla = 2.35, 1.9
    rectangulo(diap, x[1], y_tabla, w, h_tabla, AZUL,
               texto=f"Tabla única\n{miles(d['n_filas'])} filas × {d['n_vars']} variables", tam=16,
               negrita=True)
    # Columna 2: dos modelos
    rectangulo(diap, x[2], 1.6, w, 1.45, NARANJA, texto="Supervisado\namenaza en t+1", tam=16,
               negrita=True)
    rectangulo(diap, x[2], 3.95, w, 1.45, NARANJA, texto="No supervisado\ntipologías", tam=16,
               negrita=True)
    # Columna 3: agente RL
    rectangulo(diap, x[3], y_tabla, w, h_tabla, AZUL,
               texto=f"Agente RL\nasigna K = {config.K_PRESUPUESTO}", tam=16, negrita=True)
    # Columna 4 y 5: priorización y demo
    rectangulo(diap, x[4], y_tabla, w, h_tabla, AZUL, texto="Priorización\n2026", tam=16,
               negrita=True)
    rectangulo(diap, x[5], y_tabla, w, h_tabla, AZUL, texto="Demo\napp Streamlit", tam=16,
               negrita=True)
    # Flechas
    for yc in cajas_fuente:
        flecha(diap, x[0] + w, yc, x[1], y_tabla + h_tabla / 2)
    flecha(diap, x[1] + w, y_tabla + h_tabla / 2, x[2], 1.6 + 1.45 / 2)
    flecha(diap, x[1] + w, y_tabla + h_tabla / 2, x[2], 3.95 + 1.45 / 2)
    flecha(diap, x[2] + w, 1.6 + 1.45 / 2, x[3], y_tabla + h_tabla / 2)
    flecha(diap, x[2] + w, 3.95 + 1.45 / 2, x[3], y_tabla + h_tabla / 2)
    flecha(diap, x[3] + w, y_tabla + h_tabla / 2, x[4], y_tabla + h_tabla / 2)
    flecha(diap, x[4] + w, y_tabla + h_tabla / 2, x[5], y_tabla + h_tabla / 2)
    # Bandas de validación y reproducibilidad
    rectangulo(diap, MARGEN, 5.55, ANCHO - 2 * MARGEN, 0.6, GRIS_FONDO)
    cuadro_texto(diap, MARGEN + 0.15, 5.58, ANCHO - 2 * MARGEN - 0.3, 0.55,
                 [f"Validación temporal: entrenamiento {config.TRAIN[0]}–{config.TRAIN[1]} · "
                  f"validación {config.VAL[0]}–{config.VAL[1]} · prueba {config.TEST[0]}–{config.TEST[1]}"],
                 tam=16, anclaje=MSO_ANCHOR.MIDDLE)
    rectangulo(diap, MARGEN, 6.3, ANCHO - 2 * MARGEN, 0.6, GRIS_FONDO)
    cuadro_texto(diap, MARGEN + 0.15, 6.33, ANCHO - 2 * MARGEN - 0.3, 0.55,
                 ["Reproducible: run_all.py ejecuta src/01 a 07 con semillas fijas; pruebas en tests/"],
                 tam=16, anclaje=MSO_ANCHOR.MIDDLE)
    notas(diap, (
        "El flujo parte de las fuentes oficiales, que se unifican en una tabla de municipio por año. "
        "Sobre esa tabla corren en paralelo un modelo supervisado que estima la amenaza del año siguiente "
        "y un no supervisado que agrupa municipios por tipología. "
        "El agente de refuerzo combina ambos y produce la priorización 2026, que se consulta desde la demo."
    ))
    return diap


def diap_supervisado(prs, d, numero):
    s = d["sup"]
    diap = nueva_diapositiva(prs, "Modelo supervisado: amenaza en t+1", numero)
    f1 = s["test"]["f1_macro"]
    f1_pers = s["lineas_base_test"]["persistencia"]["f1_macro"]
    f1_may = s["lineas_base_test"]["mayoritaria"]["f1_macro"]
    ic_lo, ic_hi = s["bootstrap_f1_test"]["p2_5"], s["bootstrap_f1_test"]["p97_5"]
    n_test = int(s["test"]["reporte"]["macro avg"]["support"])
    vinetas(diap, MARGEN, 1.45, 6.2, 5.2, [
        f"Objetivo: nivel de actividad del año t+1 (bajo, medio o alto). Modelo: {s['modelo_elegido']}.",
        f"F1-macro en prueba: {fmt(f1, 3)} (IC 95 %: {fmt(ic_lo, 3)} a {fmt(ic_hi, 3)}), "
        f"sobre {n_test} municipio-año.",
        f"Líneas base: persistencia {fmt(f1_pers, 3)} y mayoritaria {fmt(f1_may, 3)}.",
        f"Recall de la clase alta: {pct(s['test']['recall_alto'])}.",
        f"Nota honesta: frente a persistencia, la ganancia es de {fmt(f1 - f1_pers, 3)} puntos de F1-macro; "
        "es un margen modesto y así lo reportamos.",
    ], tam=TAM_CUERPO)
    imagen_ajustada(diap, "sup_matriz_confusion.png", 6.9, 1.45, 5.95, 5.5)
    notas(diap, (
        f"El modelo supervisado alcanza un F1-macro de {fmt(f1, 3)} en el periodo de prueba, "
        f"con un intervalo de confianza entre {fmt(ic_lo, 3)} y {fmt(ic_hi, 3)}. "
        f"Frente a la mayoritaria la mejora es grande, pero frente a la persistencia el margen es de "
        f"{fmt(f1 - f1_pers, 3)} puntos. Lo presentamos así para no sobrevender el modelo."
    ))
    return diap


def diap_no_supervisado(prs, d, numero):
    u = d["nosup"]
    diap = nueva_diapositiva(prs, "No supervisado: tipologías sísmicas por municipio", numero)
    ks = [f["k"] for f in u["seleccion_k"]]
    conteo = d["clusters"]["cluster_nombre"].value_counts()
    conteo_txt = " · ".join(f"{n} {nombre}" for nombre, n in conteo.items())
    comp = u["comparacion_territorial"]
    santander, vecinos = comp["SANTANDER"], comp["VECINOS (otros dptos)"]
    nido = "Nido / sismicidad intermedia"
    activa = "Intermedia activa (periferia del Nido)"
    dbs = max(u["dbscan_fuentes"], key=lambda f: f["n"])
    imagen_ajustada(diap, "nosup_mapa_clusters.png", MARGEN, 1.45, 6.4, 5.5)
    vinetas(diap, 7.1, 1.45, 5.8, 5.4, [
        f"k = {u['k_elegido']} tipologías, elegido por silueta ({fmt(u['silueta_final'], 3)}) "
        f"entre k = {min(ks)} y {max(ks)}.",
        f"Municipios por tipología: {conteo_txt}.",
        f"Santander: {santander[nido]} municipios en el cluster Nido y {santander[activa]} en "
        f"intermedia activa; los vecinos suman {vecinos[nido]} y {vecinos[activa]}.",
        f"DBSCAN, sin etiquetas (solo lat, lon y profundidad), encuentra el Nido: {miles(dbs['n'])} eventos "
        f"centrados en {fmt(dbs['lat'], 2)}, {fmt(dbs['lon'], 2)}, con profundidad media de "
        f"{fmt(dbs['prof_media_km'], 0)} km.",
    ], tam=TAM_CUERPO)
    notas(diap, (
        f"El no supervisado agrupa los municipios en {u['k_elegido']} tipologías, con una silueta de "
        f"{fmt(u['silueta_final'], 3)}. Santander concentra la mayoría de la sismicidad intermedia activa y el "
        f"cluster del Nido. Sin etiquetas, DBSCAN recupera por sí solo el Nido de Bucaramanga."
    ))
    return diap


def diap_rl_formulacion(prs, d, numero):
    rl_cfg = d["rl"]["config"]
    diap = nueva_diapositiva(prs, "Agente RL: formulación del problema", numero)
    filas = [
        ["Componente", "Definición"],
        ["Entorno", f"PriorizacionSantanderEnv (Gymnasium). Cada episodio es un año; candidatos: "
                    f"los {d['n_mpios_sant']} municipios de Santander."],
        ["Estado", "Por candidato: P(alto) y P(medio) del supervisado, cluster, población, % rural, "
                   "emergencias, movimientos en masa, viviendas, distancia al Nido y bandera de ya elegido."],
        ["Acción", f"Elegir un municipio aún no elegido. El episodio termina tras K = {rl_cfg['k']} elecciones."],
        ["Recompensa", f"Riesgo realizado en t+1 = amenaza (energía sísmica M ≥ {fmt(config.MAG_MIN, 1)}, "
                       "normalizada) × (0,5 exposición + 0,5 vulnerabilidad). El agente no ve t+1."],
        ["Algoritmo", f"REINFORCE con línea base; política softmax lineal en numpy; "
                      f"{rl_cfg['semillas']} semillas."],
        ["Periodos", f"Entrenamiento {config.TRAIN_RL[0]}–{config.TRAIN_RL[1]} · validación "
                     f"{config.VAL_RL[0]}–{config.VAL_RL[1]} · prueba {config.TEST_RL[0]}–{config.TEST_RL[1]}."],
    ]
    tabla(diap, MARGEN, 1.5, ANCHO - 2 * MARGEN, filas, [2.6, ANCHO - 2 * MARGEN - 2.6],
          negrita_primera_col=True)
    notas(diap, (
        f"El agente elige, cada año, {rl_cfg['k']} municipios de Santander con un presupuesto fijo. "
        f"Recibe como recompensa el riesgo realizado un año después, que combina amenaza, población y vulnerabilidad. "
        f"Aprende con REINFORCE entrenando de {config.TRAIN_RL[0]} a {config.TRAIN_RL[1]} y lo evaluamos en años que no vio."
    ))
    return diap


def diap_rl_resultados(prs, d, numero):
    rl = d["rl"]["test"]
    diap = nueva_diapositiva(prs, "Agente RL: resultados frente a las líneas base", numero)
    nombres = {
        "indice_pesos_iguales": "índice de pesos iguales", "mayor_poblacion": "mayor población",
        "mayor_actividad_pasada": "mayor actividad pasada", "ranking_supervisado": "ranking supervisado",
        "azar": "azar",
    }
    bases = {k: v for k, v in rl.items() if k not in ("agente_rl", "oraculo")}
    mejor = max(bases, key=lambda k: bases[k]["pct_oraculo"])
    ag = rl["agente_rl"]
    imagen_ajustada(diap, "rl_comparacion_lineas_base.png", MARGEN, 1.4, 6.1, 4.7)
    imagen_ajustada(diap, "rl_curva_aprendizaje.png", 6.9, 1.4, 5.95, 4.7)
    cuadro_texto(diap, MARGEN, 6.15, ANCHO - 2 * MARGEN, 0.9, [
        f"El agente alcanza el {pct(ag['pct_oraculo'])} del óptimo (oráculo) frente al "
        f"{pct(bases[mejor]['pct_oraculo'])} de la mejor línea base ({nombres.get(mejor, mejor)}). "
        f"Desviación entre semillas: {fmt(ag['std_semillas'] * 100, 1)} p.p.",
    ], tam=TAM_CUERPO, negrita=True, color=AZUL)
    notas(diap, (
        f"En el periodo de prueba, el agente alcanza el {pct(ag['pct_oraculo'])} del óptimo, "
        f"y la mejor línea base llega al {pct(bases[mejor]['pct_oraculo'])}. "
        f"La curva de aprendizaje muestra la recompensa de entrenamiento a lo largo de las épocas. "
        f"La variación entre semillas es pequeña, con una desviación de {fmt(ag['std_semillas'] * 100, 1)} puntos."
    ))
    return diap


def diap_priorizacion(prs, d, numero):
    p = d["priorizacion"].head(10)
    diap = nueva_diapositiva(prs, "Priorización 2026: mapa y top-10", numero)
    imagen_ajustada(diap, "rl_mapa_priorizacion.png", MARGEN, 1.4, 6.6, 5.0)
    filas = [["Rango", "Municipio", "Confianza"]]
    for _, fila in p.iterrows():
        filas.append([int(fila["rango"]), nombre_propio(fila["nom_mpio"]), str(fila["confianza"]).capitalize()])
    tabla(diap, 7.4, 1.45, 5.4, filas, [1.0, 3.0, 1.4], tam=TAM_TABLA)
    n_alta = int((p["confianza"] == "alta").sum())
    cuadro_texto(diap, MARGEN, 6.45, ANCHO - 2 * MARGEN, 0.7, [
        f"Exploratoria: aplica la política al estado de 2025. {n_alta} de los 10 primeros tienen confianza "
        "alta, según la frecuencia con que entran al top-K.",
    ], tam=16, color=TEXTO)
    notas(diap, (
        f"Aplicamos la política aprendida al estado de 2025 para obtener la priorización 2026. "
        f"El primer lugar es {nombre_propio(p.iloc[0]['nom_mpio'])}, y {n_alta} de los diez primeros tienen confianza alta. "
        "Es una herramienta exploratoria que orienta estudios, no predice sismos."
    ))
    return diap


def diap_cierre(prs, d, numero):
    diap = nueva_diapositiva(prs, "Impacto, limitaciones y extensiones", numero)
    anchos = (ANCHO - 2 * MARGEN - 0.6) / 3
    columnas = [
        ("Impacto", AZUL, [
            "Orientar la microzonificación sísmica de la Gobernación y la CDGRD.",
            "Priorizar inspecciones estructurales en los municipios de mayor confianza.",
            "Focalizar simulacros y planes de respuesta.",
        ]),
        ("Limitaciones", NARANJA, [
            "El riesgo realizado es un proxy: no hay daños por sismo con resolución suficiente.",
            "Exposición (2020) y vulnerabilidad (2019–2025) son estáticas.",
            "Cada sismo se asigna a la cabecera más cercana, no a un polígono oficial.",
        ]),
        ("Extensiones", AZUL, [
            "Polígonos municipales del IGAC.",
            "Amenaza según NSR-10 (Aa/Av).",
            "Sismicidad histórica con daños del SISH.",
            "Costo por inspección dentro de la recompensa.",
        ]),
    ]
    for i, (titulo, color, lineas) in enumerate(columnas):
        x = MARGEN + i * (anchos + 0.3)
        rectangulo(diap, x, 1.45, anchos, 0.6, color, texto=titulo, tam=20, negrita=True)
        rectangulo(diap, x, 2.05, anchos, 4.0, GRIS_FONDO)
        vinetas(diap, x + 0.12, 2.2, anchos - 0.24, 3.8, lineas, tam=TAM_CUERPO)
    cuadro_texto(diap, MARGEN, 6.2, ANCHO - 2 * MARGEN, 0.7, [
        "La priorización es exploratoria: orienta estudios y no predice sismos.",
    ], tam=18, color=NARANJA, negrita=True)
    notas(diap, (
        "El valor del proyecto está en orientar dónde invertir estudios e inspecciones con datos oficiales. "
        "Sus limitaciones principales son que el riesgo realizado es un proxy y que la exposición y la vulnerabilidad son estáticas. "
        "Las extensiones naturales son usar polígonos oficiales, la norma NSR-10 y los daños históricos."
    ))
    return diap


# ----------------------------------------------------------------- principal
def main():
    d = cargar_datos()
    prs = Presentation()
    prs.slide_width = Inches(ANCHO)
    prs.slide_height = Inches(ALTO)

    constructores = [
        diap_portada, diap_problema, diap_datos, diap_arquitectura, diap_supervisado,
        diap_no_supervisado, diap_rl_formulacion, diap_rl_resultados, diap_priorizacion, diap_cierre,
    ]
    for numero, construir in enumerate(constructores, start=1):
        construir(prs, d, numero)

    n = len(prs.slides)
    if n != TOTAL_DIAPOSITIVAS:
        raise RuntimeError(f"Se generaron {n} diapositivas y se esperaban {TOTAL_DIAPOSITIVAS}")

    ruta = config.DOCS / "presentacion.pptx"
    prs.save(str(ruta))
    print(f"{PREFIJO} {n} diapositivas guardadas en {ruta}")


if __name__ == "__main__":
    main()
