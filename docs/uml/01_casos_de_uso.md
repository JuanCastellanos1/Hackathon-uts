# 01 · Diagrama de casos de uso

Actores:
- **Equipo analista:** construye y entrena.
- **Jurado:** verifica y evalúa.
- **Tomador de decisiones:** Gobernación de Santander, CDGRD o alcaldías, que usan la priorización.
- **Sistemas externos:** fuentes oficiales.

```mermaid
flowchart LR
    analista([👤 Equipo analista])
    jurado([👤 Jurado])
    decisor([👤 Tomador de decisiones<br/>Gobernación / CDGRD])
    sgc[[🛰️ SGC · Catálogo RSNC]]
    dane[[📊 DANE · DIVIPOLA + Población]]
    ungrd[[🚨 UNGRD · Emergencias]]

    subgraph SISTEMA["Sistema de priorización sísmica de Santander"]
        UC1(UC1 · Descargar datos oficiales)
        UC2(UC2 · Construir tabla única<br/>municipio × año)
        UC3(UC3 · Entrenar clasificador<br/>supervisado de actividad)
        UC4(UC4 · Agrupar y comparar<br/>territorios sin etiquetas)
        UC5(UC5 · Entrenar agente RL<br/>de priorización)
        UC6(UC6 · Evaluar con validación<br/>temporal e incertidumbre)
        UC7(UC7 · Generar priorización 2026)
        UC8(UC8 · Explorar la demo interactiva)
        UC9(UC9 · Reproducir el pipeline completo)
    end

    analista --> UC1
    analista --> UC2
    analista --> UC3
    analista --> UC4
    analista --> UC5
    analista --> UC9
    jurado --> UC6
    jurado --> UC8
    jurado --> UC9
    decisor --> UC7
    decisor --> UC8

    UC1 --- sgc
    UC1 --- dane
    UC1 --- ungrd
    UC2 -. include .-> UC1
    UC3 -. include .-> UC2
    UC4 -. include .-> UC2
    UC5 -. include .-> UC3
    UC5 -. include .-> UC4
    UC6 -. extend .-> UC3
    UC6 -. extend .-> UC5
    UC7 -. include .-> UC5
    UC8 -. include .-> UC7
```

| Caso de uso | Disparador | Resultado verificable |
|---|---|---|
| UC1 | `python src/01_descarga_sgc.py`, `python src/02_descarga_fuentes_complementarias.py` | `data/raw/*` + `metadata_*.json` (URL, fecha, filtros) |
| UC2 | `python src/03_tabla.py` | `data/processed/dataset_santander.csv` + `docs/diccionario_datos.csv` |
| UC3 | `python src/04_supervisado.py` | `models/supervisado.joblib`, `reports/metricas_supervisado.json` |
| UC4 | `python src/05_no_supervisado.py` | `data/processed/clusters_municipios.csv`, `reports/metricas_no_supervisado.json` |
| UC5 | `python src/06_agente_rl.py` | `models/agente_rl.npz`, `reports/metricas_rl.json` |
| UC6 | incluido en UC3 y UC5 | métricas en el conjunto de prueba 2020–2024 y comparación contra líneas base |
| UC7 | incluido en UC5 | `data/processed/priorizacion_2026.csv` |
| UC8 | `streamlit run app/app.py` | app con 5 pestañas |
| UC9 | `python run_all.py` | regenera todo con semillas fijas |
