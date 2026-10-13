# 05 · Diagramas de actividad y de estados

## 5.1 Actividad: limpieza y construcción de la tabla única (`03_tabla.py`)

```mermaid
flowchart TD
    A([Inicio]) --> B[Leer sgc_1993_2018.csv<br/>Earthworm/SEISAN]
    A --> C[Leer sgc_2018_hoy.csv<br/>SeisComP3]
    B --> D[Renombrar al esquema común<br/>mag = Ml · tipo_mag = 'Ml' · periodo_red = 0]
    C --> E[Renombrar al esquema común<br/>mag = mag · periodo_red = 1]
    D --> F[Concatenar + parsear fecha UTC]
    E --> F
    F --> G{¿Duplicado exacto<br/>fecha+lat+lon+prof+mag?}
    G -- sí --> G1[Eliminar y registrar en el log] --> H
    G -- no --> H{¿RMS > 2 s o<br/>error prof > 50 km?}
    H -- sí --> H1[es_atipico = True<br/>se excluye de las variables] --> I
    H -- no --> I["Asignar el municipio de la cabecera<br/>DANE más cercana (haversine)"]
    I --> J[Guardar interim/eventos_unificados.csv]
    J --> K[Filtrar M ≥ 2,5 y no atípicos]
    K --> L[Agregar por municipio × año<br/>1999–2025, rellenando con 0 sin eventos]
    L --> M[Rezagos: prom. 3 años, tendencia,<br/>máx. histórico, b-value ventana 5 años]
    M --> N["Objetivo: n_m25 de t+1 → bajo/medio/alto<br/>+ log_energia_t1 (auxiliar RL)"]
    N --> O[Unir DANE población + UNGRD emergencias<br/>por cod_mpio]
    O --> P[Recortar años 2002–2025]
    P --> Q{Validaciones:<br/>≥ 20 variables · sin duplicados ·<br/>sin nulos en features · objetivo NaN solo en 2025}
    Q -- falla --> X([Error: abortar con un mensaje claro])
    Q -- ok --> R[Guardar processed/dataset_santander.csv<br/>+ docs/diccionario_datos.csv]
    R --> Z([Fin])
```

## 5.2 Actividad: validación temporal (supervisado)

```mermaid
flowchart LR
    subgraph T1["Selección de hiperparámetros"]
        TR[Train<br/>2002–2015] --> VA[Validación<br/>2016–2019]
    end
    subgraph T2["Evaluación final"]
        TR2[Train<br/>2002–2019] --> TE[Prueba<br/>2020–2024]
    end
    subgraph T3["Walk-forward (estado del agente RL)"]
        WF["Para cada t en 2006..2025:<br/>entrenar con anio ≤ t−1 → predecir anio t"]
    end
    T1 --> T2 --> T3
```

> La fila del año `t` predice la actividad de `t+1`. Al final del año `t`, las filas `anio ≤ t−1` ya tienen su objetivo observado, así que el walk-forward **no usa información futura**.

## 5.3 Estados: episodio del entorno `PriorizacionEnv`

```mermaid
stateDiagram-v2
    [*] --> SinIniciar
    SinIniciar --> Seleccionando: reset(anio)\nelegidos = ∅, presupuesto = K
    Seleccionando --> Seleccionando: step(a) válido\nr = riesgo_t+1[a], presupuesto −1
    Seleccionando --> Seleccionando: step(a) inválido (ya elegido)\nr = −1, sin cambio
    Seleccionando --> Terminado: presupuesto == 0
    Terminado --> Seleccionando: reset(otro año)
    Terminado --> [*]
```

## 5.4 Estados: ciclo de vida de una tarea de subagente (orquestación)

```mermaid
stateDiagram-v2
    [*] --> Pendiente
    Pendiente --> EnEjecucion: el orquestador despacha a Haiku 5.5
    EnEjecucion --> Reportada: el subagente entrega reporte + tests
    Reportada --> EnRevision: el orquestador revisa código y corre tests
    EnRevision --> Aprobada: cumple los criterios de aceptación
    EnRevision --> Correccion: falla un criterio
    Correccion --> EnEjecucion: nuevo despacho con hallazgos concretos
    Aprobada --> [*]
```
