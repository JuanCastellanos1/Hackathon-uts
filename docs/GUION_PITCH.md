# Guion del pitch · Santander Inteligente

**Formato del jurado:** 10 min de presentación y demo + 3 min de preguntas.
**Material:** `docs/presentacion.pptx` (10 diapositivas, con notas del orador) y la demo (`demo_windows.bat` o `streamlit run app/app.py`).

Todas las cifras de este documento salen de `reports/*.json` y de `data/processed/priorizacion_2026.csv`. Si se regenera el pipeline, las cifras no cambian, porque las semillas son fijas.

---

## 0. Antes de entrar (5 minutos antes)

- [ ] La demo abierta en el navegador, en la pestaña **"Priorización 2026"**, y probada con el slider en K = 10.
- [ ] La presentación abierta en la diapositiva 1.
- [ ] Probado el mapa **en el navegador que se usará**. Si sale en blanco, activar **"Mapa compatible (sin internet)"** en la barra lateral y dejarlo así (ver §6, Plan B).
- [ ] Una terminal abierta en la carpeta del repo, por si piden ejecutar algo en vivo (`python src/06_agente_rl.py` tarda unos 45 s).

---

## 1. Guion cronometrado (10:00)

> Lo que va entre comillas es texto para decir en voz alta. Lo que va entre corchetes es una acción.

### 0:00 – 0:45 · Diapositiva 1: Portada

"Buenas tardes. Somos el equipo de *Santander Inteligente*. Construimos un sistema que integra datos oficiales del Servicio Geológico Colombiano, del DANE y de la UNGRD con tres modelos de Machine Learning: uno supervisado, uno no supervisado y un **agente entrenado por refuerzo**. Con ellos responde una pregunta práctica: **si la Gobernación solo puede hacer 10 estudios o inspecciones sísmicas al año, ¿en qué municipios de Santander debería hacerlos primero?**"

### 0:45 – 1:45 · Diapositiva 2: El problema

"Santander es uno de los departamentos con más actividad sísmica del país. Bajo la Mesa de Los Santos está el **Nido sísmico de Bucaramanga**, una de las zonas de sismicidad intermedia más activas del mundo. Solo contando sismos de magnitud 2,5 o más, hay en promedio unos 85 al mes en Santander.

Pero que tiemble mucho en un lugar no basta para decidir. Usamos el marco clásico de gestión del riesgo: **Riesgo = Amenaza × Exposición × Vulnerabilidad**. La amenaza sale del catálogo sísmico, la exposición de la población del DANE y la vulnerabilidad del historial de emergencias de la UNGRD. El reto es combinar esas tres cosas de forma que no sea arbitraria. Ahí entra el aprendizaje por refuerzo."

### 1:45 – 3:00 · Diapositiva 3: Datos e integración

"La fuente principal es el **catálogo de la Red Sismológica Nacional del SGC**. Trae dos consultas con columnas distintas: de junio de 1993 a febrero de 2018, procesada con Earthworm y SEISAN, y de marzo de 2018 a hoy, procesada con SeisComP3. Las descargamos automáticamente para un recuadro que cubre Santander y sus vecinos: **222 mil sismos**. Las unificamos en un solo esquema y eliminamos 38 duplicados.

La integración con las otras fuentes se hace por el **código DIVIPOLA** del municipio. Cada sismo se asigna a la cabecera municipal más cercana. Validamos el método contra el municipio que reporta el propio SGC y **coincide en el 95,3 %**.

Hay una decisión metodológica importante. La red sismológica se hizo más densa con los años: en los 90 solo se detectaban bien los sismos de magnitud 3, y hoy se detectan los de 1,9. Si contáramos todos los sismos, el modelo aprendería 'hay más estaciones', no 'tiembla más'. Por eso **solo usamos sismos de magnitud 2,5 o más y desde 2002**, donde el catálogo es completo.

El resultado es **una sola tabla de 4.224 filas, municipio por año, con 35 variables justificadas** en el diccionario de datos."

### 3:00 – 3:45 · Diapositiva 4: Arquitectura del flujo

"El flujo es lineal y reproducible. Datos oficiales, tabla única, dos modelos que estiman la amenaza y la tipología de cada municipio, el agente de refuerzo que prioriza y la demo. Todo se regenera con **un solo comando, `run_all.py`, en unos dos minutos**, y tiene **57 pruebas automáticas**. Todas las evaluaciones respetan el tiempo: entrenamos con el pasado y evaluamos con el futuro."

### 3:45 – 4:45 · Diapositiva 5: Modelo supervisado

"El primer modelo, el del reto Básico, es un **Random Forest** que clasifica cada municipio en actividad **baja, media o alta para el año siguiente**. La variable objetivo es el número de sismos de magnitud 2,5 o más en el año t+1, y las entradas son solo del año t o anteriores, así que no es circular.

Entrenamos con 2002–2015, elegimos hiperparámetros con 2016–2019 y evaluamos en **2020–2024**, datos que el modelo nunca vio. Obtenemos una **F1-macro de 0,77**, con un intervalo de confianza del 95 % entre 0,73 y 0,81, y detecta **el 74 % de los casos de actividad alta**. En la matriz se ve que **nunca confunde 'bajo' con 'alto'**.

Lo decimos con honestidad: una regla simple de persistencia, 'el año que viene será como este', ya logra 0,74. Le ganamos por poco, y el intervalo incluye ese valor. La sismicidad tiene mucha memoria. Por eso no usamos este modelo solo: es **una de las señales** que recibe el agente."

### 4:45 – 5:45 · Diapositiva 6: Modelo no supervisado

"El segundo modelo, el del reto Medio, es **K-Means sin etiquetas**. Compara el perfil sísmico de los 176 municipios del recuadro: tasas, magnitudes, profundidades y el valor b de Gutenberg-Richter. Elegimos **k = 5 grupos** por el criterio de silueta, que da **0,54**.

Salen grupos con sentido geológico: un **'Nido'** de 3 municipios, un anillo de **sismicidad intermedia activa** alrededor y un grupo **cortical o superficial** que domina en Boyacá y Norte de Santander. Comparando territorios, Santander tiene el 85 % de sus municipios en los grupos de sismicidad intermedia, mientras que los vecinos son mayoritariamente superficiales.

Y un resultado que nos gusta: **DBSCAN, también sin etiquetas, encuentra solo el Nido de Bucaramanga**, en 6,81° N, 73,11° O y a 148 km de profundidad, con casi 7.900 sismos. Es justamente lo que dice la literatura."

### 5:45 – 7:00 · Diapositiva 7: Agente RL, formulación

"Ahora, el componente central. Formulamos la priorización como un **problema de aprendizaje por refuerzo**:

- **Entorno:** un año de decisión con los 87 municipios de Santander y un presupuesto de 10 estudios.
- **Estado:** de cada municipio, la probabilidad de actividad alta del modelo supervisado, su grupo del no supervisado, su población, su historial de emergencias y su actividad pasada.
- **Acción:** elegir un municipio. El episodio termina tras 10 elecciones.
- **Recompensa:** el **riesgo que efectivamente se materializó el año siguiente** en ese municipio, es decir, la energía sísmica liberada en t+1 multiplicada por su exposición y vulnerabilidad. **El agente nunca ve el futuro**: solo recibe la recompensa después de decidir, como en la vida real.

Usamos **REINFORCE**, un algoritmo de gradiente de política, con una política lineal. La ventaja es que los pesos aprendidos se pueden leer: nos dicen **qué criterio valora el agente**. Entrenamos con 2006–2015 y 10 semillas distintas, y evaluamos en 2020–2024."

### 7:00 – 7:45 · Diapositiva 8: Agente RL, resultados

"¿Funciona? Lo comparamos con el **oráculo**, que es quien conoce el futuro y elige perfecto, y con cinco reglas que usaría una persona: al azar, por población, por actividad pasada, por el modelo supervisado y un índice con pesos iguales.

El agente captura el **78,6 % del riesgo máximo posible**, contra 72,8 % de la mejor regla, el índice de pesos iguales, y 25 % del azar. Es estable entre semillas, con una desviación de 2,3 puntos. La curva muestra que **aprende**: en validación pasa de 66 % a cerca de 74 %. Es decir, **los pesos aprendidos con datos son mejores que los pesos puestos a mano**."

### 7:45 – 9:15 · DEMO EN VIVO (diapositiva 9 como respaldo)

[Cambiar a la demo, pestaña **Priorización 2026**]

"Esta es la priorización exploratoria para 2026. [Señalar el mapa] Los círculos grandes son los municipios prioritarios. El top 5 es **Los Santos, El Playón, Bucaramanga, El Carmen de Chucurí y Barrancabermeja**, y 9 de los 10 primeros tienen **confianza alta**, porque los 10 agentes entrenados coinciden.

[Mover el slider de K a 15 y volver a 10] El tomador de decisiones puede ajustar el presupuesto.

[En el selector de municipio, elegir **Los Santos**] Para Los Santos, la mayor parte del puntaje viene de la **amenaza**: es el epicentro del Nido.

[Elegir **Bucaramanga**] En cambio Bucaramanga casi no tiene amenaza propia según el modelo, pero entra en el top 3 por **exposición y vulnerabilidad**: 600 mil habitantes e historial de emergencias. Eso es justo lo que queremos: **el agente no prioriza solo dónde tiembla, sino dónde un sismo haría más daño**."

[Sección **Simulador del decisor**: subir la vulnerabilidad a 70 %, K = 5 y pulsar **Reentrenar agente** (tarda unos 10 s)]

"Y algo que solo permite el aprendizaje por refuerzo: **ustedes ponen las prioridades**. Supongamos que para la Gobernación pesa más la vulnerabilidad y solo hay presupuesto para 5 estudios. [Pulsar] El agente **se reentrena en vivo** con ese objetivo. [Señalar métricas] Sigue superando a una fórmula fija con esos mismos pesos, y aquí vemos qué municipios entran y salen del ranking. **Ustedes definen qué es importante; el agente aprende cómo lograrlo.**"

> Si el tiempo aprieta, se salta el simulador. Si alguien del jurado quiere probarlo, se le deja mover los controles.

### 9:15 – 10:00 · Diapositiva 10: Impacto, limitaciones y cierre

"¿Para qué le sirve esto a Santander? Para **orientar dónde hacer primero** estudios de microzonificación, inspecciones estructurales y simulacros con un presupuesto limitado, con una justificación transparente por municipio.

Somos claros con las limitaciones. Es una priorización **exploratoria, no predice sismos**. La recompensa es un indicador aproximado del riesgo, porque no hay datos de daños por sismo con resolución municipal. La exposición y la vulnerabilidad son fijas en el tiempo, y asignamos los sismos por cabecera más cercana y no por polígonos oficiales.

Como extensiones proponemos los polígonos del IGAC, la amenaza de la norma NSR-10, los daños históricos del SISH y el costo de cada inspección dentro de la recompensa.

Todo el código, los datos y las pruebas están en el repositorio y se reproducen con un comando. Muchas gracias."

---

## 2. Cómo funciona el código, parte por parte

El proyecto es una **cadena de 7 scripts** (`src/01` a `src/07`) que usan una librería propia (`src/sismo/`). Cada paso lee lo que dejó el anterior en disco y deja su resultado para el siguiente.

```
01 descarga SGC ─┐
02 descarga DANE/UNGRD ─┴─► 03 tabla única ─► 04 supervisado ─┐
                                          └─► 05 no supervisado ─┴─► 06 agente RL ─► priorización 2026
                                                                                   └─► 07 presentación · app/
```

### 2.1 `src/01_descarga_sgc.py`: descarga del catálogo SGC
- **Qué hace:** consulta automáticamente los dos formularios del catálogo SGC. El de 1993–2018 se descarga año por año en Excel. El de 2018–hoy se consulta mes por mes, porque la web devuelve una tabla HTML.
- **Decisión:** se guardan **los archivos crudos sin tocar** en `data/raw/` junto con `metadata_descarga_sgc.json` (URL, fecha de descarga y filtros). Esa es la evidencia de la fuente oficial citada que pide la rúbrica.
- **Detalle técnico:** el Excel del SGC trae una hoja de estilos dañada que hace fallar a pandas, así que se lee el XML interno del `.xlsx` directamente.
- **En una frase:** "Descargamos el catálogo oficial de forma reproducible y guardamos el rastro de cada consulta."

### 2.2 `src/02_descarga_fuentes_complementarias.py`: DANE y UNGRD
- **Qué hace:** descarga DIVIPOLA (códigos y coordenadas de los municipios), las proyecciones de población municipal del DANE y dos archivos de emergencias de la UNGRD desde datos.gov.co.
- **Por qué estas fuentes:** cubren la **exposición** (cuánta gente) y la **vulnerabilidad** (qué tan seguido sufre emergencias el municipio), las dos partes del riesgo que el catálogo sísmico no tiene.

### 2.3 `src/sismo/catalogo.py`: unificación del catálogo
- **Qué hace:** lleva las dos consultas a un **mismo esquema** (fecha, latitud, longitud, profundidad, magnitud, RMS, gap, errores) y elimina **38 duplicados**: el mismo sismo con la misma hora, posición, profundidad y magnitud. Además marca **607 eventos atípicos** por mala calidad de localización (RMS > 2 s o error de profundidad > 50 km).
- **Decisión:** los atípicos no se borran; se marcan y se excluyen de las variables. Así queda trazabilidad.
- **Nota:** algunas profundidades salen negativas (hasta −5 km). SeisComP3 mide la profundidad desde el nivel del mar, así que un sismo superficial bajo la cordillera puede quedar "por encima" de cero. Son válidas y se cuentan como superficiales.

### 2.4 `src/sismo/territorio.py`: municipios, población y emergencias
- **Qué hace:** arma la tabla de 176 municipios del recuadro (87 de Santander) con su población de 2020, el porcentaje rural y sus emergencias. Asigna cada sismo a la **cabecera municipal más cercana** usando distancia sobre la esfera (haversine) con un `BallTree` de scikit-learn.
- **Decisión 1:** cabecera más cercana en vez de polígonos oficiales. Es más simple, funciona igual para los dos catálogos y coincide en un 95,3 % con el municipio del SGC.
- **Decisión 2:** en los datos de la UNGRD algunos códigos DIVIPOLA vienen mal escritos, así que se resuelve **primero por nombre** del municipio y luego por código. Así se recupera el 99 % de los registros.

### 2.5 `src/sismo/features.py` + `src/03_tabla.py`: la tabla única
- **Qué hace:** agrega los sismos por **municipio y año** (2002–2025) y calcula las variables. Son 35 en total, y todas están justificadas en `docs/diccionario_datos.csv`.
  - **Actividad:** conteos de magnitud ≥ 2,5, ≥ 3,0 y ≥ 3,5, magnitud máxima y media, y energía liberada (`log10 Σ 10^(1,5M+4,8)` julios).
  - **Física sísmica:** **valor b de Gutenberg-Richter**, calculado con el estimador de máxima verosimilitud de Aki. Un b bajo indica proporcionalmente más sismos grandes.
  - **Profundidad:** profundidad media y mínima, % superficial (< 30 km) y % intermedio (70–300 km, la firma del Nido).
  - **Calidad:** RMS, gap, error de profundidad y número de fases. Miden qué tan confiable es la localización.
  - **Memoria:** promedio de los 3 años anteriores, tendencia y máximo histórico.
  - **Territorio, exposición y vulnerabilidad:** distancia al Nido, población, % rural, emergencias, movimientos en masa, viviendas y personas afectadas.
- **Variable objetivo:** `nivel_actividad_t1`. Es "bajo" si el año siguiente hay 0 sismos de magnitud ≥ 2,5, "medio" si hay de 1 a 4 y "alto" si hay 5 o más.
  - Se usan umbrales fijos y no terciles, porque el 64 % de los municipio-año tienen 0 sismos.
  - **No es circular:** se calcula con el año t+1, y las entradas son del año t o anteriores.
- **Controles anti-fuga:** el código verifica con `assert` que no haya duplicados, que no haya nulos y que ninguna variable del futuro (`*_t1`) sea una entrada.

### 2.6 `src/sismo/supervisado.py` + `src/04_supervisado.py`: amenaza del año siguiente (reto Básico)
- **Modelo:** Random Forest con 300 árboles, profundidad máxima 8, mínimo 3 muestras por hoja y `class_weight="balanced_subsample"`, que compensa que la clase "alto" sea solo el 6 % de los datos.
- **Cómo se eligió:** se comparó con Gradient Boosting en una rejilla de hiperparámetros, eligiendo por F1-macro en validación (RF 0,793 frente a 0,791 del mejor HGB).
- **Validación temporal:** entrenamiento con 2002–2015, validación con 2016–2019 y prueba con 2020–2024. Para la prueba, el modelo se reentrena con 2002–2019.
- **Métrica principal: F1-macro.** Promedia la F1 de las tres clases por igual. Con el accuracy, un modelo que siempre diga "bajo" acertaría el 67 % sin servir para nada; con F1-macro ese mismo modelo saca 0,27.
- **Incertidumbre:** intervalo bootstrap con 1.000 remuestreos, de 0,731 a 0,809.
- **Explicación:** importancia por permutación. Domina el **promedio de los 3 años anteriores**, seguido de la longitud y la magnitud máxima histórica.
- **Walk-forward:** para cada año t entre 2006 y 2025 se entrena **solo con años anteriores** y se predice t. Esas probabilidades son las que recibe el agente, para que no vea información que no tendría en la vida real.

### 2.7 `src/sismo/no_supervisado.py` + `src/05_no_supervisado.py`: tipologías territoriales (reto Medio)
- **Datos:** un perfil por municipio con 11 variables (tasas, magnitudes, valor b, profundidades, energía y fracción de años activos), a las que se aplica logaritmo y estandarización. **No usa ninguna etiqueta.**
- **Elección de k:** se probaron k de 2 a 8 con silueta, Davies-Bouldin, inercia y BIC de mezcla gaussiana. Se eligió **k = 5**, que tiene la mejor silueta (0,538).
- **Nombres de los grupos:** se asignan con reglas sobre los centroides: Nido (3 municipios), Intermedia activa en la periferia del Nido (24), Intermedia moderada (75), Cortical o superficial (60) y Baja actividad (14).
- **Variables que más separan los grupos:** según un árbol de decisión que imita los clusters, son el % superficial (0,48), el valor b (0,26) y el % intermedio (0,20).
- **DBSCAN:** se aplica sobre los epicentros de magnitud ≥ 3 en coordenadas en km (x, y, profundidad) y encuentra el Nido sin que se le diga dónde está.
- **`cluster_rl`:** para el agente, el agrupamiento se ajusta **solo con 2002–2015**, de modo que el estado del agente no use información futura.

### 2.8 `src/sismo/entorno_rl.py`: el entorno (Gymnasium)
- **Clase `PriorizacionEnv`:** sigue la API estándar de Gymnasium, con `reset` y `step`.
- **Observación:** una matriz de 87 × (15 variables + "ya elegido" + "presupuesto restante").
- **Acción:** el índice de un municipio. Si el agente elige uno repetido, recibe −1 y no gasta presupuesto.
- **Fin del episodio:** termina cuando se eligen K = 10 municipios.
- **`oraculo(año)`:** devuelve la suma de las 10 mejores recompensas, el techo teórico que se usa para medir el % del óptimo.

### 2.9 `src/sismo/agente_rl.py`: el agente (REINFORCE)
- **Política:** softmax lineal. Cada municipio recibe un puntaje `w · x`, y la probabilidad de elegirlo es proporcional a `e^(w·x)` entre los que no se han elegido.
- **Aprendizaje (REINFORCE con línea base):** al final de cada episodio se calcula el retorno de cada paso y se compara con una media móvil de retornos. Si la decisión salió mejor que lo habitual, se refuerza: `w += lr · (G − b) · (x_elegido − x_promedio)`. Además se suma un pequeño término de **entropía** para que siga explorando.
- **Por qué lineal y no una red neuronal (DQN):** solo hay 10 años de entrenamiento, y una red se sobreajustaría. Además, una política lineal da **pesos interpretables**, que es lo que necesita un tomador de decisiones.
- **Pruebas:** en un entorno sintético donde la respuesta correcta se conoce, el agente llega al 99 % del óptimo. Además, los gradientes se verificaron contra diferencias finitas.

### 2.10 `src/sismo/evaluacion_rl.py` + `src/06_agente_rl.py`: entrenamiento, evaluación y priorización
- **Estado (15 variables, estandarizadas con los años de entrenamiento):**
  - Amenaza: P(alto), P(medio), actividad de los 3 años anteriores, magnitud máxima histórica, distancia al Nido y el grupo (one-hot de 5).
  - Exposición: log(población) y % rural.
  - Vulnerabilidad: emergencias, movimientos en masa y viviendas afectadas.
- **Recompensa:** `amenaza(t+1) × (0,5·exposición + 0,5·vulnerabilidad)`.
  - La amenaza es la energía sísmica real liberada al año siguiente, normalizada entre 0 y 1.
  - La exposición y la vulnerabilidad están escaladas entre 0 y 1 dentro de Santander.
- **Entrenamiento:** 10 semillas × 300 épocas. Cada época recorre los años 2006–2015 en orden aleatorio, y el % del óptimo en validación (2016–2019) se mide cada 10 épocas.
- **Evaluación en 2020–2024:** se mide la recompensa, el % del óptimo y la **precision@K** (cuántos de los 10 elegidos están en el top 10 real).

| Política | % del óptimo | precision@10 |
|---|---|---|
| **Agente RL** | **78,6 %** (± 2,3 entre semillas) | **0,61** |
| Índice de pesos iguales | 72,8 % | 0,54 |
| Mayor población | 64,8 % | 0,54 |
| Mayor actividad pasada | 64,8 % | 0,38 |
| Ranking del supervisado | 61,9 % | 0,34 |
| Azar | 25,1 % | 0,12 |

- **Pesos aprendidos (los más grandes):** emergencias (+4,6), P(alto) (+4,0), población (+3,9), movimientos en masa (+2,9) y distancia al Nido (+2,1). El agente aprendió a **combinar amenaza con exposición y vulnerabilidad**.
- **Priorización 2026:** se aplican los 10 agentes al estado de 2025.
  - El **puntaje** es el promedio de los puntajes de los 10 agentes.
  - La **confianza** es la fracción de agentes que ponen el municipio en su top 10: alta si es 0,8 o más, media si es 0,5 o más, y baja en otro caso.
  - Las **contribuciones** separan cuánto del puntaje viene de amenaza, de exposición y de vulnerabilidad.

### 2.11 `app/app.py`: la demo
Es una aplicación Streamlit que **solo lee los resultados ya calculados**: no entrena nada, así que no puede fallar por tiempo de cómputo. Tiene 5 pestañas, una por cada parte de la historia, y en "Priorización 2026" el decisor mueve K y ve la ficha de cada municipio.

### 2.12 `run_all.py`, `tests/` y reproducibilidad
- `run_all.py` corre los pasos 03 a 07 en unos 2 minutos. Con `--descargar` vuelve a bajar los datos de las fuentes oficiales.
- **57 pruebas automáticas** (`pytest`) verifican, entre otras cosas, el esquema de los datos, que no haya fuga temporal, que las probabilidades sumen 1, que el agente aprenda en un caso sintético y la mecánica del entorno.
- Las semillas son fijas. Se verificó regenerando todo en un clon limpio: los resultados son idénticos salvo redondeos del orden de 10⁻¹³.

---

### 2.13 La fórmula de riesgo y su relación con el agente

**Riesgo = Amenaza × Exposición × Vulnerabilidad.** En el código, la recompensa del agente es:

**Recompensa = Amenaza(t+1) × (0,5 · Exposición + 0,5 · Vulnerabilidad)**

| Variable | Pregunta que responde | Fuente | Cálculo (escala 0 a 1) |
|---|---|---|---|
| **Amenaza (t+1)** | ¿Cuánto tembló de verdad el año siguiente? | SGC | Energía de los sismos de magnitud ≥ 2,5 del municipio en t+1, normalizada |
| **Exposición** | ¿Cuánta gente podría verse afectada? | DANE | log(población), normalizada entre los 87 municipios |
| **Vulnerabilidad** | ¿Qué tan frágil es el municipio? | UNGRD | Promedio de emergencias por año, movimientos en masa y viviendas afectadas, cada uno normalizado |

**Por qué se multiplica:** si no tiembla, el riesgo es 0 aunque haya mucha gente; si tiembla pero no vive nadie, el riesgo es bajo. Solo hay riesgo alto cuando coinciden el sismo y la gente o la fragilidad.

**Relación con el agente:**

| Variable | ¿La ve al decidir? | Papel |
|---|---|---|
| Amenaza (t+1) | **No** (es el futuro) | Solo llega en la recompensa, después de decidir. El agente debe **anticiparla** con P(alto) del supervisado, el grupo del no supervisado, la actividad pasada y la distancia al Nido |
| Exposición | Sí (población, % rural) | Está en el estado y en la recompensa |
| Vulnerabilidad | Sí (emergencias, movimientos en masa, viviendas) | Está en el estado y en la recompensa |

Lo que el agente **realmente aprende** es a anticipar dónde habrá amenaza el próximo año y a combinarla con lo que ya sabe.

**El 0,5 + 0,5:** es el peso de la exposición frente a la vulnerabilidad (50/50 en la priorización oficial). En el simulador del decisor es justamente ese peso el que se mueve, y el agente se reentrena para el nuevo objetivo.

> Frase: *"La recompensa es el riesgo que realmente ocurrió: sismicidad del año siguiente por gente expuesta y fragilidad. El agente no ve el futuro; aprende a anticiparlo combinando lo que sí sabe."*

---

## 3. Decisiones clave y su porqué

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Magnitud ≥ 2,5 y desde 2002 | Usar todos los sismos desde 1993 | La red se hizo más densa. Con todos los sismos, el modelo aprendería el cambio de instrumentos y no el de la sismicidad |
| Unidad municipio × año | Celdas de una grilla | El municipio es la unidad en que decide la Gobernación y permite cruzar con DANE y UNGRD |
| Cabecera más cercana | Polígonos oficiales del IGAC | Es rápido, uniforme para los dos catálogos y coincide en un 95,3 %. Los polígonos quedan como extensión |
| Objetivo en el año t+1 con umbrales fijos | Terciles, o el mismo año | Los terciles fallan con un 64 % de ceros, y usar el mismo año sería circular |
| Exposición y vulnerabilidad fuera del supervisado | Meterlas como variables | El supervisado modela la amenaza física. Además, los datos de la UNGRD son de 2019–2025 y filtrarían información futura |
| F1-macro + bootstrap | Accuracy | La clase "alto" es rara. El accuracy premiaría al modelo que siempre dice "bajo" |
| Política lineal + REINFORCE | DQN o redes profundas | Pocos datos (10 años de entrenamiento), estabilidad y, sobre todo, pesos interpretables |
| Recompensa con el riesgo realizado en t+1 | Recompensa con la predicción del modelo | El agente aprende de lo que de verdad pasó, no de otra predicción |
| 10 semillas y confianza por consenso | Un solo agente | Mide qué tan robusta es la priorización |
| Comparar contra 5 reglas y el oráculo | Reportar solo la recompensa | Un número sin comparación no dice nada. El oráculo da la escala del 0 al 100 % |

---

## 4. Preguntas probables del jurado

**"¿Por qué aprendizaje por refuerzo y no un índice ponderado?"**
Porque los pesos de un índice se ponen a mano. El agente los **aprende** a partir de qué decisiones resultaron mejores en años pasados, y en datos que nunca vio supera al índice de pesos iguales: 78,6 % frente a 72,8 %. Además, decidir con presupuesto es secuencial (cada elección reduce las opciones), que es justo el tipo de problema que el refuerzo modela.

**"¿El agente hace trampa viendo el futuro?"**
No. El estado solo contiene información disponible al final del año t, y hay una aserción en el código que lo verifica. Las probabilidades del supervisado son walk-forward, entrenadas solo con años anteriores. La recompensa usa el año t+1, pero el agente solo la recibe después de decidir, como pasaría en la realidad. Además, la evaluación es en años 2020–2024 que nunca vio al entrenar.

**"El supervisado casi empata con la persistencia, ¿entonces no sirve?"**
La sismicidad tiene mucha memoria, así que la persistencia es una línea base fuerte, y lo reportamos con honestidad. El modelo aporta probabilidades calibradas para tres clases y nunca confunde "bajo" con "alto". Dentro del agente es una señal más: el agente le da un peso de +4,0 a P(alto).

**"¿Por qué Bucaramanga sale tercera si su amenaza es baja?"**
Porque el riesgo no es solo amenaza. Bucaramanga tiene 600 mil habitantes y un historial alto de emergencias. Su puntaje viene de la exposición (+12,3) y la vulnerabilidad (+15,2), mientras que su amenaza aporta −2,9. Que un sismo moderado ocurra ahí tiene más consecuencias que en un municipio rural del Nido.

**"¿Por qué la distancia al Nido tiene peso positivo?"**
Es un efecto condicional. La cercanía al Nido ya entra a través de P(alto) y del grupo "Nido". Una vez descontado eso, la distancia termina favoreciendo a municipios grandes y alejados, como Barrancabermeja, que tienen mucha exposición. Los pesos de un modelo lineal no se interpretan aislados.

**"¿Qué significa 'confianza baja' para 76 municipios?"**
Que los 10 agentes casi nunca los ponen en su top 10. Con un presupuesto de 10 sobre 87 eso es lo esperado: la confianza dice qué tan seguro es que un municipio **esté** entre los prioritarios.

**"Si el decisor cambia los pesos, ¿no es lo mismo que una fórmula?"**
No. El decisor no fija los pesos del ranking: fija **su objetivo** (la recompensa). El agente aprende **cómo cumplirlo sin saber dónde temblará**, es decir, aprende a anticipar la amenaza. En el simulador se ve que, con los mismos pesos, el agente supera a la fórmula fija (por ejemplo 86 % frente a 76 % con 70 % de vulnerabilidad y K = 5). En casos extremos (80 % exposición) empatan, porque la población domina y ya no hay mucho que anticipar.

**"¿Cómo integraron las fuentes?"**
Por el código DIVIPOLA del municipio. Cada sismo se asigna a la cabecera más cercana (coincide en un 95,3 % con el SGC), la población del DANE se cruza por código y las emergencias de la UNGRD por nombre y código (99 % resueltas). Todo queda en una sola tabla de 4.224 filas.

**"¿Cuáles son las 20 variables?"**
Hay 35, todas justificadas en `docs/diccionario_datos.csv`. Las principales son conteos por magnitud, magnitud máxima y media, energía, valor b, profundidades, porcentajes superficial e intermedio, calidad de localización, rezagos, distancia al Nido, población, % rural y cuatro de emergencias.

**"¿Lo pueden ejecutar ahora?"**
Sí: `python src/06_agente_rl.py` reentrena los 10 agentes en unos 45 segundos, y `python run_all.py` regenera todo en unos 2 minutos.

**"¿Qué harían con más tiempo?"**
Usar los polígonos del IGAC, la amenaza de la norma NSR-10 (Aa/Av), los daños históricos del SISH como recompensa más fiel, el costo de cada inspección en la recompensa y una vulnerabilidad que cambie en el tiempo.

---

## 5. Cifras para memorizar

| Dato | Valor |
|---|---|
| Sismos del catálogo SGC (recuadro) | 222.047 (tras quitar 38 duplicados) |
| Tabla única | 4.224 filas (176 municipios × 2002–2025), 35 variables |
| Coincidencia de la asignación municipal | 95,3 % |
| Supervisado: F1-macro en prueba | 0,771 (IC 95 %: 0,73–0,81) · persistencia 0,743 · mayoritaria 0,268 |
| Supervisado: recall de la clase "alto" | 74 % (34 de 46) |
| No supervisado | k = 5, silueta 0,54 · DBSCAN: Nido en 6,81° N, 73,11° O, 148 km |
| Agente RL: % del óptimo en 2020–2024 | 78,6 % (± 2,3) · mejor regla 72,8 % · azar 25,1 % |
| Top 5 de 2026 | Los Santos, El Playón, Bucaramanga, El Carmen de Chucurí, Barrancabermeja |
| Reproducibilidad | 1 comando, unos 2 min, 57 pruebas |

---

### 5.1 Si preguntan por precisión y resultados: cifras con sus salvedades

**Supervisado (prueba 2020–2024, 880 municipio-año):**

| Métrica | Valor | Salvedad al decirla |
|---|---|---|
| Exactitud | **78,2 %** (688 de 880) | No es la métrica principal: decir siempre "bajo" ya acertaría el 67 % sin servir para nada. Por eso reportamos F1-macro |
| F1-macro | **0,77** (IC 95 %: 0,73–0,81) | Es la métrica principal. La persistencia ("el año que viene será igual") da 0,74 y **cae dentro del intervalo**: le ganamos por poco y no de forma concluyente |
| Precisión en "alto" | **94,4 %** (34 de 36) | Pocos casos: 36 predicciones "alto" en la prueba |
| Recall en "alto" | **73,9 %** (34 de 46) | Se escapan 12 casos "alto", que el modelo clasifica como "medio" |
| Errores graves | **0** | Nunca confunde "bajo" con "alto" en ninguna dirección |

> La exactitud y la precisión en "alto" se calculan de la matriz de confusión (`reports/metricas_supervisado.json`, clave `matriz_confusion`); no están guardadas como campo propio.

**Agente RL (prueba 2020–2024):** captura el **78,6 %** del riesgo máximo (± 2,3 entre semillas) frente al 72,8 % de la mejor fórmula manual y el 25,1 % del azar. Acierta **6 de 10** municipios del top 10 real (precision@10 = 0,61).
- **Salvedad:** el "riesgo" de la recompensa es un **indicador aproximado** (energía sísmica del año siguiente × exposición × vulnerabilidad), no daños reales. La ventaja sobre la mejor fórmula es **moderada (+5,8 puntos) pero estable**.
- **Salvedad del simulador:** el agente supera a la fórmula con los mismos pesos en la mayoría de escenarios (por ejemplo 86 % frente a 76 % con 70 % de vulnerabilidad y K = 5), pero **empata** cuando la exposición pesa 80 %, porque la población domina el objetivo y queda poco que anticipar.

**No supervisado:** silueta **0,54** (de −1 a 1; por encima de 0,5 se considera una estructura razonable).
- **Salvedad:** no hay "acierto" que medir porque no hay etiquetas. La validación es la coherencia geológica: DBSCAN recupera el Nido donde lo ubica la literatura.

**Conclusión para cerrar si piden resumen:**
"Integramos 4 fuentes oficiales en 35 variables y construimos tres modelos que se complementan. El supervisado anticipa la actividad con 78 % de exactitud y 94 % de precisión en actividad alta, aunque frente a la persistencia la ventaja es pequeña. El no supervisado descubre el Nido sin etiquetas. El agente de refuerzo, en años que nunca vio, captura el 79 % del riesgo máximo, más que cualquier regla manual. No predecimos sismos: le decimos a Santander **dónde mirar primero** con un presupuesto limitado, de forma transparente y reproducible."

---

## 6. Plan B (si algo falla en vivo)

- **El mapa sale en blanco o sin fondo** (sin internet, Brave con Shields u otro navegador restrictivo):
  1. Abrir la barra lateral (`»` arriba a la izquierda) y activar **"Mapa compatible (sin internet)"**. Los mapas se redibujan sin WebGL ni teselas, con los mismos municipios, colores, tamaños y el Nido.
  2. Frase para el jurado: *"Cambio a la vista compatible, que no depende de internet; los datos son exactamente los mismos."*
  3. En Brave, otra opción es desactivar los Shields para `localhost` (ícono del león) y recargar.
  4. Si aun así falla, usar la diapositiva 9 (mapa en PNG) o `reports/figures/rl_mapa_priorizacion.png`.
- **La demo no arranca:** presentar con las diapositivas 8 y 9. Las figuras están en `reports/figures/`.
- **Piden ver el código del agente:** abrir `src/sismo/agente_rl.py`, método `actualizar` (la actualización REINFORCE), y `src/sismo/entorno_rl.py`, método `step` (cómo se entrega la recompensa).
- **Se acaba el tiempo:** saltar la diapositiva 4 (arquitectura) y resumir la 6 en una frase: "el no supervisado encuentra el Nido sin etiquetas".
