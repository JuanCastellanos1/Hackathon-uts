# 06 · Modelo de datos (entidad-relación)

Cómo se integran las fuentes oficiales en una **tabla única**. La llave de integración es `cod_mpio` (DIVIPOLA, texto de 5 dígitos con ceros a la izquierda, p. ej. `"68001"`).

```mermaid
erDiagram
    SGC_EVENTO_ANTIGUO {
        string FECHA
        string HORA_UTC
        float LATITUD
        float LONGITUD
        float PROFUNDIDAD_Km
        float MAGNITUD_Ml
        float MAGNITUD_Mw
        string DEPARTAMENTO
        string MUNICIPIO
        int FASES
        float RMS
        float GAP
        float ERROR_LAT_LON_PROF
    }
    SGC_EVENTO_NUEVO {
        string fecha_hora_utc
        float lat
        float lon
        float prof_km
        float mag
        string tipo_mag
        int fases
        float rms_s
        float gap_grados
        float err_lat_lon_prof_km
        string region
    }
    EVENTO_UNIFICADO {
        string event_id PK
        datetime fecha_hora_utc
        int anio
        float lat
        float lon
        float prof_km
        float mag
        string tipo_mag
        int periodo_red
        string cod_mpio FK
        float dist_cabecera_km
        bool es_atipico
    }
    DIVIPOLA {
        string cod_mpio PK
        string cod_dpto
        string dpto
        string nom_mpio
        float latitud
        float longitud
    }
    DANE_POBLACION {
        string cod_mpio FK
        int anio
        string area
        int poblacion
    }
    UNGRD_EMERGENCIA {
        date fecha
        string cod_mpio FK
        string evento
        int personas
        int viviendas_destruidas
        int viviendas_averiadas
    }
    MUNICIPIO {
        string cod_mpio PK
        string nom_mpio
        string dpto
        bool es_santander
        float lat
        float lon
        float dist_nido_km
        int poblacion_total
        float pct_rural
        float emergencias_prom_anual
        int mov_masa_total
        int viviendas_afectadas_total
        int personas_afectadas_total
    }
    DATASET_MUNICIPIO_ANIO {
        string cod_mpio PK
        int anio PK
        int n_m25
        float mag_max
        float b_value
        float log_energia_J
        float n_m25_prom_3a
        string nivel_actividad_t1
        float log_energia_t1
    }
    PREDICCION_SUP {
        string cod_mpio FK
        int anio FK
        float p_bajo
        float p_medio
        float p_alto
    }
    CLUSTER_MPIO {
        string cod_mpio FK
        int cluster
        string cluster_nombre
        int cluster_rl
    }
    PRIORIZACION_2026 {
        int rango PK
        string cod_mpio FK
        float puntaje
        float prob_top_k
        string confianza
    }

    SGC_EVENTO_ANTIGUO ||--o| EVENTO_UNIFICADO : "se unifica en"
    SGC_EVENTO_NUEVO ||--o| EVENTO_UNIFICADO : "se unifica en"
    DIVIPOLA ||--|| MUNICIPIO : "base de"
    DANE_POBLACION }o--|| MUNICIPIO : "exposición"
    UNGRD_EMERGENCIA }o--|| MUNICIPIO : "vulnerabilidad"
    MUNICIPIO ||--o{ EVENTO_UNIFICADO : "cabecera más cercana"
    EVENTO_UNIFICADO }o--|| DATASET_MUNICIPIO_ANIO : "se agrega en"
    MUNICIPIO ||--o{ DATASET_MUNICIPIO_ANIO : "tiene"
    DATASET_MUNICIPIO_ANIO ||--o| PREDICCION_SUP : "walk-forward"
    MUNICIPIO ||--|| CLUSTER_MPIO : "perfil"
    MUNICIPIO ||--o| PRIORIZACION_2026 : "si es de Santander"
```

El detalle de columnas, tipos y unidades está en [`../CONTRATOS_DATOS.md`](../CONTRATOS_DATOS.md).
