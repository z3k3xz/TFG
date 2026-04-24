# TFG: Análisis exploratorio de trayectorias y flujos en ATM

**Autor:** Yungu Rhee  
**Tutor:** Juan A. Fdez del Pozo  
**Departamento:** Inteligencia Artificial, ETSIINF (UPM)

## Estructura del proyecto

```
TFG/
├── datos/                         Datos de entrada (no se suben a GitHub)
│   ├── 2022-01-01.parquet            Trayectorias ADS-B crudas
│   └── flight_list.csv               Metadatos de vuelos
│
├── resultados/                    Datos de salida (no se suben a GitHub)
│   ├── ids_seleccionados.txt         IDs filtrados -> filtrado.py
│   ├── vuelos_ruidosos.txt           Top 10 vuelos con más ruido -> ruidos.py
│   ├── trayectorias_limpias.parquet  Salida de limpieza -> limpieza.py
│   └── trayectorias_remuestreadas.parquet  Salida de remuestreo -> remuestreo.py
│
├── src/
│   ├── exploracion/               Fase 1: inspección y análisis
│   │   ├── aviones.py                Lectura interactiva del CSV
│   │   ├── vuelos.py                 Lectura interactiva del parquet
│   │   ├── filtrado.py               Filtro de vuelos (puedes editar a cualquiera) (ej: EGLL→EIDW)
│   │   ├── ruidos.py                 Detector de mayores ruidos por vuelo 
│   │   └── altitud.py                Histograma de altitudes
│   │
│   ├── preparacion/               Fase 2: limpieza y remuestreo
│   │   ├── limpieza.py               Pipeline de 5 filtros 
│   │   └── remuestreo.py             Interpolación a intervalos de 30s
│   │
│   ├── clustering/                Fase 3: (pendiente)
│   │
│   └── visualizacion/             Scripts de gráficas y diagnóstico
│       ├── visualizacion_id.py           dashboard de 6 gráficas (1 vuelo)
│       ├── visualizacion_general.py     dashboard en bucle (100 vuelos)
│       ├── vuelos_filtrados.py  dashboard para vuelos ruidosos
│       ├── comparacion.py            datos crudos vs remuestreados
│       └── muestreo.py   mostrar intervalos temporales 
│
├── .gitignore
└── README.md
```

## Pipeline de ejecución

```
1. Exploración    →  src/exploracion/*.py
2. Limpieza       →  src/preparacion/limpieza.py
3. Remuestreo     →  src/preparacion/remuestreo.py
4. Normalización  →  (pendiente)
5. Clustering     →  (pendiente)
```

## Datos

Los datos ADS-B proceden de [OpenSky Network](https://opensky-network.org/) y no se incluyen en el repositorio por su tamaño. Coloca los ficheros en la carpeta `datos/` antes de ejecutar los scripts.