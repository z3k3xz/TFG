# TFG: Análisis exploratorio de trayectorias y flujos en ATM

**Autor:** Yungu Rhee  
**Tutor:** Juan A. Fdez del Pozo  
**Departamento:** Inteligencia Artificial, ETSIINF (UPM)

## Estructura del proyecto

```
TFG/
├── datos/
│   ├── parquet/
│   │   ├── 2022-01-01.parquet
│   │   ├── 2022-01-02.parquet
│   │   └── ...
│   └── flight_list.csv
│
├── resultados/
│   ├── preparacion/
│   │   ├── datos_fusionados.parquet
│   │   ├── trayectorias_limpias.parquet
│   │   ├── trayectorias_proyectadas.parquet
│   │   └── trayectorias_normalizadas.parquet
│   ├── macro/
│   │   ├── matriz_distancias_macro.npy
│   │   ├── clusters_macro.parquet
│   │   └── caracterizacion_macro/
│   └── micro/
│       ├── tramos_terminales_lemd.parquet
│       ├── matriz_distancias_micro.npy
│       ├── clusters_micro_lemd.parquet
│       └── caracterizacion_micro/
│
├── src/
│   ├── exploracion/          ← ya hecho
│   │   ├── aviones.py
│   │   ├── vuelos.py
│   │   ├── filtrado.py
│   │   ├── ruidos.py
│   │   └── altitud.py
│   ├── visualizacion/        ← ya hecho
│   │   ├── visualizacion_general.py
│   │   ├── visualizacion_id.py
│   │   ├── vuelos_filtrados.py
│   │   └── comparacion_pipeline.py
│   ├── preparacion/          ← Bloque 1
│   │   ├── fusionar.py
│   │   ├── limpieza.py
│   │   ├── proyeccion.py
│   │   └── remuestreo_espacial.py
│   ├── macro/                ← Bloque 2
│   │   ├── distancias_macro.py
│   │   ├── clustering_macro.py
│   │   ├── caracterizacion_macro.py
│   │   └── visualizar_macro.py
│   ├── micro/                ← Bloque 3
│   │   ├── filtrar_terminal.py
│   │   ├── distancias_micro.py
│   │   ├── clustering_micro.py
│   │   ├── caracterizacion_micro.py
│   │   └── visualizar_micro.py
│   └── dashboard/            ← Bloque 4
│       └── dashboard.py
│
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