# TFG: Análisis exploratorio de trayectorias y flujos en ATM

**Autor:** Yungu Rhee  
**Tutor:** Juan A. Fdez del Pozo  
**Departamento:** Inteligencia Artificial, ETSIINF (UPM)

## Descripción

Pipeline de análisis de trayectorias aéreas ADS-B sobre el espacio aéreo europeo.
A partir de datos brutos de radar, el sistema limpia, proyecta, remuestrea y agrupa
las trayectorias en corredores aéreos mediante clustering HDBSCAN, permitiendo
identificar y caracterizar los flujos de tráfico.

## Datos

Los datos proceden del [PRC Data Challenge 2024](https://doi.org/10.4121/8cb8484b-dbe7-4750-8b87-a5b1dbc621b4)
(EUROCONTROL + OpenSky Network). No se incluyen en el repositorio por su tamaño.

Ficheros necesarios:
- `datos/parquet/*.parquet` — Trayectorias ADS-B (un fichero por día)
- `datos/flight_list.csv` — Metadatos de los 527.162 vuelos (aeropuertos, aerolínea, tipo de aeronave)

Solo se procesan los vuelos que tienen metadatos en `flight_list.csv`.

## Estructura del proyecto

```
TFG/
├── datos/ (.gitignore)
│   ├── parquet/
│   │   ├── 2022-01-01.parquet
│   │   ├── 2022-01-02.parquet
│   │   └── ...
│   └── flight_list.csv
│
├── resultados/ (.gitignore)
│   ├── preparacion/
│   │   ├── datos_fusionados.parquet
│   │   ├── trayectorias_limpias.parquet
│   │   ├── trayectorias_proyectadas.parquet
│   │   └── trayectorias_normalizadas.parquet
│   └── macro/
│       ├── matriz_distancias_macro.npy
│       ├── ids_vuelos_macro.npy
│       └── clusters_macro.parquet
│
├── src/
│   ├── exploracion/
│   │   ├── aviones.py
│   │   ├── vuelos.py
│   │   ├── filtrado.py
│   │   ├── columnas.py
│   │   ├── altitud.py
│   │   └── ruidos.py
│   ├── preparacion/
│   │   ├── fusionar.py
│   │   ├── limpieza.py
│   │   ├── proyeccion.py
│   │   └── remuestreo_espacial.py
│   ├── macro/
│   │   ├── distancias_macro.py
│   │   ├── clustering_macro.py
│   │   ├── caracterizacion_macro.py
│   │   └── visualizar_macro.py
│   ├── dashboard/
│   │   └── dashboard.py
│   └── script/
│       └── pipeline_macro.bat
│
└── README.md
```

## Flujo del pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    PREPARACIÓN DE DATOS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  *.parquet + flight_list.csv                                │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │ fusionar.py  │  Concatena parquets, descarta vuelos      │
│  │              │  sin metadatos en flight_list.csv          │
│  └──────┬──────┘                                            │
│         │  datos_fusionados.parquet                          │
│         ▼                                                   │
│  ┌─────────────┐                                            │
│  │ limpieza.py  │  5 filtros: saltos de posición, errores   │
│  │              │  de altitud, rango, huecos, puntos mín.   │
│  └──────┬──────┘                                            │
│         │  trayectorias_limpias.parquet                      │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ proyeccion.py │  WGS84 (grados) → LCC EPSG:3034 (metros)│
│  │               │  Altitud: pies → metros                  │
│  └──────┬───────┘                                           │
│         │  trayectorias_proyectadas.parquet                  │
│         ▼                                                   │
│  ┌───────────────────┐                                      │
│  │ remuestreo_espacial│  50 puntos equidistantes por        │
│  │ .py                │  distancia física recorrida         │
│  └──────┬────────────┘                                      │
│         │  trayectorias_normalizadas.parquet                 │
│         │  (N vuelos × 50 puntos × 3 coords)                │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         │           ANÁLISIS MACRO                          │
├─────────┼───────────────────────────────────────────────────┤
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ distancias_macro  │  Matriz de distancias euclídeas      │
│  │ .py               │  NxN entre vectores de 150 comp.    │
│  └──────┬───────────┘                                       │
│         │  matriz_distancias_macro.npy                       │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ clustering_macro  │  HDBSCAN con min_cluster_size        │
│  │ .py               │  interactivo (recomendación 0.5-2%) │
│  └──────┬───────────┘                                       │
│         │  clusters_macro.parquet                            │
│         ▼                                                   │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ caracterizacion   │  │ visualizar_macro │                 │
│  │ _macro.py         │  │ .py              │                 │
│  │ (consola)         │  │ (plt.show)       │                 │
│  └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         ▼           DASHBOARD                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐                                           │
│  │ dashboard.py  │  Dash + Plotly → http://127.0.0.1:8050   │
│  │               │  Mapa 2D interactivo + vista 3D          │
│  │               │  + caracterización por cluster           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

## Ejecución rápida

```bash
cd src/script/
pipeline_macro.bat
```

Esto ejecuta los pasos 1-6 del pipeline. Después, para ver los resultados:

```bash
cd src/dashboard/
python dashboard.py
```

Y abrir `http://127.0.0.1:8050` en el navegador.

## Dependencias

```
pandas
numpy
scipy
pyproj
hdbscan
matplotlib
dash
plotly
```
