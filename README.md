# TFG: Análisis exploratorio de trayectorias y flujos en ATM

**Autor:** Yungu Rhee  
**Tutor:** Juan A. Fdez del Pozo  
**Departamento:** Inteligencia Artificial, ETSIINF (UPM)

## Descripción

Pipeline de análisis de trayectorias aéreas ADS-B sobre el espacio aéreo europeo.
A partir de datos brutos de radar, el sistema limpia, proyecta, remuestrea y agrupa
las trayectorias en corredores aéreos mediante clustering HDBSCAN a dos niveles:

- **Macro**: identificación de corredores a escala continental usando distancia euclídea.
- **Micro**: identificación de flujos de llegada (STARs) en el área terminal de un
  aeropuerto usando distancia euclídea ponderada (WED), basado en el enfoque de
  Corrado et al. (2020).

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
│   │   ├── ids_vuelos_macro.npy
│   │   └── clusters_macro.parquet
│   └── micro/
│       ├── trayectorias_micro_{ICAO}.parquet
│       ├── matriz_micro_{ICAO}.npy
│       ├── ids_micro_{ICAO}.npy
│       └── clusters_micro_{ICAO}.parquet
│
├── src/
│   ├── exploracion/
│   │   ├── aviones.py
│   │   ├── vuelos.py
│   │   ├── filtrado.py
│   │   ├── columnas.py
│   │   ├── altitud.py
│   │   ├── ruidos.py
│   │   └── rutas_descartadas.py
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
│   ├── micro/
│   │   ├── recorte_micro.py
│   │   ├── distancias_micro.py
│   │   └── clustering_micro.py
│   ├── visualizacion/
│   │   ├── visualizacion_general.py
│   │   ├── visualizacion_id.py
│   │   ├── vuelos_filtrados.py
│   │   └── comparacion_pipeline.py
│   ├── dashboard/
│   │   └── dashboard_general.py
│   └── script/
│       ├── pipeline_macro.bat
│       ├── pipeline_micro.bat
│       └── limpiar_micro.bat
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
│  │              │  de altitud, rango, huecos >5min, mín.    │
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
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         │           ANÁLISIS MICRO                          │
│         │     (sobre trayectorias_proyectadas.parquet)      │
├─────────┼───────────────────────────────────────────────────┤
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ recorte_micro.py  │  Recorta trayectorias al radio de   │
│  │                   │  100 km del aeropuerto + remuestreo  │
│  └──────┬───────────┘                                       │
│         │  trayectorias_micro_{ICAO}.parquet                │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ distancias_micro  │  Weighted Euclidean Distance (WED)   │
│  │ .py               │  Peso 1.0 entrada TMA → 0.2 aero.   │
│  └──────┬───────────┘                                       │
│         │  matriz_micro_{ICAO}.npy                          │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ clustering_micro  │  HDBSCAN con min_cluster_size        │
│  │ .py               │  interactivo                         │
│  └──────┬───────────┘                                       │
│         │  clusters_micro_{ICAO}.parquet                     │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────┐
│         ▼           DASHBOARD                               │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐                                     │
│  │ dashboard_general   │  Dash + Plotly → http://127.0.0.1  │
│  │ .py                 │  :8050. Macro y micro en un único  │
│  │                     │  dashboard interactivo             │
│  └────────────────────┘                                     │
└─────────────────────────────────────────────────────────────┘
```

## Ejecución rápida

### Pipeline Macro (corredores continentales)
```bash
cd src/script/
pipeline_macro.bat
```

### Pipeline Micro (flujos en área terminal)
```bash
cd src/script/
pipeline_micro.bat
# Pide el código ICAO del aeropuerto (ej: EKCH, LOWW, LEBL)
```

Para limpiar resultados micro y empezar de cero:
```bash
cd src/script/
limpiar_micro.bat
```

### Dashboard
```bash
cd src/dashboard/
python dashboard_general.py
# Abrir http://127.0.0.1:8050
```

## Diferencia Macro vs Micro

| | Macro | Micro |
|---|---|---|
| Escala | Continental (Europa) | Terminal (100 km del aeropuerto) |
| Trayectorias | Completas | Recortadas al área terminal |
| Distancia | Euclídea | Euclídea ponderada (WED) |
| Pesos | Iguales | Más peso en entrada al TMA |
| Identifica | Corredores aéreos | STARs / flujos de aproximación |
| Referencia | Olive et al. (2019) | Corrado et al. (2020) |

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