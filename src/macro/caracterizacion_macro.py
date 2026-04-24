import pandas as pd
import numpy as np
import time

# === CONFIGURACIÓN ===
RUTA_CLUSTERS = "../../resultados/macro/clusters_macro.parquet"
RUTA_TRAYECTORIAS = "../../resultados/preparacion/trayectorias_proyectadas.parquet"
RUTA_METADATOS = "../../datos/flight_list.csv"

    
def caracterizar_cluster(df_tray, df_meta, cluster_id, vuelos_cluster):
    """
    Genera un resumen estadístico de un cluster:
    - Número de vuelos
    - Aeropuertos de origen y destino más frecuentes
    - Rutas (pares origen-destino) más frecuentes
    - Altitud máxima media (crucero)
    - Distancia recorrida media
    - Franja horaria de salida
    """
    resumen = {'cluster': cluster_id, 'n_vuelos': len(vuelos_cluster)}

    # --- Metadatos de los vuelos del cluster ---
    meta_cluster = df_meta[df_meta['flight_id'].isin(vuelos_cluster)]

    # Aeropuertos de origen
    if 'adep' in meta_cluster.columns:
        origenes = meta_cluster['adep'].value_counts()
        resumen['top_origenes'] = ', '.join([f"{a} ({c})" for a, c in origenes.head(5).items()])
        resumen['n_origenes_distintos'] = origenes.nunique() if hasattr(origenes, 'nunique') else len(origenes)
    else:
        resumen['top_origenes'] = 'N/A'
        resumen['n_origenes_distintos'] = 0

    # Aeropuertos de destino
    if 'ades' in meta_cluster.columns:
        destinos = meta_cluster['ades'].value_counts()
        resumen['top_destinos'] = ', '.join([f"{a} ({c})" for a, c in destinos.head(5).items()])
        resumen['n_destinos_distintos'] = destinos.nunique() if hasattr(destinos, 'nunique') else len(destinos)
    else:
        resumen['top_destinos'] = 'N/A'
        resumen['n_destinos_distintos'] = 0

    # Rutas más frecuentes (origen-destino)
    if 'adep' in meta_cluster.columns and 'ades' in meta_cluster.columns:
        meta_cluster = meta_cluster.copy()
        meta_cluster['ruta'] = meta_cluster['adep'] + ' → ' + meta_cluster['ades']
        rutas = meta_cluster['ruta'].value_counts()
        resumen['top_rutas'] = ', '.join([f"{r} ({c})" for r, c in rutas.head(5).items()])
    else:
        resumen['top_rutas'] = 'N/A'

    # --- Estadísticas de trayectoria ---
    tray_cluster = df_tray[df_tray['flight_id'].isin(vuelos_cluster)]

    # Altitud máxima por vuelo (proxy de altitud de crucero)
    alt_max_por_vuelo = tray_cluster.groupby('flight_id')['altitude'].max()
    resumen['alt_crucero_media'] = f"{alt_max_por_vuelo.mean():.0f} m"
    resumen['alt_crucero_min'] = f"{alt_max_por_vuelo.min():.0f} m"
    resumen['alt_crucero_max'] = f"{alt_max_por_vuelo.max():.0f} m"

    # Distancia recorrida por vuelo
    distancias = []
    for fid, grupo in tray_cluster.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        dx = np.diff(grupo['x'].values)
        dy = np.diff(grupo['y'].values)
        dist_km = np.sum(np.sqrt(dx**2 + dy**2)) / 1000
        distancias.append(dist_km)

    resumen['dist_media_km'] = f"{np.mean(distancias):.0f} km"
    resumen['dist_min_km'] = f"{np.min(distancias):.0f} km"
    resumen['dist_max_km'] = f"{np.max(distancias):.0f} km"

    # Franja horaria de salida
    primer_punto = tray_cluster.sort_values('timestamp').groupby('flight_id')['timestamp'].first()
    if pd.api.types.is_datetime64_any_dtype(primer_punto):
        horas = primer_punto.dt.hour
        resumen['hora_salida_media'] = f"{horas.mean():.1f}h"
        resumen['hora_salida_rango'] = f"{horas.min():.0f}h - {horas.max():.0f}h"
    else:
        resumen['hora_salida_media'] = 'N/A'
        resumen['hora_salida_rango'] = 'N/A'

    return resumen


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" CARACTERIZACIÓN DE CLUSTERS (MACRO)")
    print("=" * 60)
    inicio = time.time()

    # --- Carga ---
    print(f"Cargando clusters: {RUTA_CLUSTERS}")
    df_clusters = pd.read_parquet(RUTA_CLUSTERS)

    print(f"Cargando trayectorias: {RUTA_TRAYECTORIAS}")
    df_tray = pd.read_parquet(RUTA_TRAYECTORIAS)

    print(f"Cargando metadatos: {RUTA_METADATOS}")
    df_meta = pd.read_csv(RUTA_METADATOS)

    # --- Columnas disponibles en metadatos ---
    print(f"Columnas en flight_list.csv: {list(df_meta.columns)}")

    # --- Caracterización ---
    clusters_validos = sorted(df_clusters[df_clusters['cluster'] >= 0]['cluster'].unique())
    print(f"\nCaracterizando {len(clusters_validos)} clusters...\n")

    for cluster_id in clusters_validos:
        vuelos = df_clusters[df_clusters['cluster'] == cluster_id]['flight_id'].values
        resumen = caracterizar_cluster(df_tray, df_meta, cluster_id, vuelos)

        print(f"--- Cluster {cluster_id} ({resumen['n_vuelos']} vuelos) ---")
        print(f"  Orígenes:   {resumen['top_origenes']}")
        print(f"  Destinos:   {resumen['top_destinos']}")
        print(f"  Rutas:      {resumen['top_rutas']}")
        print(f"  Alt crucero: {resumen['alt_crucero_media']} (rango: {resumen['alt_crucero_min']} - {resumen['alt_crucero_max']})")
        print(f"  Distancia:  {resumen['dist_media_km']} (rango: {resumen['dist_min_km']} - {resumen['dist_max_km']})")
        print(f"  Hora salida: {resumen['hora_salida_media']} (rango: {resumen['hora_salida_rango']})")
        print()

    # --- Caracterización del ruido ---
    vuelos_ruido = df_clusters[df_clusters['cluster'] == -1]['flight_id'].values
    if len(vuelos_ruido) > 0:
        resumen_ruido = caracterizar_cluster(df_tray, df_meta, -1, vuelos_ruido)

        print(f"--- RUIDO ({resumen_ruido['n_vuelos']} vuelos) ---")
        print(f"  Orígenes:   {resumen_ruido['top_origenes']}")
        print(f"  Destinos:   {resumen_ruido['top_destinos']}")
        print(f"  Alt crucero: {resumen_ruido['alt_crucero_media']} (rango: {resumen_ruido['alt_crucero_min']} - {resumen_ruido['alt_crucero_max']})")
        print(f"  Distancia:  {resumen_ruido['dist_media_km']} (rango: {resumen_ruido['dist_min_km']} - {resumen_ruido['dist_max_km']})")
        print()

    print(f"Completado en {time.time() - inicio:.1f} segundos")