import numpy as np
import pandas as pd
import hdbscan
import time
import os
import sys

# === CONFIGURACIÓN ===
RUTA_MICRO = "../../resultados/micro/"

# min_samples para HDBSCAN
MIN_SAMPLES = 5


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" CLUSTERING HDBSCAN (MICRO)")
    print("=" * 60)
    inicio = time.time()

    # --- Detección de aeropuertos disponibles ---
    archivos = [f for f in os.listdir(RUTA_MICRO) if f.startswith('matriz_micro_') and f.endswith('.npy')]
    aeropuertos = [f.replace('matriz_micro_', '').replace('.npy', '') for f in archivos]

    if len(aeropuertos) == 0:
        print("ERROR: No hay matrices de distancias micro. Ejecuta distancias_micro.py primero.")
        exit(1)

    print(f"Aeropuertos disponibles: {', '.join(sorted(aeropuertos))}")

    if len(sys.argv) > 1:
        codigo = sys.argv[1].strip().upper()
        print(f"Aeropuerto (argumento): {codigo}")
    else:
        codigo = input(f"Código ICAO del aeropuerto: ").strip().upper()

    if codigo not in aeropuertos:
        print(f"ERROR: No hay datos para {codigo}")
        exit(1)

    # --- Carga ---
    ruta_matriz = os.path.join(RUTA_MICRO, f"matriz_micro_{codigo}.npy")
    ruta_ids = os.path.join(RUTA_MICRO, f"ids_micro_{codigo}.npy")

    print(f"\nCargando matriz de distancias: {ruta_matriz}")
    dist_matrix = np.load(ruta_matriz)
    ids_vuelos = np.load(ruta_ids, allow_pickle=True)
    n_vuelos = len(ids_vuelos)
    print(f"Matriz: {dist_matrix.shape}")

    # --- Selección de min_cluster_size ---
    reco_bajo = max(int(n_vuelos * 0.005), 2)
    reco_alto = max(int(n_vuelos * 0.02), 5)
    print(f"\nVuelos totales: {n_vuelos}")
    print(f"Rango recomendado para min_cluster_size: {reco_bajo} - {reco_alto}")
    entrada = input(f"min_cluster_size [{reco_bajo}]: ").strip()
    MIN_CLUSTER_SIZE = int(entrada) if entrada else reco_bajo

    # --- HDBSCAN ---
    print(f"\nEjecutando HDBSCAN...")
    print(f"  min_cluster_size = {MIN_CLUSTER_SIZE}")
    print(f"  min_samples = {MIN_SAMPLES}")

    t_cluster = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='precomputed'
    )
    etiquetas = clusterer.fit_predict(dist_matrix)
    print(f"  Calculado en {time.time() - t_cluster:.2f} segundos")

    # --- Resultados ---
    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    n_ruido = np.sum(etiquetas == -1)
    pct_ruido = n_ruido / n_vuelos * 100

    print(f"\n{'=' * 60}")
    print(f"RESULTADO — {codigo}")
    print(f"{'=' * 60}")
    print(f"  Clusters encontrados: {n_clusters}")
    print(f"  Vuelos en clusters: {n_vuelos - n_ruido} ({100 - pct_ruido:.1f}%)")
    print(f"  Vuelos ruido: {n_ruido} ({pct_ruido:.1f}%)")

    print(f"\nDistribución por cluster:")
    df_resultado = pd.DataFrame({'flight_id': ids_vuelos, 'cluster': etiquetas})
    conteo = df_resultado['cluster'].value_counts().sort_index()
    for cluster_id, count in conteo.items():
        if cluster_id == -1:
            print(f"  Ruido: {count} vuelos")
        else:
            print(f"  Cluster {cluster_id}: {count} vuelos")

    # --- Guardado ---
    ruta_clusters = os.path.join(RUTA_MICRO, f"clusters_micro_{codigo}.parquet")
    df_resultado.to_parquet(ruta_clusters, index=False)

    print(f"\nGuardado en: {ruta_clusters}")
    print(f"Completado en {time.time() - inicio:.1f} segundos")