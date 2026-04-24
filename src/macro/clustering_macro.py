import numpy as np
import pandas as pd
import hdbscan
import time

# === CONFIGURACIÓN ===
RUTA_MATRIZ = "../../resultados/macro/matriz_distancias_macro.npy"
RUTA_IDS = "../../resultados/macro/ids_vuelos_macro.npy"
RUTA_SALIDA = "../../resultados/macro/clusters_macro.parquet"

#pip install hdbscan

# Parámetros de HDBSCAN
# min_cluster_size: tamaño mínimo de un cluster para ser considerado válido.
#   Valores bajos → más clusters pequeños. Valores altos → menos clusters, más ruido.
# min_samples: controla lo conservador que es el algoritmo.
#   Valores altos → más puntos etiquetados como ruido (-1).
MIN_CLUSTER_SIZE = 15
MIN_SAMPLES = 10


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" CLUSTERING HDBSCAN (MACRO)")
    print("=" * 60)
    inicio = time.time()

    # --- Carga ---
    print(f"Cargando matriz de distancias: {RUTA_MATRIZ}")
    dist_matrix = np.load(RUTA_MATRIZ)
    ids_vuelos = np.load(RUTA_IDS, allow_pickle=True)
    n_vuelos = len(ids_vuelos)
    print(f"Matriz: {dist_matrix.shape}")

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
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Clusters encontrados: {n_clusters}")
    print(f"  Vuelos asignados a cluster: {n_vuelos - n_ruido} ({100 - pct_ruido:.1f}%)")
    print(f"  Vuelos etiquetados como ruido: {n_ruido} ({pct_ruido:.1f}%)")

    # --- Distribución por cluster ---
    print(f"\nDistribución por cluster:")
    df_resultado = pd.DataFrame({
        'flight_id': ids_vuelos,
        'cluster': etiquetas
    })

    conteo = df_resultado['cluster'].value_counts().sort_index()
    for cluster_id, count in conteo.items():
        if cluster_id == -1:
            print(f"  Ruido (outliers): {count} vuelos")
        else:
            print(f"  Cluster {cluster_id}: {count} vuelos")

    # --- Guardado ---
    df_resultado.to_parquet(RUTA_SALIDA, index=False)
    print(f"\nGuardado en: {RUTA_SALIDA}")
    print(f"Completado en {time.time() - inicio:.1f} segundos")