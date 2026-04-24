import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

# === CONFIGURACIÓN ===
RUTA_ENTRADA = "../../resultados/preparacion/trayectorias_normalizadas.parquet"
RUTA_SALIDA_MATRIZ = "../../resultados/macro/matriz_distancias_macro.npy"
RUTA_SALIDA_IDS = "../../resultados/macro/ids_vuelos_macro.npy"

import os


def preparar_matriz(df, n_puntos=50):
    """
    Convierte el DataFrame en formato largo (n_puntos filas por vuelo)
    en una matriz donde cada fila es un vuelo y cada columna una
    componente del vector aplanado (x_0, x_1, ..., y_0, ..., alt_0, ...).
    """
    # Pivotar: una fila por vuelo, columnas = point_index x variable
    vuelos = df.pivot_table(
        index='flight_id',
        columns='point_index',
        values=['x', 'y', 'altitude'],
        aggfunc='first'
    )

    # Ordenar columnas: x_0..x_49, y_0..y_49, alt_0..alt_49
    vuelos = vuelos.reindex(columns=sorted(vuelos.columns, key=lambda c: (c[0], c[1])))

    # Aplanar los MultiIndex de columnas
    ids_vuelos = vuelos.index.values
    matriz = vuelos.values.astype(np.float64)

    return matriz, ids_vuelos


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" CÁLCULO DE MATRIZ DE DISTANCIAS (MACRO)")
    print("=" * 60)
    inicio = time.time()

    # --- Crear directorio de salida ---
    os.makedirs(os.path.dirname(RUTA_SALIDA_MATRIZ), exist_ok=True)

    # --- Carga ---
    print(f"Cargando: {RUTA_ENTRADA}")
    df = pd.read_parquet(RUTA_ENTRADA)
    n_vuelos = df['flight_id'].nunique()
    print(f"Cargados {n_vuelos} vuelos")

    # --- Construcción de la matriz de vectores ---
    print(f"Construyendo matriz de vectores...")
    matriz, ids_vuelos = preparar_matriz(df)
    print(f"  Forma: {matriz.shape} ({matriz.shape[0]} vuelos x {matriz.shape[1]} componentes)")

    # --- Verificar NaN ---
    nans = np.isnan(matriz).sum()
    if nans > 0:
        print(f"  AVISO: {nans} valores NaN en la matriz")
    else:
        print(f"  Sin NaN")

    # --- Cálculo de distancias euclídeas ---
    print(f"Calculando distancias euclídeas entre {n_vuelos} vuelos...")
    print(f"  Pares a calcular: {n_vuelos * (n_vuelos - 1) // 2:,}")

    t_dist = time.time()
    dist_condensada = pdist(matriz, metric='euclidean')
    dist_matrix = squareform(dist_condensada)
    print(f"  Calculado en {time.time() - t_dist:.2f} segundos")

    # --- Estadísticas de distancias ---
    # Extraer solo el triángulo superior (sin la diagonal)
    tri_superior = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"\nEstadísticas de distancias:")
    print(f"  Mínima:  {tri_superior.min():.0f} metros")
    print(f"  Mediana: {np.median(tri_superior):.0f} metros")
    print(f"  Máxima:  {tri_superior.max():.0f} metros")
    print(f"  Media:   {tri_superior.mean():.0f} metros")

    # --- Guardado ---
    np.save(RUTA_SALIDA_MATRIZ, dist_matrix)
    np.save(RUTA_SALIDA_IDS, ids_vuelos)

    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Matriz: {dist_matrix.shape}")
    print(f"  Guardada en: {RUTA_SALIDA_MATRIZ}")
    print(f"  IDs guardados en: {RUTA_SALIDA_IDS}")
    print(f"  Completado en {time.time() - inicio:.1f} segundos")