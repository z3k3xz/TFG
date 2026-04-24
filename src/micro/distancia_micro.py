import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time
import os

# === CONFIGURACIÓN ===
RUTA_MICRO = "../../resultados/micro/"

# Número de puntos por vuelo
N_PUNTOS = 50


def preparar_matriz(df, n_puntos):
    """
    Convierte el DataFrame en formato largo en una matriz donde
    cada fila es un vuelo y cada columna una componente del vector.
    """
    vuelos = df.pivot_table(
        index='flight_id', columns='point_index',
        values=['x', 'y', 'altitude'], aggfunc='first'
    )
    vuelos = vuelos.reindex(columns=sorted(vuelos.columns, key=lambda c: (c[0], c[1])))

    ids_vuelos = vuelos.index.values
    matriz = vuelos.values.astype(np.float64)

    return matriz, ids_vuelos


def calcular_distancias_wed(matriz, n_puntos):
    """
    Calcula la matriz de distancias usando Weighted Euclidean Distance (WED).

    Los pesos dan más importancia a los puntos del inicio del tramo terminal
    (entrada al TMA, lejos del aeropuerto) y menos a los puntos finales
    (cerca del aeropuerto, donde todos convergen).

    El punto 0 es la entrada al TMA, el punto 49 es el aeropuerto.
    Los pesos van de 1.0 (punto 0) a un valor mínimo (punto 49).
    Se usa un decrecimiento lineal: w_i = 1 - 0.8 * (i / (n_puntos - 1))
    Así el primer punto pesa 1.0 y el último 0.2.
    """
    # Pesos por punto: de 1.0 (entrada TMA) a 0.2 (aeropuerto)
    pesos_por_punto = np.array([1.0 - 0.8 * (i / (n_puntos - 1)) for i in range(n_puntos)])

    # Expandir pesos a las 3 coordenadas (x, y, alt) de cada punto
    # La matriz tiene columnas: x_0..x_49, y_0..y_49, alt_0..alt_49
    pesos = np.concatenate([pesos_por_punto, pesos_por_punto, pesos_por_punto])

    # Aplicar pesos a la matriz
    matriz_ponderada = matriz * pesos

    # Distancia euclídea sobre los vectores ponderados
    dist_condensada = pdist(matriz_ponderada, metric='euclidean')
    return squareform(dist_condensada)


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" CÁLCULO DE DISTANCIAS PONDERADAS (WED) — MICRO")
    print("=" * 60)
    inicio = time.time()

    # --- Detección de aeropuertos disponibles ---
    archivos = [f for f in os.listdir(RUTA_MICRO) if f.startswith('trayectorias_micro_') and f.endswith('.parquet')]
    aeropuertos = [f.replace('trayectorias_micro_', '').replace('.parquet', '') for f in archivos]

    if len(aeropuertos) == 0:
        print("ERROR: No hay trayectorias micro. Ejecuta recorte_micro.py primero.")
        exit(1)

    print(f"Aeropuertos disponibles: {', '.join(sorted(aeropuertos))}")
    codigo = input(f"Código ICAO del aeropuerto: ").strip().upper()

    if codigo not in aeropuertos:
        print(f"ERROR: No hay datos para {codigo}")
        exit(1)

    # --- Carga ---
    ruta_tray = os.path.join(RUTA_MICRO, f"trayectorias_micro_{codigo}.parquet")
    print(f"\nCargando: {ruta_tray}")
    df = pd.read_parquet(ruta_tray)
    n_vuelos = df['flight_id'].nunique()
    print(f"Cargados {n_vuelos} vuelos")

    # --- Construcción de la matriz ---
    print(f"Construyendo matriz de vectores...")
    matriz, ids_vuelos = preparar_matriz(df, N_PUNTOS)
    print(f"  Forma: {matriz.shape}")

    # --- Verificar NaN ---
    nans = np.isnan(matriz).sum()
    if nans > 0:
        print(f"  AVISO: {nans} valores NaN")
    else:
        print(f"  Sin NaN")

    # --- Cálculo de distancias WED ---
    print(f"\nCalculando distancias ponderadas (WED) entre {n_vuelos} vuelos...")
    print(f"  Pares a calcular: {n_vuelos * (n_vuelos - 1) // 2:,}")

    t_dist = time.time()
    dist_matrix = calcular_distancias_wed(matriz, N_PUNTOS)
    print(f"  Calculado en {time.time() - t_dist:.2f} segundos")

    # --- Estadísticas ---
    tri_superior = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    print(f"\nEstadísticas de distancias WED:")
    print(f"  Mínima:  {tri_superior.min():.0f}")
    print(f"  Mediana: {np.median(tri_superior):.0f}")
    print(f"  Máxima:  {tri_superior.max():.0f}")

    # --- Guardado ---
    ruta_matriz = os.path.join(RUTA_MICRO, f"matriz_micro_{codigo}.npy")
    ruta_ids = os.path.join(RUTA_MICRO, f"ids_micro_{codigo}.npy")
    np.save(ruta_matriz, dist_matrix)
    np.save(ruta_ids, ids_vuelos)

    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Matriz: {dist_matrix.shape}")
    print(f"  Guardada en: {ruta_matriz}")
    print(f"  IDs guardados en: {ruta_ids}")
    print(f"  Completado en {time.time() - inicio:.1f} segundos")