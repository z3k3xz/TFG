import pandas as pd
import numpy as np
import time

# === CONFIGURACIÓN ===
RUTA_ENTRADA = "../../resultados/preparacion/trayectorias_proyectadas.parquet"
RUTA_SALIDA = "../../resultados/preparacion/trayectorias_normalizadas.parquet"

# Número de puntos equidistantes por vuelo
N_PUNTOS = 50


def remuestrear_vuelo(grupo):
    """
    Dado un DataFrame con los puntos de un vuelo (x, y, altitude en metros),
    calcula la distancia acumulada 3D y re-interpola a N_PUNTOS equidistantes.

    1. Calcula la distancia euclídea 3D entre puntos consecutivos.
    2. Acumula esas distancias para obtener un eje de "distancia recorrida".
    3. Divide la distancia total en N_PUNTOS tramos iguales.
    4. Interpola x, y, altitude en esos puntos.

    El resultado es que el punto 0 es el inicio del vuelo, el punto 49 es
    el final, y el punto 25 es exactamente la mitad del recorrido en distancia.
    """
    # Ordenar por timestamp
    grupo = grupo.sort_values('timestamp').reset_index(drop=True)

    x = grupo['x'].values
    y = grupo['y'].values
    alt = grupo['altitude'].values

    # Distancia euclídea 3D entre puntos consecutivos
    dx = np.diff(x)
    dy = np.diff(y)
    dalt = np.diff(alt)
    dist_segmentos = np.sqrt(dx**2 + dy**2 + dalt**2)

    # Distancia acumulada (empieza en 0)
    dist_acum = np.concatenate([[0], np.cumsum(dist_segmentos)])
    dist_total = dist_acum[-1]

    # Si el vuelo no se ha movido, no se puede interpolar
    if dist_total < 1.0:
        return None

    # Puntos equidistantes sobre la distancia total
    dist_objetivo = np.linspace(0, dist_total, N_PUNTOS)

    # Interpolación lineal de x, y, altitude
    x_interp = np.interp(dist_objetivo, dist_acum, x)
    y_interp = np.interp(dist_objetivo, dist_acum, y)
    alt_interp = np.interp(dist_objetivo, dist_acum, alt)

    return pd.DataFrame({
        'flight_id': grupo['flight_id'].iloc[0],
        'point_index': np.arange(N_PUNTOS),
        'x': x_interp,
        'y': y_interp,
        'altitude': alt_interp
    })


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(f" REMUESTREO ESPACIAL ({N_PUNTOS} puntos equidistantes)")
    print("=" * 60)
    inicio = time.time()

    # --- Carga ---
    print(f"Cargando: {RUTA_ENTRADA}")
    df = pd.read_parquet(RUTA_ENTRADA)
    n_vuelos = df['flight_id'].nunique()
    n_puntos = len(df)
    print(f"Cargados {n_puntos:,} puntos de {n_vuelos:,} vuelos")

    # --- Estadísticas de distancia antes del remuestreo ---
    print(f"\nCalculando distancias por vuelo...")
    distancias = []
    for fid, grupo in df.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        dx = np.diff(grupo['x'].values)
        dy = np.diff(grupo['y'].values)
        dalt = np.diff(grupo['altitude'].values)
        dist = np.sum(np.sqrt(dx**2 + dy**2 + dalt**2))
        distancias.append({'flight_id': fid, 'distancia_total_km': dist / 1000})

    df_dist = pd.DataFrame(distancias)
    print(f"  Distancia mínima:  {df_dist['distancia_total_km'].min():.1f} km")
    print(f"  Distancia mediana: {df_dist['distancia_total_km'].median():.1f} km")
    print(f"  Distancia máxima:  {df_dist['distancia_total_km'].max():.1f} km")

    # --- Remuestreo ---
    print(f"\nRemuestreando a {N_PUNTOS} puntos equidistantes...")
    resultados = []
    descartados = 0

    for fid, grupo in df.groupby('flight_id'):
        resultado = remuestrear_vuelo(grupo)
        if resultado is not None:
            resultados.append(resultado)
        else:
            descartados += 1

    df_remuestreado = pd.concat(resultados, ignore_index=True)

    # --- Resumen ---
    n_vuelos_final = df_remuestreado['flight_id'].nunique()

    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Vuelos: {n_vuelos} -> {n_vuelos_final} (Descartados: {descartados})")
    print(f"  Puntos por vuelo: {N_PUNTOS}")
    print(f"  Dimensiones de la matriz: {n_vuelos_final} x {N_PUNTOS} x 3 (x, y, alt)")
    print(f"  Vector por vuelo: {N_PUNTOS * 3} componentes")

    # --- Guardado ---
    df_remuestreado.to_parquet(RUTA_SALIDA, index=False)
    print(f"  Guardado en: {RUTA_SALIDA}")
    print(f"  Completado en {time.time() - inicio:.1f} segundos")