import pandas as pd
import numpy as np
import time
import os
import sys

# === CONFIGURACIÓN ===
RUTA_TRAYECTORIAS = "../../resultados/preparacion/trayectorias_proyectadas.parquet"
RUTA_METADATOS = "../../datos/flight_list.csv"
RUTA_SALIDA = "../../resultados/micro/"

# Radio del área terminal en metros (100 km)
RADIO_TERMINAL = 100_000

# Número de puntos tras remuestreo
N_PUNTOS = 50


def obtener_posicion_aeropuerto(df_tray, df_meta, codigo_aeropuerto):
    """
    Obtiene la posición del aeropuerto en coordenadas LCC.
    Usa el último punto de las trayectorias que aterrizan ahí
    como proxy de la posición del aeropuerto.
    """
    # Vuelos que aterrizan en este aeropuerto
    ids_destino = df_meta[df_meta['ades'] == codigo_aeropuerto]['flight_id'].values
    df_destino = df_tray[df_tray['flight_id'].isin(ids_destino)]

    if len(df_destino) == 0:
        return None, None

    # Último punto de cada vuelo (proxy de posición del aeropuerto)
    ultimos = df_destino.sort_values('timestamp').groupby('flight_id').last()
    x_aero = ultimos['x'].median()
    y_aero = ultimos['y'].median()

    return x_aero, y_aero


def recortar_terminal(df_tray, flight_ids, x_aero, y_aero, radio):
    """
    Para cada vuelo, conserva solo los puntos dentro del radio
    del aeropuerto de destino. Calcula la distancia al aeropuerto
    para cada punto y filtra los que están dentro del radio.
    """
    resultados = []
    descartados = 0

    for fid in flight_ids:
        vuelo = df_tray[df_tray['flight_id'] == fid].sort_values('timestamp')

        if len(vuelo) == 0:
            descartados += 1
            continue

        # Distancia de cada punto al aeropuerto
        dist_al_aero = np.sqrt((vuelo['x'].values - x_aero)**2 + (vuelo['y'].values - y_aero)**2)

        # Puntos dentro del radio
        mascara = dist_al_aero <= radio

        if mascara.sum() < 10:
            descartados += 1
            continue

        vuelo_recortado = vuelo[mascara].copy()
        resultados.append(vuelo_recortado)

    return resultados, descartados


def remuestrear_terminal(vuelo_df, n_puntos):
    """
    Remuestrea un tramo terminal a n_puntos equidistantes.
    Igual que en el macro pero sobre el tramo recortado.
    """
    vuelo_df = vuelo_df.sort_values('timestamp').reset_index(drop=True)

    x = vuelo_df['x'].values
    y = vuelo_df['y'].values
    alt = vuelo_df['altitude'].values

    # Distancia acumulada 3D
    dx = np.diff(x)
    dy = np.diff(y)
    dalt = np.diff(alt)
    dist_segmentos = np.sqrt(dx**2 + dy**2 + dalt**2)
    dist_acum = np.concatenate([[0], np.cumsum(dist_segmentos)])
    dist_total = dist_acum[-1]

    if dist_total < 1.0:
        return None

    # Interpolación a n_puntos equidistantes
    dist_objetivo = np.linspace(0, dist_total, n_puntos)
    x_interp = np.interp(dist_objetivo, dist_acum, x)
    y_interp = np.interp(dist_objetivo, dist_acum, y)
    alt_interp = np.interp(dist_objetivo, dist_acum, alt)

    return pd.DataFrame({
        'flight_id': vuelo_df['flight_id'].iloc[0],
        'point_index': np.arange(n_puntos),
        'x': x_interp,
        'y': y_interp,
        'altitude': alt_interp
    })


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" RECORTE Y REMUESTREO — ÁREA TERMINAL")
    print("=" * 60)
    inicio = time.time()

    # --- Crear directorio ---
    os.makedirs(RUTA_SALIDA, exist_ok=True)

    # --- Carga ---
    print(f"Cargando trayectorias: {RUTA_TRAYECTORIAS}")
    df_tray = pd.read_parquet(RUTA_TRAYECTORIAS)
    print(f"Cargando metadatos: {RUTA_METADATOS}")
    df_meta = pd.read_csv(RUTA_METADATOS)

    # --- Selección de aeropuerto ---
    ids_presentes = df_tray['flight_id'].unique()
    meta_presentes = df_meta[df_meta['flight_id'].isin(ids_presentes)]
    top_destinos = meta_presentes['ades'].value_counts().head(15)

    print(f"\nTop 15 aeropuertos de destino:")
    for i, (code, count) in enumerate(top_destinos.items(), 1):
        nombre = meta_presentes[meta_presentes['ades'] == code]['name_ades'].iloc[0]
        print(f"  {i:2d}. {code} ({nombre}) — {count} vuelos")

    if len(sys.argv) > 1:
        codigo = sys.argv[1].strip().upper()
        print(f"\nAeropuerto (argumento): {codigo}")
    else:
        codigo = input(f"\nCódigo ICAO del aeropuerto a analizar: ").strip().upper()

    # --- Posición del aeropuerto ---
    print(f"\nCalculando posición de {codigo}...")
    x_aero, y_aero = obtener_posicion_aeropuerto(df_tray, df_meta, codigo)

    if x_aero is None:
        print(f"ERROR: No se encontraron vuelos con destino {codigo}")
        exit(1)

    print(f"  Posición LCC: ({x_aero:.0f}, {y_aero:.0f})")

    # --- Vuelos que aterrizan en este aeropuerto ---
    ids_destino = df_meta[df_meta['ades'] == codigo]['flight_id'].values
    ids_destino = [fid for fid in ids_destino if fid in ids_presentes]
    print(f"  Vuelos con destino {codigo}: {len(ids_destino)}")

    # --- Recorte al área terminal ---
    print(f"\nRecortando al área terminal ({RADIO_TERMINAL/1000:.0f} km)...")
    tramos, descartados = recortar_terminal(df_tray, ids_destino, x_aero, y_aero, RADIO_TERMINAL)
    print(f"  Tramos válidos: {len(tramos)}")
    print(f"  Descartados (< 10 puntos en el terminal): {descartados}")

    # --- Remuestreo ---
    print(f"\nRemuestreando a {N_PUNTOS} puntos equidistantes...")
    resultados = []
    descartados_remuestreo = 0
    for tramo in tramos:
        resultado = remuestrear_terminal(tramo, N_PUNTOS)
        if resultado is not None:
            resultados.append(resultado)
        else:
            descartados_remuestreo += 1

    df_micro = pd.concat(resultados, ignore_index=True)
    n_vuelos = df_micro['flight_id'].nunique()

    # --- Guardado ---
    ruta_salida = os.path.join(RUTA_SALIDA, f"trayectorias_micro_{codigo}.parquet")
    df_micro.to_parquet(ruta_salida, index=False)

    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Aeropuerto: {codigo}")
    print(f"  Vuelos: {n_vuelos}")
    print(f"  Puntos por vuelo: {N_PUNTOS}")
    print(f"  Guardado en: {ruta_salida}")
    print(f"  Completado en {time.time() - inicio:.1f} segundos")