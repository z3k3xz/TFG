import pandas as pd
import os
import time

# === CONFIGURACIÓN ===
RUTA_PARQUETS = "../../datos/parquet/"
RUTA_METADATOS = "../../datos/flight_list.csv"
RUTA_SALIDA = "../../resultados/preparacion/datos_fusionados.parquet"

# Columnas que nos interesan del parquet de trayectorias
COLUMNAS_TRAYECTORIAS = [
    'flight_id', 'timestamp', 'latitude', 'longitude',
    'altitude', 'groundspeed', 'vertical_rate'
]

def fusionar_parquets(ruta_directorio, columnas):
    """
    Lee todos los ficheros .parquet de un directorio y los concatena
    en un unico DataFrame, conservando solo las columnas especificadas.
    """
    archivos = sorted([
        f for f in os.listdir(ruta_directorio) if f.endswith('.parquet')
    ])

    if len(archivos) == 0:
        print(f"ERROR: No se encontraron ficheros .parquet en {ruta_directorio}")
        return None

    print(f"Encontrados {len(archivos)} ficheros .parquet:")
    for f in archivos:
        print(f"  - {f}")

    # Lectura y concatenacion
    dfs = []
    for f in archivos:
        ruta = os.path.join(ruta_directorio, f)
        df = pd.read_parquet(ruta, columns=columnas)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        print(f"  {f}: {df['flight_id'].nunique()} vuelos, {len(df):,} puntos")
        dfs.append(df)

    df_total = pd.concat(dfs, ignore_index=True)
    return df_total


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" FUSION DE FICHEROS PARQUET")
    print("=" * 60)
    inicio = time.time()

    # --- Crear directorio de salida si no existe ---
    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)

    # --- Fusion de trayectorias ---
    print(f"\nLeyendo ficheros de: {RUTA_PARQUETS}")
    df_fusionado = fusionar_parquets(RUTA_PARQUETS, COLUMNAS_TRAYECTORIAS)

    if df_fusionado is None:
        exit(1)

    # --- Cruce con metadatos ---
    print(f"\nCargando metadatos de: {RUTA_METADATOS}")
    df_meta = pd.read_csv(RUTA_METADATOS)

    # Filtramos metadatos solo para los flight_id que tenemos
    ids_presentes = df_fusionado['flight_id'].unique()
    df_meta_filtrado = df_meta[df_meta['flight_id'].isin(ids_presentes)]

    print(f"Metadatos encontrados para {len(df_meta_filtrado)} de {len(ids_presentes)} vuelos")

    # Verificamos vuelos sin metadatos
    ids_con_meta = set(df_meta_filtrado['flight_id'].unique())
    ids_sin_meta = set(ids_presentes) - ids_con_meta
    if len(ids_sin_meta) > 0:
        print(f"  Vuelos sin metadatos: {len(ids_sin_meta)} (se descartan)")

    # Filtrar: conservar solo vuelos con metadatos
    n_antes = df_fusionado['flight_id'].nunique()
    df_fusionado = df_fusionado[df_fusionado['flight_id'].isin(ids_con_meta)]
    n_despues = df_fusionado['flight_id'].nunique()
    print(f"  Vuelos tras filtrado: {n_antes} -> {n_despues}")

    # --- Resumen ---
    n_vuelos = df_fusionado['flight_id'].nunique()
    n_puntos = len(df_fusionado)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTADO DE LA FUSION")
    print(f"{'=' * 60}")
    print(f"  Ficheros procesados: {len(os.listdir(RUTA_PARQUETS))}")
    print(f"  Vuelos totales:      {n_vuelos:,}")
    print(f"  Puntos totales:      {n_puntos:,}")
    print(f"  Rango temporal:      {df_fusionado['timestamp'].min()} - {df_fusionado['timestamp'].max()}")

    # --- Guardado ---
    df_fusionado.to_parquet(RUTA_SALIDA, index=False)
    print(f"\nGuardado en: {RUTA_SALIDA}")
    print(f"Completado en {time.time() - inicio:.1f} segundos")