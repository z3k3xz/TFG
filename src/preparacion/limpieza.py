import pandas as pd
import numpy as np
import time

# === CONFIGURACIÓN ===
RUTA_ENTRADA = "../../resultados/preparacion/datos_fusionados.parquet"
RUTA_SALIDA = "../../resultados/preparacion/trayectorias_limpias.parquet"

COLUMNAS = ['flight_id', 'timestamp', 'latitude', 'longitude', 'altitude',
            'groundspeed', 'vertical_rate']

# Umbrales ajustados para mayor precisión
UMBRAL_HUECO_MAX = 60   # max segundos sin senal antes de descartar el vuelo
VELOCIDAD_MAX_KT = 700  # ningún comercial pasa de esto en ground speed
CAMBIO_ALT_MAX = 100    # ft/s, equivale a ~6000ft/min
ALTITUD_MIN = -100      # margen para aeropuertos bajo nivel del mar
ALTITUD_MAX = 45000     # techo operativo de comerciales
MIN_PUNTOS = 600        # ~10 min de datos a 1pt/s

def limpiar_dataset(df):
    inicio_proceso = len(df)

    # 1. ORDENACIÓN CRÍTICA: Asegura que los cálculos de diff() sean correctos
    df = df.sort_values(['flight_id', 'timestamp']).reset_index(drop=True)

    # 2. CÁLCULO DE DIFERENCIAS TEMPORALES (por vuelo)
    # Transformamos a segundos para los cálculos
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()

    # FILTRO A: SALTOS DE POSICIÓN (Velocidad Implícita)
    dlat = df.groupby('flight_id')['latitude'].diff()
    dlon = df.groupby('flight_id')['longitude'].diff()
    cos_lat = np.cos(np.radians(df['latitude']))

    # Distancia en Millas Náuticas (Aproximación rápida)
    dist_nm = np.sqrt((dlat * 60)**2 + (dlon * 60 * cos_lat)**2)
    dt_horas = df['dt'] / 3600
    velocidad_calc = dist_nm / dt_horas

    # Mascara para saltos imposibles (el primer punto de cada vuelo es NaN -> True)
    mask_pos = (velocidad_calc.fillna(0) <= VELOCIDAD_MAX_KT)
    puntos_antes = len(df)
    df = df[mask_pos]
    print(f"  Puntos con salto de posición imposible: {puntos_antes - len(df):,}")

    # FILTRO B: ERRORES DE ALTITUD (Vertical Rate Real)
    # Recalculamos dt tras el primer filtro
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    dalt = df.groupby('flight_id')['altitude'].diff().abs()
    cambio_alt_seg = dalt / df['dt']

    mask_alt = (cambio_alt_seg.fillna(0) <= CAMBIO_ALT_MAX)
    puntos_antes = len(df)
    df = df[mask_alt]
    print(f"  Puntos con error de altitud (>100ft/s): {puntos_antes - len(df):,}")

    # FILTRO C: RANGO OPERATIVO
    puntos_antes = len(df)
    df = df[(df['altitude'] >= ALTITUD_MIN) & (df['altitude'] <= ALTITUD_MAX)]
    print(f"  Puntos fuera de rango [{ALTITUD_MIN}, {ALTITUD_MAX}] ft: {puntos_antes - len(df):,}")


    # FILTRO D: DESCARTE DE VUELOS COMPLETOS (Huecos y Longitud)
    # Huecos temporales
    df['dt'] = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds()
    vuelos_con_huecos = df.groupby('flight_id')['dt'].max()
    ids_validos_hueco = vuelos_con_huecos[vuelos_con_huecos <= UMBRAL_HUECO_MAX].index

    n_vuelos_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_validos_hueco)]
    print(f"  Vuelos descartados por hueco > {UMBRAL_HUECO_MAX}s: {n_vuelos_antes - df['flight_id'].nunique():,}")

    # Mínimo de puntos
    conteo = df.groupby('flight_id').size()
    ids_suficientes = conteo[conteo >= MIN_PUNTOS].index

    n_vuelos_antes = df['flight_id'].nunique()
    df = df[df['flight_id'].isin(ids_suficientes)]
    print(f"  Vuelos descartados por < {MIN_PUNTOS} puntos: {n_vuelos_antes - df['flight_id'].nunique():,}")

    # Limpieza de columnas auxiliares antes de devolver
    return df.drop(columns=['dt']).reset_index(drop=True)


# EJECUCIÓN
if __name__ == "__main__":
    print("=" * 60)
    print(" LIMPIEZA DE TRAYECTORIAS ADS-B")
    print("=" * 60)
    inicio_t = time.time()

    # Carga
    print(f"\nCargando: {RUTA_ENTRADA}")
    df_raw = pd.read_parquet(RUTA_ENTRADA)
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True)

    n_v_ini = df_raw['flight_id'].nunique()
    n_p_ini = len(df_raw)
    print(f"Cargados {n_p_ini:,} puntos de {n_v_ini:,} vuelos")

    # Limpieza
    print(f"\nEliminando datos erróneos...")
    df_clean = limpiar_dataset(df_raw)

    # Resultados
    n_v_fin = df_clean['flight_id'].nunique()
    n_p_fin = len(df_clean)

    print(f"\n{'=' * 60}")
    print(f"RESULTADO FINAL")
    print(f"{'=' * 60}")
    print(f"  Vuelos: {n_v_ini:,} -> {n_v_fin:,} (Eliminados: {n_v_ini - n_v_fin})")
    print(f"  Puntos: {n_p_ini:,} -> {n_p_fin:,} (Reducción: {n_p_ini/n_p_fin:.1f}x)")

    # Guardado
    df_clean.to_parquet(RUTA_SALIDA, index=False)
    print(f"\nGuardado en: {RUTA_SALIDA}")
    print(f"Completado en {time.time() - inicio_t:.1f} segundos")