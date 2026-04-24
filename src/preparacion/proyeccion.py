import pandas as pd
import numpy as np
from pyproj import Transformer
import time

#pip install pyproj

# === CONFIGURACIÓN ===
RUTA_ENTRADA = "../../resultados/preparacion/trayectorias_limpias.parquet"
RUTA_SALIDA = "../../resultados/preparacion/trayectorias_proyectadas.parquet"

# Proyección LCC centrada en Europa
# EPSG:4326 = WGS84 (lat/lon en grados)
# EPSG:3034 = ETRS89 / LCC Europe (x/y en metros)
CRS_ORIGEN = "EPSG:4326"
CRS_DESTINO = "EPSG:3034"

# Factor de conversión de pies a metros
PIES_A_METROS = 0.3048


def proyectar(df, transformer):
    """
    Transforma las coordenadas geográficas (lat/lon en grados, altitud en pies)
    a coordenadas cartesianas (x/y en metros, altitud en metros).

    La proyección LCC (Lambert Conformal Conic) preserva las formas locales
    y minimiza las distorsiones a escala continental. Tras la transformación,
    las distancias entre puntos se calculan directamente en metros.
    """
    # Transformar lat/lon a x/y en metros
    x, y = transformer.transform(df['latitude'].values, df['longitude'].values)

    # Convertir altitud de pies a metros
    alt_metros = df['altitude'].values * PIES_A_METROS

    df_proyectado = pd.DataFrame({
        'flight_id': df['flight_id'].values,
        'timestamp': df['timestamp'].values,
        'x': x,
        'y': y,
        'altitude': alt_metros
    })

    return df_proyectado


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" PROYECCIÓN LCC (WGS84 → ETRS89/LCC Europe)")
    print("=" * 60)
    inicio = time.time()

    # --- Carga ---
    print(f"Cargando: {RUTA_ENTRADA}")
    df = pd.read_parquet(RUTA_ENTRADA)
    n_vuelos = df['flight_id'].nunique()
    n_puntos = len(df)
    print(f"Cargados {n_puntos:,} puntos de {n_vuelos:,} vuelos")

    # --- Rango original ---
    print(f"\nCoordenadas originales (WGS84):")
    print(f"  Latitud:  [{df['latitude'].min():.4f}, {df['latitude'].max():.4f}] grados")
    print(f"  Longitud: [{df['longitude'].min():.4f}, {df['longitude'].max():.4f}] grados")
    print(f"  Altitud:  [{df['altitude'].min():.0f}, {df['altitude'].max():.0f}] pies")

    # --- Proyección ---
    print(f"\nProyectando {CRS_ORIGEN} → {CRS_DESTINO}...")
    transformer = Transformer.from_crs(CRS_ORIGEN, CRS_DESTINO, always_xy=False)
    df_proy = proyectar(df, transformer)

    # --- Rango proyectado ---
    print(f"\nCoordenadas proyectadas (LCC):")
    print(f"  X: [{df_proy['x'].min():.0f}, {df_proy['x'].max():.0f}] metros")
    print(f"  Y: [{df_proy['y'].min():.0f}, {df_proy['y'].max():.0f}] metros")
    print(f"  Altitud: [{df_proy['altitude'].min():.0f}, {df_proy['altitude'].max():.0f}] metros")

    # --- Verificación de NaN ---
    nans = df_proy[['x', 'y', 'altitude']].isna().sum().sum()
    if nans > 0:
        print(f"\nAVISO: {nans} valores NaN tras la proyección")
    else:
        print(f"\nSin valores NaN tras la proyección")

    # --- Guardado ---
    df_proy.to_parquet(RUTA_SALIDA, index=False)

    print(f"\n{'=' * 60}")
    print(f"RESULTADO")
    print(f"{'=' * 60}")
    print(f"  Vuelos: {n_vuelos:,}")
    print(f"  Puntos: {len(df_proy):,}")
    print(f"  Guardado en: {RUTA_SALIDA}")
    print(f"  Completado en {time.time() - inicio:.1f} segundos")