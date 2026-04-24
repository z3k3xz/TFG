import pandas as pd
import os

RUTA_PARQUETS = "datos/parquet/"
RUTA_METADATOS = "datos/flight_list.csv"

# Cargar metadatos
meta = pd.read_csv(RUTA_METADATOS)
ids_meta = set(meta['flight_id'].unique())
print(f"flight_list.csv: {len(ids_meta)} flight_ids unicos")
print()

# Comprobar cada parquet
archivos = sorted([f for f in os.listdir(RUTA_PARQUETS) if f.endswith('.parquet')])

for f in archivos:
    ruta = os.path.join(RUTA_PARQUETS, f)
    pq = pd.read_parquet(ruta, columns=['flight_id'])
    ids_pq = set(pq['flight_id'].unique())
    coinciden = len(ids_pq & ids_meta)
    sin_match = len(ids_pq - ids_meta)
    print(f"{f}: {len(ids_pq)} vuelos, {coinciden} con metadatos, {sin_match} sin metadatos")