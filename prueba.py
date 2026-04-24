import pandas as pd

# === CONFIGURACIÓN ===
RUTA_FUSIONADO = "resultados/preparacion/datos_fusionados.parquet"
RUTA_METADATOS = "datos/flight_list.csv"

# --- Carga ---
print("Cargando datos...")
df = pd.read_parquet(RUTA_FUSIONADO, columns=['flight_id'])
ids = df['flight_id'].unique()

meta = pd.read_csv(RUTA_METADATOS)
meta = meta[meta['flight_id'].isin(ids)]

n_vuelos = len(meta)
print(f"Vuelos con metadatos: {n_vuelos}")

# --- Top 10 destinos ---
print(f"\n{'=' * 50}")
print(f" TOP 10 DESTINOS")
print(f"{'=' * 50}")
destinos = meta['ades'].value_counts().head(10)
for i, (code, count) in enumerate(destinos.items(), 1):
    nombre = meta[meta['ades'] == code]['name_ades'].iloc[0]
    pct = count / n_vuelos * 100
    print(f"  {i:2d}. {code} ({nombre}) — {count} vuelos ({pct:.1f}%)")

# --- Top 10 orígenes ---
print(f"\n{'=' * 50}")
print(f" TOP 10 ORÍGENES")
print(f"{'=' * 50}")
origenes = meta['adep'].value_counts().head(10)
for i, (code, count) in enumerate(origenes.items(), 1):
    nombre = meta[meta['adep'] == code]['name_adep'].iloc[0]
    pct = count / n_vuelos * 100
    print(f"  {i:2d}. {code} ({nombre}) — {count} vuelos ({pct:.1f}%)")