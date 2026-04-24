import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar el dataset
df = pd.read_parquet("../../datos/2022-01-01.parquet")

# 2. Filtrar solo el vuelo deseado
FLIGHT_ID = 248754498
profile = df[df['flight_id'] == FLIGHT_ID].sort_values('timestamp')

lat = profile['latitude'].values
lon = profile['longitude'].values
alt = profile['altitude'].values
n = len(profile)

# Calculamos las diferencias
lat_diff = np.log(np.abs(np.diff(lat)) + 1)
lon_diff = np.log(np.abs(np.diff(lon)) + 1)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f"Análisis de Vuelo - ID: {FLIGHT_ID}", fontsize=16)

# --- FILA SUPERIOR ---
axs[0, 0].plot(lat, color='red', lw=0.5)
axs[0, 0].scatter(range(n), lat, s=1, color='black')
axs[0, 0].set_title("Perfil Latitud (N-S)")

axs[0, 1].plot(lon, color='red', lw=0.5)
axs[0, 1].scatter(range(n), lon, s=1, color='black')
axs[0, 1].set_title("Perfil Longitud (E-O)")

axs[0, 2].plot(lon, lat, color='red', lw=0.8)
axs[0, 2].scatter(lon, lat, s=1, color='black')
axs[0, 2].set_xlabel("Longitud")
axs[0, 2].set_ylabel("Latitud")
axs[0, 2].set_title("Trayectoria XY (Mapa Real)")

# --- FILA INFERIOR ---
jitter_lat = lat_diff + np.random.normal(0, 0.001, len(lat_diff))
jitter_lon = lon_diff + np.random.normal(0, 0.001, len(lon_diff))

axs[1, 0].plot(jitter_lon, jitter_lat, color='red', alpha=0.2, lw=0.5)
axs[1, 0].scatter(jitter_lon, jitter_lat, s=1, color='black')
axs[1, 0].set_xlabel("Error/Cambio en Longitud (E-O)")
axs[1, 0].set_ylabel("Error/Cambio en Latitud (N-S)")
axs[1, 0].set_title("Detector de Errores (Log-Diff)")

axs[1, 1].plot(alt, color='red', lw=0.5)
axs[1, 1].scatter(range(n), alt, s=1, color='black')
axs[1, 1].set_title("Perfil Altitud (Pies)")

axs[1, 2].axis('off')

plt.tight_layout()
plt.show()