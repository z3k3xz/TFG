import pandas as pd
import numpy as np

#ruta_parquet = "../../datos/2022-01-01.parquet"
ruta_parquet = "../../resultados/trayectorias_remuestreadas.parquet"
df = pd.read_parquet(ruta_parquet, columns=['flight_id', 'latitude', 'longitude'])

# CALCULO DE RUIDO
def calcular_puntuacion_ruido(group):
    # Calculamos el salto entre coordenadas
    diff_lat = np.abs(np.diff(group['latitude']))
    diff_lon = np.abs(np.diff(group['longitude']))
    # El ruido es la desviación estándar de esos saltos
    return np.std(diff_lat) + np.std(diff_lon)

# TOMAMOS UNA MUESTRA (200)
vuelos_unicos = df['flight_id'].unique()[:200]
df_muestra = df[df['flight_id'].isin(vuelos_unicos)]

# Aplicamos el cálculo
puntuaciones = df_muestra.groupby('flight_id').apply(calcular_puntuacion_ruido)

# MOSTRAR 
top_ruido = puntuaciones.sort_values(ascending=False).head(10)

print("\n--- VUELOS CON MÁS RUIDO ---")
print(top_ruido)

# GUARDAR LA MUESTRA EN UN .TXT
top_ruido.to_csv("../../resultados/vuelos_ruidosos.txt", header=False)