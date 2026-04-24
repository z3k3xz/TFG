import pandas as pd
import matplotlib.pyplot as plt

# 1. Configurar ruta 
#ruta_parquet = "../../datos/2022-01-01.parquet"
ruta_parquet = "../../resultados/trayectorias_remuestreadas.parquet"

print("Leyendo altitudes de todo el sector...")
# Leemos SOLO la columna de altitud para ir rápido
df_alt = pd.read_parquet(ruta_parquet, columns=['altitude'])

# 2. Crear el Histograma
plt.figure(figsize=(12, 6))
# Filtramos altitudes menores a 1000 pies (ruido en tierra) para ver el vuelo
plt.hist(df_alt[df_alt['altitude'] > 1000]['altitude'], bins=100, color='skyblue', edgecolor='black')

plt.title("Distribución Vertical del Tráfico Aéreo (Niveles de Vuelo)")
plt.xlabel("Altitud (Pies)")
plt.ylabel("Número de registros (puntos de trayectoria)")
plt.grid(axis='y', alpha=0.3)

plt.show()