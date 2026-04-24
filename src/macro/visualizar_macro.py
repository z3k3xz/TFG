import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# === CONFIGURACIÓN ===
RUTA_TRAYECTORIAS = "../../resultados/preparacion/trayectorias_proyectadas.parquet"
RUTA_CLUSTERS = "../../resultados/macro/clusters_macro.parquet"


def dibujar_mapa_global(df, df_clusters):
    """
    Dibuja todas las trayectorias coloreadas por cluster.
    Los outliers (cluster -1) se pintan en gris claro de fondo.
    """
    # Unir trayectorias con asignación de clusters
    df = df.merge(df_clusters, on='flight_id', how='inner')

    # Separar ruido y clusters
    df_ruido = df[df['cluster'] == -1]
    df_clusters_validos = df[df['cluster'] >= 0]

    n_clusters = df_clusters_validos['cluster'].nunique()

    # Paleta de colores
    colores = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Pintar ruido de fondo en gris
    for fid, grupo in df_ruido.groupby('flight_id'):
        grupo = grupo.sort_values('timestamp')
        ax.plot(grupo['x'], grupo['y'], color='lightgray', alpha=0.3, linewidth=0.5)

    # Pintar clusters
    for cluster_id in sorted(df_clusters_validos['cluster'].unique()):
        df_c = df_clusters_validos[df_clusters_validos['cluster'] == cluster_id]
        color = colores[cluster_id % len(colores)]
        for fid, grupo in df_c.groupby('flight_id'):
            grupo = grupo.sort_values('timestamp')
            ax.plot(grupo['x'], grupo['y'], color=color, alpha=0.5, linewidth=0.8)
        # Etiqueta en la leyenda (solo una línea por cluster)
        ax.plot([], [], color=color, linewidth=2, label=f'Cluster {cluster_id} ({len(df_c["flight_id"].unique())} vuelos)')

    n_ruido = df_ruido['flight_id'].nunique()
    ax.plot([], [], color='lightgray', linewidth=2, label=f'Ruido ({n_ruido} vuelos)')

    ax.set_xlabel('X (metros, LCC)')
    ax.set_ylabel('Y (metros, LCC)')
    ax.set_title(f'Clustering HDBSCAN — {n_clusters} clusters, {n_ruido} outliers')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def dibujar_clusters_individuales(df, df_clusters):
    """
    Muestra un mapa individual por cada cluster, mostrando solo
    las trayectorias de ese cluster sobre el fondo de ruido.
    """
    df = df.merge(df_clusters, on='flight_id', how='inner')

    df_ruido = df[df['cluster'] == -1]
    clusters_validos = sorted(df[df['cluster'] >= 0]['cluster'].unique())
    colores = cm.tab20(np.linspace(0, 1, max(len(clusters_validos), 1)))

    for i, cluster_id in enumerate(clusters_validos):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        # Fondo: ruido en gris
        for fid, grupo in df_ruido.groupby('flight_id'):
            grupo = grupo.sort_values('timestamp')
            ax.plot(grupo['x'], grupo['y'], color='lightgray', alpha=0.2, linewidth=0.3)

        # Cluster actual
        df_c = df[df['cluster'] == cluster_id]
        n_vuelos_cluster = df_c['flight_id'].nunique()
        color = colores[i % len(colores)]

        for fid, grupo in df_c.groupby('flight_id'):
            grupo = grupo.sort_values('timestamp')
            ax.plot(grupo['x'], grupo['y'], color=color, alpha=0.6, linewidth=1.0)

        ax.set_xlabel('X (metros, LCC)')
        ax.set_ylabel('Y (metros, LCC)')
        ax.set_title(f'Cluster {cluster_id} — {n_vuelos_cluster} vuelos')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("=" * 60)
    print(" VISUALIZACIÓN DE CLUSTERS (MACRO)")
    print("=" * 60)
    inicio = time.time()

    # --- Carga ---
    print(f"Cargando trayectorias: {RUTA_TRAYECTORIAS}")
    df = pd.read_parquet(RUTA_TRAYECTORIAS)
    print(f"Cargando clusters: {RUTA_CLUSTERS}")
    df_clusters = pd.read_parquet(RUTA_CLUSTERS)

    n_vuelos = df_clusters['flight_id'].nunique()
    n_clusters = df_clusters[df_clusters['cluster'] >= 0]['cluster'].nunique()
    n_ruido = df_clusters[df_clusters['cluster'] == -1].shape[0]
    print(f"{n_vuelos} vuelos, {n_clusters} clusters, {n_ruido} outliers")

    # --- Mapa global ---
    print(f"\nMostrando mapa global...")
    dibujar_mapa_global(df, df_clusters)

    # --- Mapas individuales ---
    print(f"Mostrando mapas individuales por cluster...")
    dibujar_clusters_individuales(df, df_clusters)

    print(f"\nCompletado en {time.time() - inicio:.1f} segundos")