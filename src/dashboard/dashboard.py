import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# === CONFIGURACIÓN ===
RUTA_TRAYECTORIAS = "../../resultados/preparacion/trayectorias_normalizadas.parquet"
RUTA_CLUSTERS = "../../resultados/macro/clusters_macro.parquet"
RUTA_METADATOS = "../../datos/flight_list.csv"

# === CARGA DE DATOS ===
print("Cargando datos...")
df_tray = pd.read_parquet(RUTA_TRAYECTORIAS)
df_clusters = pd.read_parquet(RUTA_CLUSTERS)
df_meta = pd.read_csv(RUTA_METADATOS)

# Unir trayectorias con clusters
df = df_tray.merge(df_clusters, on='flight_id', how='inner')

# Unir con metadatos (aeropuertos)
df = df.merge(
    df_meta[['flight_id', 'adep', 'name_adep', 'ades', 'name_ades', 'aircraft_type', 'airline']],
    on='flight_id',
    how='left'
)

# Ordenar por point_index dentro de cada vuelo
df = df.sort_values(['flight_id', 'point_index'])

# Lista de clusters
clusters_validos = sorted(df[df['cluster'] >= 0]['cluster'].unique())
n_ruido = df[df['cluster'] == -1]['flight_id'].nunique()

# Paleta de colores (20 colores distinguibles)
COLORES = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]
COLOR_RUIDO = '#d3d3d3'

print(f"Datos cargados: {df['flight_id'].nunique()} vuelos, {len(clusters_validos)} clusters, {n_ruido} outliers")


def crear_figura_global(clusters_seleccionados, mostrar_ruido):
    """
    Genera el mapa con las trayectorias coloreadas por cluster.
    """
    fig = go.Figure()

    # Ruido de fondo
    if mostrar_ruido:
        df_ruido = df[df['cluster'] == -1]
        for fid in df_ruido['flight_id'].unique():
            vuelo = df_ruido[df_ruido['flight_id'] == fid]
            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=COLOR_RUIDO, width=0.5),
                opacity=0.3,
                showlegend=False,
                hoverinfo='skip'
            ))

    # Clusters seleccionados
    for cluster_id in clusters_seleccionados:
        df_c = df[df['cluster'] == cluster_id]
        color = COLORES[cluster_id % len(COLORES)]
        n_vuelos = df_c['flight_id'].nunique()

        primera_linea = True
        for fid in df_c['flight_id'].unique():
            vuelo = df_c[df_c['flight_id'] == fid]

            # Info para hover
            meta_vuelo = vuelo.iloc[0]
            adep = meta_vuelo.get('adep', '?')
            ades = meta_vuelo.get('ades', '?')
            aerolinea = meta_vuelo.get('airline', '?')

            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=color, width=1),
                opacity=0.6,
                name=f'Cluster {cluster_id} ({n_vuelos})' if primera_linea else None,
                showlegend=primera_linea,
                legendgroup=f'cluster_{cluster_id}',
                hovertemplate=(
                    f'<b>Vuelo:</b> {fid}<br>'
                    f'<b>Ruta:</b> {adep} → {ades}<br>'
                    f'<b>Aerolínea:</b> {aerolinea}<br>'
                    f'<b>Cluster:</b> {cluster_id}'
                    '<extra></extra>'
                )
            ))
            primera_linea = False

    fig.update_layout(
        title=f'Tráfico aéreo — {len(clusters_seleccionados)} clusters seleccionados',
        xaxis_title='X (metros, LCC)',
        yaxis_title='Y (metros, LCC)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        plot_bgcolor='white',
        height=700,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=60)
    )

    return fig


def crear_figura_3d(cluster_id):
    """
    Genera una vista 3D (x, y, altitud) de un cluster individual.
    """
    fig = go.Figure()

    df_c = df[df['cluster'] == cluster_id]
    color = COLORES[cluster_id % len(COLORES)]

    for fid in df_c['flight_id'].unique():
        vuelo = df_c[df_c['flight_id'] == fid]
        meta_vuelo = vuelo.iloc[0]
        adep = meta_vuelo.get('adep', '?')
        ades = meta_vuelo.get('ades', '?')

        fig.add_trace(go.Scatter3d(
            x=vuelo['x'], y=vuelo['y'], z=vuelo['altitude'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.6,
            showlegend=False,
            hovertemplate=(
                f'<b>{fid}</b><br>'
                f'{adep} → {ades}<br>'
                'Alt: %{z:.0f} m'
                '<extra></extra>'
            )
        ))

    n_vuelos = df_c['flight_id'].nunique()
    fig.update_layout(
        title=f'Cluster {cluster_id} — {n_vuelos} vuelos (vista 3D)',
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Altitud (m)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def crear_tabla_resumen(cluster_id):
    """
    Genera una tabla HTML con el resumen del cluster.
    """
    df_c = df[df['cluster'] == cluster_id]
    vuelos = df_c['flight_id'].unique()
    n_vuelos = len(vuelos)

    # Rutas
    rutas = df_c.groupby('flight_id')[['adep', 'ades']].first()
    rutas['ruta'] = rutas['adep'].fillna('?') + ' → ' + rutas['ades'].fillna('?')
    top_rutas = rutas['ruta'].value_counts().head(5)

    # Altitud de crucero
    alt_max = df_c.groupby('flight_id')['altitude'].max()

    # Distancias
    distancias = []
    for fid in vuelos:
        v = df_c[df_c['flight_id'] == fid]
        dx = np.diff(v['x'].values)
        dy = np.diff(v['y'].values)
        distancias.append(np.sum(np.sqrt(dx**2 + dy**2)) / 1000)

    resumen = html.Div([
        html.H4(f'Cluster {cluster_id} — {n_vuelos} vuelos'),
        html.Table([
            html.Tr([html.Td('Distancia media'), html.Td(f'{np.mean(distancias):.0f} km')]),
            html.Tr([html.Td('Distancia rango'), html.Td(f'{np.min(distancias):.0f} – {np.max(distancias):.0f} km')]),
            html.Tr([html.Td('Alt. crucero media'), html.Td(f'{alt_max.mean():.0f} m')]),
            html.Tr([html.Td('Alt. crucero rango'), html.Td(f'{alt_max.min():.0f} – {alt_max.max():.0f} m')]),
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '10px'}),
        html.H5('Rutas más frecuentes'),
        html.Ul([html.Li(f'{ruta}: {count} vuelos') for ruta, count in top_rutas.items()])
    ])

    return resumen


# === APLICACIÓN DASH ===
app = dash.Dash(__name__)

app.layout = html.Div([
    # Título
    html.H1('Dashboard — Clustering de trayectorias aéreas',
            style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginBottom': '5px'}),
    html.P(f'{df["flight_id"].nunique()} vuelos · {len(clusters_validos)} clusters · {n_ruido} outliers',
           style={'textAlign': 'center', 'color': '#666', 'marginTop': '0'}),

    # Controles
    html.Div([
        # Selector de clusters
        html.Div([
            html.Label('Clusters a mostrar:', style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='selector-clusters',
                options=[{'label': f' Cluster {c}', 'value': c} for c in clusters_validos],
                value=clusters_validos,
                inline=True,
                style={'fontSize': '13px'}
            ),
        ], style={'flex': '3', 'marginRight': '20px'}),

        # Mostrar ruido
        html.Div([
            html.Label('Opciones:', style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='mostrar-ruido',
                options=[{'label': ' Mostrar outliers', 'value': 'ruido'}],
                value=['ruido'],
                style={'fontSize': '13px'}
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'padding': '10px 20px', 'backgroundColor': '#f5f5f5',
              'borderRadius': '5px', 'margin': '10px 20px'}),

    # Mapa global
    dcc.Graph(id='mapa-global', style={'margin': '0 20px'}),

    # Panel inferior: vista 3D + resumen
    html.Div([
        html.Div([
            html.Label('Inspeccionar cluster:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='selector-cluster-individual',
                options=[{'label': f'Cluster {c}', 'value': c} for c in clusters_validos],
                value=clusters_validos[0] if clusters_validos else None,
                style={'width': '200px'}
            ),
        ], style={'padding': '10px 20px'}),

        html.Div([
            # Vista 3D
            html.Div([
                dcc.Graph(id='vista-3d')
            ], style={'flex': '2'}),

            # Resumen
            html.Div(id='resumen-cluster',
                     style={'flex': '1', 'padding': '20px', 'fontFamily': 'Arial',
                            'fontSize': '14px', 'backgroundColor': '#f9f9f9',
                            'borderRadius': '5px', 'margin': '10px'})
        ], style={'display': 'flex', 'margin': '0 20px'})
    ])

], style={'fontFamily': 'Arial', 'maxWidth': '1400px', 'margin': '0 auto'})


# === CALLBACKS ===

@app.callback(
    Output('mapa-global', 'figure'),
    [Input('selector-clusters', 'value'),
     Input('mostrar-ruido', 'value')]
)
def actualizar_mapa(clusters_seleccionados, opciones_ruido):
    mostrar_ruido = 'ruido' in (opciones_ruido or [])
    return crear_figura_global(clusters_seleccionados or [], mostrar_ruido)


@app.callback(
    [Output('vista-3d', 'figure'),
     Output('resumen-cluster', 'children')],
    Input('selector-cluster-individual', 'value')
)
def actualizar_detalle(cluster_id):
    if cluster_id is None:
        return go.Figure(), html.P('Selecciona un cluster')
    return crear_figura_3d(cluster_id), crear_tabla_resumen(cluster_id)


# === EJECUCIÓN ===
if __name__ == '__main__':
    print("\nIniciando dashboard en http://127.0.0.1:8050")
    print("Pulsa Ctrl+C para detener\n")
    app.run(debug=False, port=8050)