import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# === CONFIGURACIÓN ===
RUTA_MICRO = "../../resultados/micro/"
RUTA_METADATOS = "../../datos/flight_list.csv"

# === DETECCIÓN DE AEROPUERTOS ANALIZADOS ===
archivos_clusters = [f for f in os.listdir(RUTA_MICRO) if f.startswith('clusters_micro_') and f.endswith('.parquet')]
aeropuertos_micro = [f.replace('clusters_micro_', '').replace('.parquet', '') for f in archivos_clusters]

if len(aeropuertos_micro) == 0:
    print("ERROR: No hay resultados micro. Ejecuta clustering_micro.py primero.")
    exit(1)

print(f"Aeropuertos con análisis micro: {aeropuertos_micro}")

# === CARGA DE METADATOS ===
df_meta = pd.read_csv(RUTA_METADATOS)

# Paleta de colores
COLORES = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]
COLOR_RUIDO = '#d3d3d3'


def cargar_datos_micro(codigo):
    """
    Carga las trayectorias y clusters micro de un aeropuerto.
    """
    ruta_tray = os.path.join(RUTA_MICRO, f"trayectorias_micro_{codigo}.parquet")
    ruta_clusters = os.path.join(RUTA_MICRO, f"clusters_micro_{codigo}.parquet")

    df_tray = pd.read_parquet(ruta_tray)
    df_clusters = pd.read_parquet(ruta_clusters)

    # Unir con clusters
    df = df_tray.merge(df_clusters, on='flight_id', how='inner')

    # Unir con metadatos
    df = df.merge(
        df_meta[['flight_id', 'adep', 'name_adep', 'ades', 'name_ades', 'airline']],
        on='flight_id', how='left'
    )

    df = df.sort_values(['flight_id', 'point_index'])
    return df


def crear_mapa_micro(df, codigo, clusters_seleccionados, mostrar_ruido):
    """
    Mapa 2D de las trayectorias en el área terminal.
    """
    fig = go.Figure()

    # Ruido
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

    # Clusters
    for cluster_id in clusters_seleccionados:
        df_c = df[df['cluster'] == cluster_id]
        color = COLORES[cluster_id % len(COLORES)]
        n_vuelos = df_c['flight_id'].nunique()

        primera = True
        for fid in df_c['flight_id'].unique():
            vuelo = df_c[df_c['flight_id'] == fid]
            meta_v = vuelo.iloc[0]

            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=color, width=1.2),
                opacity=0.6,
                name=f'Cluster {cluster_id} ({n_vuelos})' if primera else None,
                showlegend=primera,
                legendgroup=f'micro_{cluster_id}',
                hovertemplate=(
                    f'<b>Vuelo:</b> {fid}<br>'
                    f'<b>Origen:</b> {meta_v.get("adep", "?")} ({meta_v.get("name_adep", "")})<br>'
                    f'<b>Aerolínea:</b> {meta_v.get("airline", "?")}<br>'
                    f'<b>Cluster:</b> {cluster_id}'
                    '<extra></extra>'
                )
            ))
            primera = False

    n_total = df['flight_id'].nunique()
    n_ruido = df[df['cluster'] == -1]['flight_id'].nunique()

    # Rango fijo
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    margen_x = (x_max - x_min) * 0.05
    margen_y = (y_max - y_min) * 0.05

    fig.update_layout(
        title=f'Área terminal {codigo} — {len(clusters_seleccionados)} clusters, {n_ruido} outliers',
        xaxis_title='X (metros, LCC)',
        yaxis_title='Y (metros, LCC)',
        xaxis=dict(scaleanchor='y', scaleratio=1, range=[x_min - margen_x, x_max + margen_x]),
        yaxis=dict(range=[y_min - margen_y, y_max + margen_y]),
        plot_bgcolor='white',
        height=600,
        legend=dict(font=dict(size=9), itemclick='toggle', itemdoubleclick='toggleothers'),
        margin=dict(l=60, r=20, t=50, b=60)
    )

    return fig


def crear_vista_3d_micro(df, cluster_id):
    """
    Vista 3D de un cluster micro (x, y, altitud).
    """
    fig = go.Figure()
    df_c = df[df['cluster'] == cluster_id]
    color = COLORES[cluster_id % len(COLORES)]

    for fid in df_c['flight_id'].unique():
        vuelo = df_c[df_c['flight_id'] == fid]
        meta_v = vuelo.iloc[0]

        fig.add_trace(go.Scatter3d(
            x=vuelo['x'], y=vuelo['y'], z=vuelo['altitude'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.6,
            showlegend=False,
            hovertemplate=(
                f'<b>{fid}</b><br>'
                f'{meta_v.get("adep", "?")} ({meta_v.get("name_adep", "")})<br>'
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
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def crear_resumen_micro(df, cluster_id):
    """
    Tabla resumen de un cluster micro.
    """
    df_c = df[df['cluster'] == cluster_id]
    vuelos = df_c['flight_id'].unique()
    n_vuelos = len(vuelos)

    # Orígenes más frecuentes
    origenes = df_c.groupby('flight_id')[['adep', 'name_adep']].first()
    top_origenes = origenes['adep'].value_counts().head(5)

    # Altitud de entrada al TMA (primer punto)
    alt_entrada = df_c[df_c['point_index'] == 0].groupby('flight_id')['altitude'].first()

    resumen = html.Div([
        html.H4(f'Cluster {cluster_id} — {n_vuelos} vuelos'),
        html.Table([
            html.Tr([html.Td('Alt. entrada media'), html.Td(f'{alt_entrada.mean():.0f} m')]),
            html.Tr([html.Td('Alt. entrada rango'), html.Td(f'{alt_entrada.min():.0f} – {alt_entrada.max():.0f} m')]),
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '10px'}),
        html.H5('Orígenes más frecuentes'),
        html.Ul([
            html.Li(f'{code} ({origenes[origenes["adep"] == code]["name_adep"].iloc[0]}): {count}')
            for code, count in top_origenes.items()
        ])
    ])

    return resumen


# === APLICACIÓN DASH ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Dashboard Micro — Flujos en Área Terminal',
            style={'textAlign': 'center', 'fontFamily': 'Arial', 'marginBottom': '5px'}),

    # Selector de aeropuerto
    html.Div([
        html.Label('Aeropuerto analizado:', style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='selector-aeropuerto-micro',
            options=[{'label': c, 'value': c} for c in sorted(aeropuertos_micro)],
            value=aeropuertos_micro[0],
            style={'width': '300px'}
        ),
    ], style={'padding': '10px 20px'}),

    # Info general
    html.Div(id='info-general-micro',
             style={'padding': '5px 20px', 'color': '#666', 'fontFamily': 'Arial'}),

    # Controles de clusters
    html.Div([
        html.Div([
            html.Label('Clusters a mostrar:', style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='selector-clusters-micro',
                options=[],
                value=[],
                inline=True,
                style={'fontSize': '13px'}
            ),
        ], style={'flex': '3', 'marginRight': '20px'}),

        html.Div([
            dcc.Checklist(
                id='mostrar-ruido-micro',
                options=[{'label': ' Mostrar outliers', 'value': 'ruido'}],
                value=[],
                style={'fontSize': '13px'}
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'padding': '10px 20px', 'backgroundColor': '#f5f5f5',
              'borderRadius': '5px', 'margin': '10px 20px'}),

    # Mapa 2D
    dcc.Graph(id='mapa-micro', style={'margin': '0 20px'}),

    # Inspector de cluster
    html.Div([
        html.Div([
            html.Label('Inspeccionar cluster:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='selector-cluster-individual-micro',
                options=[],
                value=None,
                style={'width': '300px'}
            ),
        ], style={'padding': '10px 20px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='vista-3d-micro')
            ], style={'flex': '2'}),
            html.Div(id='resumen-micro',
                     style={'flex': '1', 'padding': '20px', 'fontFamily': 'Arial',
                            'fontSize': '14px', 'backgroundColor': '#f9f9f9',
                            'borderRadius': '5px', 'margin': '10px'})
        ], style={'display': 'flex', 'margin': '0 20px'})
    ]),

], style={'fontFamily': 'Arial', 'maxWidth': '1400px', 'margin': '0 auto'})


# === DATOS EN MEMORIA ===
datos_cache = {}


def obtener_datos(codigo):
    if codigo not in datos_cache:
        datos_cache[codigo] = cargar_datos_micro(codigo)
    return datos_cache[codigo]


# === CALLBACKS ===

@app.callback(
    [Output('info-general-micro', 'children'),
     Output('selector-clusters-micro', 'options'),
     Output('selector-clusters-micro', 'value'),
     Output('selector-cluster-individual-micro', 'options'),
     Output('selector-cluster-individual-micro', 'value')],
    Input('selector-aeropuerto-micro', 'value')
)
def actualizar_aeropuerto_micro(codigo):
    if codigo is None:
        return '', [], [], [], None

    df = obtener_datos(codigo)
    clusters = sorted(df[df['cluster'] >= 0]['cluster'].unique())
    n_vuelos = df['flight_id'].nunique()
    n_ruido = df[df['cluster'] == -1]['flight_id'].nunique()
    n_clusters = len(clusters)

    info = f'{n_vuelos} vuelos · {n_clusters} clusters · {n_ruido} outliers'

    opciones_check = [{'label': f' Cluster {c}', 'value': c} for c in clusters]
    opciones_drop = [{'label': f'Cluster {c}', 'value': c} for c in clusters]

    return info, opciones_check, [], opciones_drop, clusters[0] if clusters else None


@app.callback(
    Output('mapa-micro', 'figure'),
    [Input('selector-aeropuerto-micro', 'value'),
     Input('selector-clusters-micro', 'value'),
     Input('mostrar-ruido-micro', 'value')]
)
def actualizar_mapa_micro(codigo, clusters_sel, opciones_ruido):
    if codigo is None:
        return go.Figure()
    df = obtener_datos(codigo)
    mostrar_ruido = 'ruido' in (opciones_ruido or [])
    return crear_mapa_micro(df, codigo, clusters_sel or [], mostrar_ruido)


@app.callback(
    [Output('vista-3d-micro', 'figure'),
     Output('resumen-micro', 'children')],
    [Input('selector-aeropuerto-micro', 'value'),
     Input('selector-cluster-individual-micro', 'value')]
)
def actualizar_detalle_micro(codigo, cluster_id):
    if codigo is None or cluster_id is None:
        return go.Figure(), html.P('Selecciona un cluster')
    df = obtener_datos(codigo)
    return crear_vista_3d_micro(df, cluster_id), crear_resumen_micro(df, cluster_id)


# === EJECUCIÓN ===
if __name__ == '__main__':
    print("\nIniciando dashboard micro en http://127.0.0.1:8051")
    print("Pulsa Ctrl+C para detener\n")
    app.run(debug=False, port=8051)