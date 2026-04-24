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

# Info resumida por cluster para el dropdown
info_clusters = {}
for c in clusters_validos:
    df_c = df[df['cluster'] == c]
    n = df_c['flight_id'].nunique()
    # Ruta más frecuente
    rutas = df_c.groupby('flight_id')[['adep', 'name_adep', 'ades', 'name_ades']].first()
    rutas['ruta'] = rutas['adep'].fillna('?') + ' → ' + rutas['ades'].fillna('?')
    top_ruta = rutas['ruta'].value_counts().index[0] if len(rutas) > 0 else '?'
    # Distancia media
    dists = []
    for fid in df_c['flight_id'].unique():
        v = df_c[df_c['flight_id'] == fid]
        dx = np.diff(v['x'].values)
        dy = np.diff(v['y'].values)
        dists.append(np.sum(np.sqrt(dx**2 + dy**2)) / 1000)
    dist_media = np.mean(dists) if dists else 0
    info_clusters[c] = {'n_vuelos': n, 'top_ruta': top_ruta, 'dist_media_km': dist_media}

# Identificar clusters destacados
cluster_max = max(info_clusters, key=lambda c: info_clusters[c]['n_vuelos'])
cluster_min = min(info_clusters, key=lambda c: info_clusters[c]['n_vuelos'])
cluster_largo = max(info_clusters, key=lambda c: info_clusters[c]['dist_media_km'])
cluster_corto = min(info_clusters, key=lambda c: info_clusters[c]['dist_media_km'])

# Aeropuertos destino más frecuentes (global, todos los clusters)
destinos_global = df[df['cluster'] >= 0].groupby('flight_id')[['ades', 'name_ades']].first()
conteo_destinos = destinos_global['ades'].value_counts().head(10)
nombres_destinos = df[['ades', 'name_ades']].drop_duplicates().set_index('ades')['name_ades'].to_dict()

# Aeropuertos origen más frecuentes (global)
origenes_global = df[df['cluster'] >= 0].groupby('flight_id')[['adep', 'name_adep']].first()
conteo_origenes = origenes_global['adep'].value_counts().head(10)
nombres_origenes = df[['adep', 'name_adep']].drop_duplicates().set_index('adep')['name_adep'].to_dict()

# Panel de estadísticas globales
panel_estadisticas = html.Div([
    html.H3('Estadísticas globales', style={'marginBottom': '10px'}),
    html.Div([
        # Clusters destacados
        html.Div([
            html.H4('Clusters destacados', style={'marginTop': '0'}),
            html.Table([
                html.Tr([html.Td('Más vuelos'), html.Td(f'Cluster {cluster_max} ({info_clusters[cluster_max]["n_vuelos"]} vuelos)')]),
                html.Tr([html.Td('Menos vuelos'), html.Td(f'Cluster {cluster_min} ({info_clusters[cluster_min]["n_vuelos"]} vuelos)')]),
                html.Tr([html.Td('Más largo'), html.Td(f'Cluster {cluster_largo} ({info_clusters[cluster_largo]["dist_media_km"]:.0f} km)')]),
                html.Tr([html.Td('Más corto'), html.Td(f'Cluster {cluster_corto} ({info_clusters[cluster_corto]["dist_media_km"]:.0f} km)')]),
            ], style={'width': '100%', 'borderCollapse': 'collapse'}),
        ], style={'flex': '1', 'padding': '10px'}),

        # Top destinos
        html.Div([
            html.H4('Top 10 destinos', style={'marginTop': '0'}),
            html.Ol([
                html.Li(f'{code} ({nombres_destinos.get(code, "")}) — {count} vuelos')
                for code, count in conteo_destinos.items()
            ], style={'paddingLeft': '20px', 'margin': '0'}),
        ], style={'flex': '1', 'padding': '10px'}),

        # Top orígenes
        html.Div([
            html.H4('Top 10 orígenes', style={'marginTop': '0'}),
            html.Ol([
                html.Li(f'{code} ({nombres_origenes.get(code, "")}) — {count} vuelos')
                for code, count in conteo_origenes.items()
            ], style={'paddingLeft': '20px', 'margin': '0'}),
        ], style={'flex': '1', 'padding': '10px'}),
    ], style={'display': 'flex'}),
], style={'padding': '10px 20px', 'backgroundColor': '#f0f4f8',
          'borderRadius': '5px', 'margin': '10px 20px', 'fontFamily': 'Arial', 'fontSize': '13px'})

# Opciones del dropdown (simple)
opciones_dropdown = []
for c in clusters_validos:
    ic = info_clusters[c]
    label = f'Cluster {c} — {ic["n_vuelos"]} vuelos, {ic["dist_media_km"]:.0f} km'
    opciones_dropdown.append({'label': label, 'value': c})

# Lista de aeropuertos únicos (origen + destino) para el filtro
meta_por_vuelo = df.groupby('flight_id')[['adep', 'name_adep', 'ades', 'name_ades']].first()
aeropuertos = {}
for _, row in meta_por_vuelo.iterrows():
    if pd.notna(row['adep']):
        aeropuertos[row['adep']] = row.get('name_adep', '')
    if pd.notna(row['ades']):
        aeropuertos[row['ades']] = row.get('name_ades', '')
opciones_aeropuertos = sorted([
    {'label': f'{code} ({nombre})', 'value': code}
    for code, nombre in aeropuertos.items()
], key=lambda x: x['label'])

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
            name_adep = meta_vuelo.get('name_adep', '')
            name_ades = meta_vuelo.get('name_ades', '')
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
                    f'<b>Origen:</b> {adep} ({name_adep})<br>'
                    f'<b>Destino:</b> {ades} ({name_ades})<br>'
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
        name_adep = meta_vuelo.get('name_adep', '')
        name_ades = meta_vuelo.get('name_ades', '')

        fig.add_trace(go.Scatter3d(
            x=vuelo['x'], y=vuelo['y'], z=vuelo['altitude'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=0.6,
            showlegend=False,
            hovertemplate=(
                f'<b>{fid}</b><br>'
                f'{adep} ({name_adep}) → {ades} ({name_ades})<br>'
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
    rutas = df_c.groupby('flight_id')[['adep', 'name_adep', 'ades', 'name_ades']].first()
    rutas['ruta'] = (rutas['adep'].fillna('?') + ' (' + rutas['name_adep'].fillna('') + ') → ' +
                     rutas['ades'].fillna('?') + ' (' + rutas['name_ades'].fillna('') + ')')
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


def crear_figura_aeropuerto(codigo_aeropuerto, modo_filtro):
    """
    Genera un mapa con las trayectorias que tienen el aeropuerto
    seleccionado como origen, destino o ambos. Coloreadas por cluster,
    ruido en gris.
    """
    fig = go.Figure()

    # Filtrar vuelos por aeropuerto
    if modo_filtro == 'origen':
        ids_filtrados = df[df['adep'] == codigo_aeropuerto]['flight_id'].unique()
    elif modo_filtro == 'destino':
        ids_filtrados = df[df['ades'] == codigo_aeropuerto]['flight_id'].unique()
    else:
        ids_filtrados = df[(df['adep'] == codigo_aeropuerto) | (df['ades'] == codigo_aeropuerto)]['flight_id'].unique()

    df_filtrado = df[df['flight_id'].isin(ids_filtrados)]

    if len(df_filtrado) == 0:
        fig.update_layout(title='Sin vuelos para este aeropuerto', height=500)
        return fig, html.P('Sin datos')

    # Separar ruido y clusters
    df_ruido = df_filtrado[df_filtrado['cluster'] == -1]
    df_con_cluster = df_filtrado[df_filtrado['cluster'] >= 0]

    # Pintar ruido en gris
    for fid in df_ruido['flight_id'].unique():
        vuelo = df_ruido[df_ruido['flight_id'] == fid]
        meta_v = vuelo.iloc[0]
        fig.add_trace(go.Scattergl(
            x=vuelo['x'], y=vuelo['y'],
            mode='lines',
            line=dict(color=COLOR_RUIDO, width=1),
            opacity=0.4,
            showlegend=False,
            hovertemplate=(
                f'<b>{fid}</b><br>'
                f'{meta_v.get("adep", "?")} → {meta_v.get("ades", "?")}<br>'
                '<b>Ruido</b>'
                '<extra></extra>'
            )
        ))

    # Pintar clusters
    clusters_presentes = sorted(df_con_cluster['cluster'].unique())
    for cluster_id in clusters_presentes:
        df_c = df_con_cluster[df_con_cluster['cluster'] == cluster_id]
        color = COLORES[cluster_id % len(COLORES)]
        n_vuelos_c = df_c['flight_id'].nunique()
        primera = True
        for fid in df_c['flight_id'].unique():
            vuelo = df_c[df_c['flight_id'] == fid]
            meta_v = vuelo.iloc[0]
            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=color, width=1.2),
                opacity=0.7,
                name=f'Cluster {cluster_id} ({n_vuelos_c})' if primera else None,
                showlegend=primera,
                legendgroup=f'aero_cluster_{cluster_id}',
                hovertemplate=(
                    f'<b>{fid}</b><br>'
                    f'{meta_v.get("adep", "?")} ({meta_v.get("name_adep", "")}) → '
                    f'{meta_v.get("ades", "?")} ({meta_v.get("name_ades", "")})<br>'
                    f'<b>Cluster {cluster_id}</b>'
                    '<extra></extra>'
                )
            ))
            primera = False

    n_total = len(ids_filtrados)
    n_cluster = df_con_cluster['flight_id'].nunique()
    n_ruido_aero = df_ruido['flight_id'].nunique()
    nombre = aeropuertos.get(codigo_aeropuerto, '')

    fig.update_layout(
        title=f'{codigo_aeropuerto} ({nombre}) — {n_total} vuelos ({n_cluster} en clusters, {n_ruido_aero} ruido)',
        xaxis_title='X (metros, LCC)',
        yaxis_title='Y (metros, LCC)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        plot_bgcolor='white',
        height=600,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=60)
    )

    # Resumen del aeropuerto
    pct_ruido = n_ruido_aero / n_total * 100 if n_total > 0 else 0

    # Tabla de rutas desglosada
    rutas_info = df_filtrado.groupby('flight_id')[['adep', 'name_adep', 'ades', 'name_ades', 'cluster']].first()
    rutas_info['ruta'] = rutas_info['adep'].fillna('?') + ' → ' + rutas_info['ades'].fillna('?')

    # Agrupar por ruta y cluster
    filas_tabla = []
    for ruta, grupo_ruta in rutas_info.groupby('ruta'):
        for cluster_id, grupo_cluster in grupo_ruta.groupby('cluster'):
            n = len(grupo_cluster)
            if cluster_id == -1:
                etiqueta = 'Ruido'
            else:
                etiqueta = f'Cluster {cluster_id}'
            filas_tabla.append({'ruta': ruta, 'cluster': etiqueta, 'cluster_id': cluster_id, 'n_vuelos': n})

    filas_tabla.sort(key=lambda x: (-x['n_vuelos']))

    tabla_rutas = html.Table(
        [html.Tr([
            html.Th('Ruta', style={'textAlign': 'left', 'padding': '4px 8px', 'borderBottom': '2px solid #ccc'}),
            html.Th('Cluster', style={'textAlign': 'left', 'padding': '4px 8px', 'borderBottom': '2px solid #ccc'}),
            html.Th('Vuelos', style={'textAlign': 'right', 'padding': '4px 8px', 'borderBottom': '2px solid #ccc'}),
        ])] +
        [html.Tr([
            html.Td(f['ruta'], style={'padding': '3px 8px', 'borderBottom': '1px solid #eee'}),
            html.Td(f['cluster'], style={'padding': '3px 8px', 'borderBottom': '1px solid #eee',
                                         'color': '#999' if f['cluster_id'] == -1 else 'inherit'}),
            html.Td(f'{f["n_vuelos"]}', style={'textAlign': 'right', 'padding': '3px 8px', 'borderBottom': '1px solid #eee'}),
        ]) for f in filas_tabla],
        style={'width': '100%', 'borderCollapse': 'collapse', 'fontSize': '12px'}
    )

    resumen_aero = html.Div([
        html.H4(f'{codigo_aeropuerto} ({nombre})'),
        html.Table([
            html.Tr([html.Td('Vuelos totales'), html.Td(f'{n_total}')]),
            html.Tr([html.Td('En clusters'), html.Td(f'{n_cluster} ({100 - pct_ruido:.1f}%)')]),
            html.Tr([html.Td('Ruido'), html.Td(f'{n_ruido_aero} ({pct_ruido:.1f}%)')]),
            html.Tr([html.Td('Clusters distintos'), html.Td(f'{len(clusters_presentes)}')]),
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '15px'}),
        html.H5('Desglose por ruta'),
        html.Div(tabla_rutas, style={'maxHeight': '400px', 'overflowY': 'auto'}),
    ], style={'fontSize': '13px'})

    return fig, resumen_aero


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
                value=[],
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
                value=[],
                style={'fontSize': '13px'}
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex', 'padding': '10px 20px', 'backgroundColor': '#f5f5f5',
              'borderRadius': '5px', 'margin': '10px 20px'}),

    # Mapa global
    dcc.Graph(id='mapa-global', style={'margin': '0 20px'}),

    # Panel de estadísticas globales
    panel_estadisticas,

    # Panel inferior: vista 3D + resumen
    html.Div([
        html.Div([
            html.Label('Inspeccionar cluster:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='selector-cluster-individual',
                options=opciones_dropdown,
                value=clusters_validos[0] if clusters_validos else None,
                style={'width': '600px'}
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
    ]),

    # Separador
    html.Hr(style={'margin': '20px'}),

    # Filtro por aeropuerto
    html.Div([
        html.H2('Filtrar por aeropuerto', style={'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.Label('Aeropuerto:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='selector-aeropuerto',
                    options=opciones_aeropuertos,
                    value=None,
                    placeholder='Selecciona un aeropuerto...',
                    style={'width': '400px'}
                ),
            ], style={'marginRight': '20px'}),
            html.Div([
                html.Label('Mostrar como:', style={'fontWeight': 'bold'}),
                dcc.RadioItems(
                    id='modo-filtro-aeropuerto',
                    options=[
                        {'label': ' Origen', 'value': 'origen'},
                        {'label': ' Destino', 'value': 'destino'},
                        {'label': ' Ambos', 'value': 'ambos'},
                    ],
                    value='ambos',
                    inline=True,
                    style={'fontSize': '13px'}
                ),
            ]),
        ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginBottom': '10px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='mapa-aeropuerto')
            ], style={'flex': '3'}),
            html.Div(id='resumen-aeropuerto',
                     style={'flex': '1', 'padding': '20px', 'fontFamily': 'Arial',
                            'fontSize': '14px', 'backgroundColor': '#f9f9f9',
                            'borderRadius': '5px', 'margin': '10px'})
        ], style={'display': 'flex'}),

        # Filtro por ruta específica
        html.Div([
            html.Hr(style={'margin': '15px 0'}),
            html.Div([
                html.Div([
                    html.Label('Filtrar por ruta:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='selector-ruta',
                        options=[],
                        value=None,
                        placeholder='Selecciona primero un aeropuerto...',
                        style={'width': '500px'}
                    ),
                ]),
            ]),
            html.Div([
                html.Div([
                    dcc.Graph(id='mapa-ruta')
                ], style={'flex': '3'}),
                html.Div(id='resumen-ruta',
                         style={'flex': '1', 'padding': '20px', 'fontFamily': 'Arial',
                                'fontSize': '14px', 'backgroundColor': '#f9f9f9',
                                'borderRadius': '5px', 'margin': '10px'})
            ], style={'display': 'flex'}),
        ]),

    ], style={'padding': '0 20px'}),

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


@app.callback(
    [Output('mapa-aeropuerto', 'figure'),
     Output('resumen-aeropuerto', 'children')],
    [Input('selector-aeropuerto', 'value'),
     Input('modo-filtro-aeropuerto', 'value')]
)
def actualizar_aeropuerto(codigo, modo):
    if codigo is None:
        return go.Figure(), html.P('Selecciona un aeropuerto')
    return crear_figura_aeropuerto(codigo, modo)


@app.callback(
    [Output('selector-ruta', 'options'),
     Output('selector-ruta', 'value'),
     Output('selector-ruta', 'placeholder')],
    [Input('selector-aeropuerto', 'value'),
     Input('modo-filtro-aeropuerto', 'value')]
)
def actualizar_opciones_ruta(codigo, modo):
    if codigo is None:
        return [], None, 'Selecciona primero un aeropuerto...'

    # Filtrar vuelos por aeropuerto
    if modo == 'origen':
        ids = df[df['adep'] == codigo]['flight_id'].unique()
    elif modo == 'destino':
        ids = df[df['ades'] == codigo]['flight_id'].unique()
    else:
        ids = df[(df['adep'] == codigo) | (df['ades'] == codigo)]['flight_id'].unique()

    df_f = df[df['flight_id'].isin(ids)]
    rutas = df_f.groupby('flight_id')[['adep', 'ades']].first()
    rutas['ruta'] = rutas['adep'].fillna('?') + ' → ' + rutas['ades'].fillna('?')
    conteo = rutas['ruta'].value_counts()

    opciones = [{'label': f'{ruta} ({count} vuelos)', 'value': ruta} for ruta, count in conteo.items()]
    return opciones, None, 'Selecciona una ruta...'


@app.callback(
    [Output('mapa-ruta', 'figure'),
     Output('resumen-ruta', 'children')],
    [Input('selector-ruta', 'value'),
     Input('selector-aeropuerto', 'value'),
     Input('modo-filtro-aeropuerto', 'value')]
)
def actualizar_mapa_ruta(ruta, codigo_aeropuerto, modo):
    if ruta is None or codigo_aeropuerto is None:
        return go.Figure(), html.P('Selecciona una ruta')

    # Parsear ruta
    partes = ruta.split(' → ')
    if len(partes) != 2:
        return go.Figure(), html.P('Ruta no válida')
    adep_filtro, ades_filtro = partes

    # Filtrar vuelos de esta ruta exacta
    meta_vuelos = df.groupby('flight_id')[['adep', 'ades', 'name_adep', 'name_ades', 'cluster', 'airline']].first()
    ids_ruta = meta_vuelos[(meta_vuelos['adep'] == adep_filtro) & (meta_vuelos['ades'] == ades_filtro)].index
    df_ruta = df[df['flight_id'].isin(ids_ruta)]

    if len(df_ruta) == 0:
        return go.Figure(), html.P('Sin vuelos para esta ruta')

    fig = go.Figure()

    df_ruido = df_ruta[df_ruta['cluster'] == -1]
    df_con_cluster = df_ruta[df_ruta['cluster'] >= 0]

    # Pintar ruido
    n_ruido_r = df_ruido['flight_id'].nunique()
    primera_ruido = True
    for fid in df_ruido['flight_id'].unique():
        vuelo = df_ruido[df_ruido['flight_id'] == fid]
        meta_v = vuelo.iloc[0]
        fig.add_trace(go.Scattergl(
            x=vuelo['x'], y=vuelo['y'],
            mode='lines',
            line=dict(color=COLOR_RUIDO, width=1.5),
            opacity=0.5,
            name=f'Ruido ({n_ruido_r})' if primera_ruido else None,
            showlegend=primera_ruido,
            legendgroup='ruta_ruido',
            hovertemplate=(
                f'<b>{fid}</b><br>'
                f'Aerolínea: {meta_v.get("airline", "?")}<br>'
                '<b>Ruido</b>'
                '<extra></extra>'
            )
        ))
        primera_ruido = False

    # Pintar clusters
    clusters_ruta = sorted(df_con_cluster['cluster'].unique())
    for cluster_id in clusters_ruta:
        df_c = df_con_cluster[df_con_cluster['cluster'] == cluster_id]
        color = COLORES[cluster_id % len(COLORES)]
        n_c = df_c['flight_id'].nunique()
        primera = True
        for fid in df_c['flight_id'].unique():
            vuelo = df_c[df_c['flight_id'] == fid]
            meta_v = vuelo.iloc[0]
            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=color, width=1.5),
                opacity=0.7,
                name=f'Cluster {cluster_id} ({n_c})' if primera else None,
                showlegend=primera,
                legendgroup=f'ruta_cluster_{cluster_id}',
                hovertemplate=(
                    f'<b>{fid}</b><br>'
                    f'Aerolínea: {meta_v.get("airline", "?")}<br>'
                    f'<b>Cluster {cluster_id}</b>'
                    '<extra></extra>'
                )
            ))
            primera = False

    n_total_ruta = len(ids_ruta)
    n_cluster_ruta = df_con_cluster['flight_id'].nunique()
    name_adep = aeropuertos.get(adep_filtro, '')
    name_ades = aeropuertos.get(ades_filtro, '')

    fig.update_layout(
        title=f'{adep_filtro} ({name_adep}) → {ades_filtro} ({name_ades}) — {n_total_ruta} vuelos',
        xaxis_title='X (metros, LCC)',
        yaxis_title='Y (metros, LCC)',
        xaxis=dict(scaleanchor='y', scaleratio=1),
        plot_bgcolor='white',
        height=500,
        legend=dict(font=dict(size=10)),
        margin=dict(l=60, r=20, t=50, b=60)
    )

    # Resumen
    pct_ruido_r = n_ruido_r / n_total_ruta * 100 if n_total_ruta > 0 else 0
    resumen_ruta = html.Div([
        html.H4(f'{adep_filtro} → {ades_filtro}'),
        html.P(f'{name_adep} → {name_ades}', style={'color': '#666', 'marginTop': '0'}),
        html.Table([
            html.Tr([html.Td('Vuelos totales'), html.Td(f'{n_total_ruta}')]),
            html.Tr([html.Td('En clusters'), html.Td(f'{n_cluster_ruta} ({100 - pct_ruido_r:.1f}%)')]),
            html.Tr([html.Td('Ruido'), html.Td(f'{n_ruido_r} ({pct_ruido_r:.1f}%)')]),
        ], style={'width': '100%', 'borderCollapse': 'collapse', 'marginBottom': '10px'}),
        html.H5('Clusters que contienen esta ruta'),
        html.Ul([html.Li(f'Cluster {c}: {df_con_cluster[df_con_cluster["cluster"] == c]["flight_id"].nunique()} vuelos')
                 for c in clusters_ruta]) if clusters_ruta else html.P('Todos son ruido'),
    ], style={'fontSize': '13px'})

    return fig, resumen_ruta


# === EJECUCIÓN ===
if __name__ == '__main__':
    print("\nIniciando dashboard en http://127.0.0.1:8050")
    print("Pulsa Ctrl+C para detener\n")
    app.run(debug=False, port=8050)