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
    seleccionado como origen, destino o ambos. Agrupadas por ruta
    en la leyenda para poder activar/desactivar cada una.
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

    # Obtener ruta de cada vuelo
    meta_vuelos = df_filtrado.groupby('flight_id')[['adep', 'name_adep', 'ades', 'name_ades', 'cluster', 'airline']].first()
    meta_vuelos['ruta'] = meta_vuelos['adep'].fillna('?') + ' → ' + meta_vuelos['ades'].fillna('?')

    # Ordenar rutas por frecuencia
    conteo_rutas = meta_vuelos['ruta'].value_counts()

    # Paleta de colores por ruta
    rutas_unicas = conteo_rutas.index.tolist()
    colores_ruta = {}
    for i, ruta in enumerate(rutas_unicas):
        colores_ruta[ruta] = COLORES[i % len(COLORES)]

    # Pintar por ruta
    for ruta in rutas_unicas:
        ids_ruta = meta_vuelos[meta_vuelos['ruta'] == ruta].index
        n_vuelos_ruta = len(ids_ruta)
        color = colores_ruta[ruta]

        # Contar cuántos son cluster y cuántos ruido
        n_cluster_r = meta_vuelos.loc[ids_ruta, 'cluster'].apply(lambda x: x >= 0).sum()
        n_ruido_r = n_vuelos_ruta - n_cluster_r

        # Etiqueta de leyenda
        if n_ruido_r == n_vuelos_ruta:
            label = f'{ruta} ({n_vuelos_ruta}) [ruido]'
        elif n_ruido_r > 0:
            label = f'{ruta} ({n_vuelos_ruta}, {n_ruido_r} ruido)'
        else:
            label = f'{ruta} ({n_vuelos_ruta})'

        primera = True
        for fid in ids_ruta:
            vuelo = df_filtrado[df_filtrado['flight_id'] == fid]
            meta_v = meta_vuelos.loc[fid]
            cluster_id = meta_v['cluster']
            es_ruido = cluster_id == -1

            fig.add_trace(go.Scattergl(
                x=vuelo['x'], y=vuelo['y'],
                mode='lines',
                line=dict(color=color, width=1.2, dash='dot' if es_ruido else 'solid'),
                opacity=0.4 if es_ruido else 0.7,
                name=label if primera else None,
                showlegend=primera,
                legendgroup=f'ruta_{ruta}',
                hovertemplate=(
                    f'<b>Vuelo:</b> {fid}<br>'
                    f'<b>Ruta:</b> {meta_v.get("adep", "?")} ({meta_v.get("name_adep", "")}) → '
                    f'{meta_v.get("ades", "?")} ({meta_v.get("name_ades", "")})<br>'
                    f'<b>Aerolínea:</b> {meta_v.get("airline", "?")}<br>'
                    f'<b>{"Ruido" if es_ruido else f"Cluster {cluster_id}"}</b>'
                    '<extra></extra>'
                )
            ))
            primera = False

    n_total = len(ids_filtrados)
    n_ruido_aero = (meta_vuelos['cluster'] == -1).sum()
    n_cluster = n_total - n_ruido_aero
    clusters_presentes = sorted(meta_vuelos[meta_vuelos['cluster'] >= 0]['cluster'].unique())
    nombre = aeropuertos.get(codigo_aeropuerto, '')

    # Calcular rango fijo de ejes (con margen del 5%)
    x_min, x_max = df_filtrado['x'].min(), df_filtrado['x'].max()
    y_min, y_max = df_filtrado['y'].min(), df_filtrado['y'].max()
    margen_x = (x_max - x_min) * 0.05
    margen_y = (y_max - y_min) * 0.05

    fig.update_layout(
        title=f'{codigo_aeropuerto} ({nombre}) — {n_total} vuelos ({n_cluster} en clusters, {n_ruido_aero} ruido)',
        xaxis_title='X (metros, LCC)',
        yaxis_title='Y (metros, LCC)',
        xaxis=dict(scaleanchor='y', scaleratio=1, range=[x_min - margen_x, x_max + margen_x]),
        yaxis=dict(range=[y_min - margen_y, y_max + margen_y]),
        plot_bgcolor='white',
        height=600,
        legend=dict(font=dict(size=9), itemclick='toggle', itemdoubleclick='toggleothers'),
        margin=dict(l=60, r=20, t=50, b=60)
    )

    # Resumen del aeropuerto
    pct_ruido = n_ruido_aero / n_total * 100 if n_total > 0 else 0

    # Tabla de rutas desglosada
    filas_tabla = []
    for ruta in rutas_unicas:
        ids_ruta = meta_vuelos[meta_vuelos['ruta'] == ruta].index
        # Obtener nombres de la primera fila
        primera_fila = meta_vuelos.loc[ids_ruta[0]]
        name_adep_r = primera_fila.get('name_adep', '')
        name_ades_r = primera_fila.get('name_ades', '')
        adep_r = primera_fila.get('adep', '?')
        ades_r = primera_fila.get('ades', '?')
        ruta_con_nombre = f'{adep_r} ({name_adep_r}) → {ades_r} ({name_ades_r})'
        for cluster_id, grupo in meta_vuelos.loc[ids_ruta].groupby('cluster'):
            n = len(grupo)
            if cluster_id == -1:
                etiqueta = 'Ruido'
            else:
                etiqueta = f'Cluster {cluster_id}'
            filas_tabla.append({'ruta': ruta_con_nombre, 'cluster': etiqueta, 'cluster_id': cluster_id, 'n_vuelos': n})

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


# === EJECUCIÓN ===
if __name__ == '__main__':
    print("\nIniciando dashboard en http://127.0.0.1:8050")
    print("Pulsa Ctrl+C para detener\n")
    app.run(debug=False, port=8050)