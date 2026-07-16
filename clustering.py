from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data_service import obtener_dataframe_actual
from preprocesar_tabla import convertir_float, convertir_int


clustering_bp = Blueprint("clustering", __name__)

def crear_age_map(df):
    return {
        row.Nombre: convertir_int(row.Edad)
        for row in df[["Nombre", "Edad"]].itertuples()
        if pd.notna(row.Edad)
    }


def crear_height_map(df):
    return {
        row.Nombre: convertir_float(row.Altura)
        for row in df[["Nombre", "Altura"]].itertuples()
        if pd.notna(row.Altura)
    }


def crear_value_map(df):
    return {
        row.Nombre: convertir_float(row.ValorNum)
        for row in df[["Nombre", "ValorNum"]].itertuples()
        if pd.notna(row.ValorNum)
    }


def limpiar_metricas_clustering(df, cols):
    """
    Asegura que las métricas usadas por KMeans sean numéricas.
    La limpieza real ya viene de preprocesar_tabla.py, pero esta función
    protege el clustering ante valores vacíos o no válidos.
    """
    df = df.copy()

    for c in cols:
        if c in df.columns:
            df[c] = df[c].apply(convertir_float)

    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)

    return df

@clustering_bp.route('/clustering')
def clustering():
    return render_template('clustering.html')


#Para la gráfica
@clustering_bp.route('/clustering/porteros')
def clustering_porteros():
    df_actual = obtener_dataframe_actual()
    jugadores = df_actual["Nombre"].unique().tolist()

    porteros = (
        df_actual[df_actual["Posición"] == "POR"]["Nombre"]
        .sort_values()
        .unique()
        .tolist()
    )

    age_map = crear_age_map(df_actual)
    height_map = crear_height_map(df_actual)
    value_map = crear_value_map(df_actual)

    return render_template(
        "clusteringpor.html",
        jugadores_list=jugadores,
        porteros_list=porteros,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )


#Para la gráfica
@clustering_bp.route('/clustering/defensas')
def clustering_defensas():
    df_actual = obtener_dataframe_actual()
    jugadores = df_actual["Nombre"].unique().tolist()

    defensas = (
        df_actual[df_actual['Posición'].str.contains(r'\bDF\b', na=False)]['Nombre']
        .sort_values()
        .unique()
        .tolist()
    )

    age_map = crear_age_map(df_actual)
    height_map = crear_height_map(df_actual)
    value_map = crear_value_map(df_actual)

    return render_template(
        'clusteringdef.html',
        jugadores_list=jugadores,
        defensas_list=defensas,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )


#Para la gráfica
@clustering_bp.route('/clustering/centrocampistas')
def clustering_centrocampistas():
    df_actual = obtener_dataframe_actual()
    jugadores = df_actual["Nombre"].unique().tolist()

    midfielders = (
        df_actual[df_actual['Posición'].str.contains(r'\bMC\b', na=False)]['Nombre']
        .sort_values()
        .unique()
        .tolist()
    )

    age_map = crear_age_map(df_actual)
    height_map = crear_height_map(df_actual)
    value_map = crear_value_map(df_actual)

    return render_template(
        'clusteringmed.html',
        jugadores_list=jugadores,
        midfielders_list=midfielders,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )



#Para el mapa
@clustering_bp.route('/clustering/delanteros')
def clustering_delanteros():
    df_actual = obtener_dataframe_actual()
    jugadores = df_actual["Nombre"].unique().tolist()

    delanteros = (
        df_actual[df_actual["Posición"].str.contains(r"\bDL\b", na=False)]["Nombre"]
        .sort_values()
        .unique()
        .tolist()
    )

    age_map = crear_age_map(df_actual)
    height_map = crear_height_map(df_actual)
    value_map = crear_value_map(df_actual)

    return render_template(
        'clusteringdel.html',
        jugadores_list=jugadores,
        delanteros_list=delanteros,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )


"""
Ahora implementamos las APIS de los cluster de los demas
"""

@clustering_bp.route("/api/cluster_por")
def api_cluster_por():
    try:
        # 1️ Parámetro k (con valor por defecto 4)
        k = int(request.args.get("k", 4))

        # 2️ Carga siempre actualizada de los datos procesados
        df_jugadores = obtener_dataframe_actual()

        # 3️ Aplica clustering a los porteros
        df_por, attrs, cluster_names = cluster_goalkeepers(df_jugadores, k=k)

        # 4️Devuelve JSON con los resultados
        return jsonify({
            "jugadores":    df_por["Nombre"].tolist(),
            "labels":       df_por["cluster"].tolist(),
            "coords2":      df_por[["x_pca", "y_pca"]].values.tolist(),
            "clusterNames": cluster_names,
            "attrs":        attrs
        })

    except ValueError as ve:
        return jsonify({"error": f"Parámetro inválido: {ve}"}), 400

    except FileNotFoundError as fnf:
        # Si no existe jugadores.pkl
        return jsonify({"error": str(fnf)}), 500

    except Exception as e:
        # Cualquier otro error
        return jsonify({"error": f"Error al generar clusters: {e}"}), 500


@clustering_bp.route('/api/cluster_def')
def api_cluster_def():
    try:
        # 1️ Parámetro k (por defecto 4)
        k = request.args.get('k', default=4, type=int)

        # 2️ Carga fresca de los datos preprocesados
        df_jugadores = obtener_dataframe_actual()

        # 3️ Aplica clustering a los defensas
        df_def, attrs, names = cluster_defenders(df_jugadores, k)

        # 4️ Devuelve el JSON con resultados
        return jsonify({
            'jugadores':    df_def['Nombre'].tolist(),
            'labels':       df_def['cluster'].tolist(),
            'coords2':      df_def[['x_pca', 'y_pca']].values.tolist(),
            'clusterNames': names,
            'attrs':        attrs
        })

    except ValueError as ve:
        # Parámetro k inválido
        return jsonify({'error': f'Parámetro invalido: {ve}'}), 400

    except FileNotFoundError as fnf:
        return jsonify({'error': str(fnf)}), 500

    except Exception as e:
        return jsonify({'error': f'Error al generar clusters de defensas: {e}'}), 500


@clustering_bp.route("/api/cluster_mid")
def api_cluster_mid():
    try:
        # 1️ Número de clusters (por defecto 4)
        k = int(request.args.get("k", 4))

        # 2️ Carga siempre fresca del DataFrame procesado
        df_jugadores = obtener_dataframe_actual()

        # 3️ Aplica clustering a los centrocampistas
        df_mid, attrs, cluster_names = cluster_midfielders(df_jugadores, k)

        # 4️ Devuelve JSON con nombres, etiquetas y coordenadas
        return jsonify({
            "jugadores":    df_mid["Nombre"].tolist(),
            "labels":       df_mid["cluster"].tolist(),
            "coords2":      df_mid[["x_pca", "y_pca"]].values.tolist(),
            "clusterNames": cluster_names,
            "attrs":        attrs
        })

    except ValueError as ve:
        return jsonify({"error": f"Parámetro inválido: {ve}"}), 400

    except FileNotFoundError as fnf:
        return jsonify({"error": str(fnf)}), 500

    except Exception as e:
        return jsonify({"error": f"Error al generar clusters: {e}"}), 500

@clustering_bp.route("/api/cluster_fw")
def api_cluster_fw():
    try:
        # 1️ Número de clusters (por defecto 4)
        k = int(request.args.get("k", 4))

        # 2️ Carga siempre fresca de los datos procesados
        df_jugadores = obtener_dataframe_actual()

        # 3️ Aplica clustering a los delanteros
        df_fw, attrs, names = cluster_forwards(df_jugadores, k)

        # 4️ Devuelve JSON con nombres, etiquetas y coordenadas
        return jsonify({
            "jugadores":    df_fw["Nombre"].tolist(),
            "labels":       df_fw["cluster"].tolist(),
            "coords2":      df_fw[["x_pca", "y_pca"]].values.tolist(),
            "clusterNames": names,
            "attrs":        attrs
        })

    except ValueError as ve:
        # Si k no es un entero válido
        return jsonify({"error": f"Parámetro inválido: {ve}"}), 400

    except FileNotFoundError as fnf:
        # Si falta el pickle en disco
        return jsonify({"error": str(fnf)}), 500

    except Exception as e:
        # Cualquier otro error
        return jsonify({"error": f"Error al generar clusters de delanteros: {e}"}), 500


"""
Funciones del cluster
"""

def cluster_goalkeepers(df, k=4):
    # 1 Filtrar sólo porteros
    df_por = df[df["Posición"] == "POR"].copy()

    # Filtrar porteros con pocos minutos para evitar perfiles poco representativos
    if "Min" in df_por.columns:
        df_por["Min"] = df_por["Min"].apply(convertir_float)
        df_por = df_por[df_por["Min"] >= 450].copy()

    if df_por.empty:
        raise ValueError("No hay suficientes porteros con minutos mínimos para clustering.")

    # 2 Métricas relevantes originales
    base_feats = [
        "Enc/90",
        "Portería imbatida",
        "Rp %",
        "BDs",
        "BRe",
        "Pen. parados",
        "BAt"
    ]
    cols = [c for c in base_feats if c in df_por.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas de portero: {cols}")

    # 3 Limpiar y convertir todas estas a float
    df_por = limpiar_metricas_clustering(df_por, cols)

    # 4 Derivar nuevas métricas numéricas si están disponibles
    extras = []

    # 5 Recalcular lista de columnas tras derivar
    cols = [c for c in cols + extras if c in df_por.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas de portero después de añadir extras: {cols}")

    # 6 Descartar columnas con muchos NaN y rellenar medianas
    df_por = df_por.dropna(axis=1, thresh=len(df_por) * 0.7)
    cols   = [c for c in cols if c in df_por.columns]
    df_por[cols] = df_por[cols].fillna(df_por[cols].median())

    # 7 Escalar
    Xs = StandardScaler().fit_transform(df_por[cols].values)

    # 8 K-Means
    k_eff  = min(k, Xs.shape[0])
    km     = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(Xs)

    # 9 PCA para 2D
    coords = PCA(n_components=2, random_state=0).fit_transform(Xs)

    # 10 Mapeo a descripciones legibles asegurando unicidad
    pretty = {
        "Enc/90": "Goles encajados / 90′",
        "Portería imbatida": "Porterías imbatidas",
        "Rp %": "Porcentaje de paradas",
        "Pen. parados": "Penaltis parados",
        "BDs": "Despejes totales",
        "BRe": "Balones rechazados",
        "BAt": "Balones atrapados"
    }
    centros = km.cluster_centers_
    n_clusters = centros.shape[0]
    n_metrics = len(cols)

    # Generar todos los triples (cluster, métrica, valor) y ordenarlos
    triples = [(ci, mi, centros[ci, mi])
               for ci in range(n_clusters) for mi in range(n_metrics)]
    triples.sort(key=lambda x: x[2], reverse=True)

    assignments   = {}
    used_metrics  = set()
    used_clusters = set()
    for ci, mi, _ in triples:
        if ci not in used_clusters and mi not in used_metrics:
            assignments[ci]   = mi
            used_clusters.add(ci)
            used_metrics.add(mi)
        if len(used_clusters) == n_clusters:
            break

    cluster_names = []
    for ci in range(n_clusters):
        mi   = assignments.get(ci, int(np.argmax(centros[ci])))
        name = pretty.get(cols[mi], cols[mi])
        cluster_names.append(f"Alto en {name}")

    # 11 Adjuntar resultados al DataFrame
    df_por = df_por.reset_index(drop=True)
    df_por["cluster"] = labels
    df_por["x_pca"]   = coords[:, 0]
    df_por["y_pca"]   = coords[:, 1]

    return df_por, cols, cluster_names




def cluster_defenders(df, k=4):
    """
    Realiza clustering de defensas usando métricas clave:
    - Entr/90: entradas ganadas por 90'
    - Bal aér/90: duelos aéreos ganados por 90'
    - Int/90: intercepciones por 90'
    - Desp: despejes
    - Pos Gan/90: recuperaciones de posición por 90'
    - % Pase: precisión de pase
    """
    # 1 Filtrar defensas (etiqueta "DF")
    df_def = df[df["Posición"].str.contains(r"\bDF\b", na=False)].copy()
    # Filtrar jugadores con pocos minutos para evitar perfiles poco representativos
    if "Min" in df_def.columns:
        df_def["Min"] = df_def["Min"].apply(convertir_float)
        df_def = df_def[df_def["Min"] >= 450].copy()

    if df_def.empty:
        raise ValueError("No hay suficientes defensas con minutos mínimos para clustering.")

    # 2 Limpieza inicial de columnas base
    base_feats = [
        "Entr/90",
        "Rob/90",
        "Desp",
        "Bal aér/90",
        "Pos Gan/90",
        "% Pase",
        "Pases prog/90"
    ]
    
    cols = [c for c in base_feats if c in df_def.columns]

    df_def = limpiar_metricas_clustering(df_def, cols)

    # 3 Derivar métricas adicionales si disponemos de datos
    extras = []
    # Entradas limpiadoras
    if "Ent Cl" in df_def.columns:
        df_def["Ent Cl"] = df_def["Ent Cl"].apply(convertir_float)
        extras.append("Ent Cl")

    # 4 Reconstruir lista de columnas tras extras
    cols = [c for c in cols + extras if c in df_def.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas de defensa: {cols}")

    # 5 Eliminar columnas con >30% NaN y rellenar con medianas
    df_def = df_def.dropna(axis=1, thresh=len(df_def)*0.7)
    cols = [c for c in cols if c in df_def.columns]
    df_def[cols] = df_def[cols].fillna(df_def[cols].median())

    # 6 Escalar y clusterizar
    X = StandardScaler().fit_transform(df_def[cols].values)
    k_eff = min(k, X.shape[0])
    km = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 7 PCA para 2D
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    # 8 Mapear nombres sin repeticiones
    pretty_def = {
        "Entr/90": "Entradas / 90′",
        "Rob/90": "Robos / 90′",
        "Desp": "Despejes totales",
        "Bal aér/90": "Balones aéreos ganados / 90′",
        "Pos Gan/90": "Posesiones ganadas / 90′",
        "% Pase": "Precisión de pase",
        "Pases prog/90": "Pases progresivos / 90′",
        "Ent Cl": "Entradas clave"
    }
    centers = km.cluster_centers_
    n_clusters = centers.shape[0]
    n_metrics = len(cols)

    # crear triple lista y ordenar por valor
    triples = [(ci, mi, centers[ci, mi])
               for ci in range(n_clusters) for mi in range(n_metrics)]
    triples.sort(key=lambda x: x[2], reverse=True)

    assignments = {}
    used_metrics = set()
    used_clusters = set()
    for ci, mi, _ in triples:
        if ci not in used_clusters and mi not in used_metrics:
            assignments[ci] = mi
            used_clusters.add(ci)
            used_metrics.add(mi)
        if len(used_clusters) == n_clusters:
            break

    cluster_names = []
    for ci in range(n_clusters):
        mi = assignments.get(ci, int(np.argmax(centers[ci])))
        name = pretty_def.get(cols[mi], cols[mi])
        cluster_names.append(f"Alto en {name}")

    # 9) Adjuntar resultados al DataFrame
    df_out = df_def.reset_index(drop=True)
    df_out['cluster'] = labels
    df_out['x_pca'] = coords[:, 0]
    df_out['y_pca'] = coords[:, 1]

    return df_out, cols, cluster_names







def cluster_midfielders(df, k=4):
    """
    Clustering de mediocentros (MC) con métricas clave:
    - Reg/90     : Regates completados por 90'
    - Pas Clv/90 : Pases clave por 90'
    - % Pase     : Precisión de pase
    - Asis/90    : Asistencias por 90'
    - Distancia  : Kilómetros recorridos
    - Pas Prog/90: Pases progresivos por 90'
    - Rob/90     : Recuperaciones (robos) por 90'
    """
    # 1 Filtrar mediocentros
    df_mid = df[df["Posición"].str.contains(r"\bMC\b", na=False)].copy()
    # Filtrar jugadores con pocos minutos para evitar perfiles poco representativos
    if "Min" in df_mid.columns:
        df_mid["Min"] = df_mid["Min"].apply(convertir_float)
        df_mid = df_mid[df_mid["Min"] >= 450].copy()

    if df_mid.empty:
        raise ValueError("No hay suficientes centrocampistas con minutos mínimos para clustering.")

    # 2 Métricas base
    base_feats = [
        "Asis/90",
        "Pas Clv/90",
        "% Pase",
        "Pases prog/90",
        "Reg/90",
        "Rob/90",
        "Distancia"
    ]
    cols = [c for c in base_feats if c in df_mid.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas para mediocentros: {cols}")

    # 3 Limpiar y convertir a float
    df_mid = limpiar_metricas_clustering(df_mid, cols)

    # 4 Derivar nuevas métricas
    extras = []

    # 5 Reconstruir lista de columnas tras extras
    cols = [c for c in cols + extras if c in df_mid.columns]

    # 6 Eliminar columnas con >30% NaN y rellenar medianas
    df_mid = df_mid.dropna(axis=1, thresh=len(df_mid)*0.7)
    cols   = [c for c in cols if c in df_mid.columns]
    df_mid[cols] = df_mid[cols].fillna(df_mid[cols].median())

    # 7 Escalar y clusterizar
    X = StandardScaler().fit_transform(df_mid[cols].values)
    k_eff = min(k, X.shape[0])
    km    = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 8 PCA para visualización
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    # 9 Nombrar clusters sin repetir
    pretty_mid = {
        "Asis/90": "Asistencias / 90′",
        "Pas Clv/90": "Pases clave / 90′",
        "% Pase": "Precisión de pase",
        "Pases prog/90": "Pases progresivos / 90′",
        "Reg/90": "Regates completados / 90′",
        "Rob/90": "Robos / 90′",
        "Distancia": "Kilómetros recorridos",
    }
    centers = km.cluster_centers_
    n_clusters = centers.shape[0]
    n_metrics = len(cols)

    triples = [(ci, mi, centers[ci, mi]) for ci in range(n_clusters) for mi in range(n_metrics)]
    triples.sort(key=lambda x: x[2], reverse=True)

    assignments = {}
    used_metrics = set()
    used_clusters = set()
    for ci, mi, _ in triples:
        if ci not in used_clusters and mi not in used_metrics:
            assignments[ci] = mi
            used_clusters.add(ci)
            used_metrics.add(mi)
        if len(used_clusters) == n_clusters:
            break

    cluster_names = []
    for ci in range(n_clusters):
        mi   = assignments.get(ci, int(np.argmax(centers[ci])))
        name = pretty_mid.get(cols[mi], cols[mi])
        cluster_names.append(f"Alto en {name}")

    # 10 Devolver resultados
    df_mid = df_mid.reset_index(drop=True)
    df_mid["cluster"] = labels
    df_mid["x_pca"]   = coords[:, 0]
    df_mid["y_pca"]   = coords[:, 1]
    return df_mid, cols, cluster_names








def cluster_forwards(df, k=4):
    """
    Clustering de delanteros (DL) con métricas:
    - Gol/90
    - Asis/90
    - Reg/90
    - % Pase
    - Disparos
    - Min/Par
    - OC/90
    """
    # 1 Filtrar delanteros
    df_fw = df[df["Posición"].str.contains(r"\bDL\b", na=False)].copy()
    # Filtrar jugadores con pocos minutos para evitar perfiles poco representativos
    if "Min" in df_fw.columns:
        df_fw["Min"] = df_fw["Min"].apply(convertir_float)
        df_fw = df_fw[df_fw["Min"] >= 450].copy()

    if df_fw.empty:
        raise ValueError("No hay suficientes delanteros con minutos mínimos para clustering.")

    # 2 Métricas base
    base_feats = [
        "Gol/90",
        "xG",
        "Asis/90",
        "Disparos",
        "Tir/90",
        "Reg/90",
        "% disparos"
    ]
    cols = [c for c in base_feats if c in df_fw.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas para delanteros: {cols}")

    # 3 Limpiar y convertir a float
    df_fw = limpiar_metricas_clustering(df_fw, cols)

    # 4Derivar métricas adicionales
    extras = []


    # 5Reconstruir lista de columnas tras extras
    cols = [c for c in cols + extras if c in df_fw.columns]

    # 6Eliminar columnas con >30% NaN y rellenar con medianas
    df_fw = df_fw.dropna(axis=1, thresh=len(df_fw) * 0.7)
    cols  = [c for c in cols if c in df_fw.columns]
    df_fw[cols] = df_fw[cols].fillna(df_fw[cols].median())

    # 7 Escalar y clusterizar
    X = StandardScaler().fit_transform(df_fw[cols].values)
    k_eff = min(k, X.shape[0])
    km    = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 8 PCA 2D
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    # 9 Nombrar clusters sin repetir
    pretty_fw = {
        "Gol/90": "Goles / 90′",
        "Asis/90": "Asistencias / 90′",
        "Disparos": "Disparos totales",
        "Tir/90": "Tiros / 90′",
        "Reg/90": "Regates / 90′",
        "% disparos": "Precisión de disparo",
        "xG": "Goles esperados"
    }
    centers = km.cluster_centers_
    n_clusters = centers.shape[0]
    n_metrics = len(cols)
    triples = [(ci, mi, centers[ci, mi]) for ci in range(n_clusters) for mi in range(n_metrics)]
    triples.sort(key=lambda x: x[2], reverse=True)

    assignments = {}
    used_metrics = set()
    used_clusters = set()
    for ci, mi, _ in triples:
        if ci not in used_clusters and mi not in used_metrics:
            assignments[ci] = mi
            used_clusters.add(ci)
            used_metrics.add(mi)
        if len(used_clusters) == n_clusters:
            break

    cluster_names = []
    for ci in range(n_clusters):
        mi = assignments.get(ci, int(np.argmax(centers[ci])))
        name = pretty_fw.get(cols[mi], cols[mi])
        cluster_names.append(f"Alto en {name}")

    # 10 DataFrame resultado
    df_out = df_fw.reset_index(drop=True)
    df_out["cluster"] = labels
    df_out["x_pca"]   = coords[:, 0]
    df_out["y_pca"]   = coords[:, 1]
    return df_out, cols, cluster_names

