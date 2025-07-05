from flask import Flask, render_template, request, abort, jsonify
import pandas as pd
import os
import re
import numpy as np
import requests
from scipy.stats import percentileofscore
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from pyngrok import ngrok


app = Flask(__name__)
public_url = ngrok.connect(5000)
print(" * Tunnel URL:", public_url)

# --- Carga del DataFrame desde pickle ---
pickle_path = os.path.join(os.getcwd(), "jugadores.pkl")
if not os.path.exists(pickle_path):
    raise FileNotFoundError(
        "Falta jugadores.pkl. Ejecuta preprocesar_tabla.py primero para generarlo."
    )
df_jugadores = pd.read_pickle(pickle_path)


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/faqs')
def faqs():
    return render_template("faqs.html")

@app.route("/comparacion")
def comparacion():
    # 1) Lista completa de nombres
    jugadores = df_jugadores["Nombre"].unique().tolist()

    # 2) Leemos el array players[] que mandó el formulario
    seleccionados = request.args.getlist("players[]")
    # 3) Rellenamos selected1 y selected2 con fallback a los dos primeros de la lista
    j1 = seleccionados[0] if len(seleccionados) > 0 else (jugadores[0] if jugadores else "")
    j2 = seleccionados[1] if len(seleccionados) > 1 else (jugadores[1] if len(jugadores) > 1 else j1)

    # 4) Renderizamos pasando también selected1/selected2
    return render_template(
        "comparacion.html",
        jugadores_list=jugadores,
        selected1=j1,
        selected2=j2
    )


@app.route('/clustering')
def clustering():
    # Ya no recopilamos porteros, ni mapas de filtros…
    return render_template('clustering.html')


@app.route('/clustering/porteros')
def clustering_porteros():
    jugadores = df_jugadores["Nombre"].unique().tolist()

    porteros = (
        df_jugadores[df_jugadores["Posición"] == "POR"]["Nombre"]
        .sort_values()
        .unique()
        .tolist()
    )

    # Mapa nombre → edad (int)
    age_map = {
        row.Nombre: int(row.Edad)
        for row in df_jugadores[["Nombre", "Edad"]].itertuples()
        if pd.notna(row.Edad)
    }

    # Construimos un dict nombre → posición primaria
    pos_map = {
        row.Nombre: row.Posición.split(",")[0].strip()
        for row in df_jugadores[["Nombre","Posición"]].itertuples()
    }

    # Mapa nombre → altura (float, en cm)
    height_map = {}
    for row in df_jugadores[["Nombre", "Altura"]].itertuples():
        h = row.Altura
        if isinstance(h, str):
            # elimina todo lo que no sea dígito o punto
            h_clean = re.sub(r"[^\d\.]", "", h)
            if h_clean:
                height_map[row.Nombre] = float(h_clean)
        elif pd.notna(h):
            height_map[row.Nombre] = float(h)

    # Mapa nombre → valor de mercado (float, en millones o la unidad que uses)
    value_map = {}
    for row in df_jugadores[["Nombre", "ValorNum"]].itertuples():
        v = row.ValorNum
        if pd.notna(v):
            # si ya es numérico, lo guardamos directamente
            try:
                value_map[row.Nombre] = float(v)
            except:
                # si viene como string con 'M', 'k', etc.
                v_str = str(v)
                v_clean = re.sub(r"[^\d\.]", "", v_str)
                if v_clean:
                    value_map[row.Nombre] = float(v_clean)

    return render_template(
        "clusteringpor.html",
        jugadores_list=jugadores,
        age_map=age_map,
        porteros_list=porteros,     # <— nueva
        height_map=height_map,
        value_map=value_map
    )                     


@app.route('/clustering/defensas')
def clustering_defensas():
    jugadores = df_jugadores['Nombre'].unique().tolist()

    # Extrae sólo los defensas para el datalist (opcional)
    defensas = (
        df_jugadores[df_jugadores['Posición'].str.contains(r'\bDF\b', na=False)]['Nombre']
        .sort_values()
        .unique()
        .tolist()
    )

    # Mapa nombre → edad, altura, valor (si tu html lo necesita para filtros)
    age_map = {
        row.Nombre: int(row.Edad)
        for row in df_jugadores[['Nombre','Edad']].itertuples()
        if pd.notna(row.Edad)
    }
    height_map = {}
    for row in df_jugadores[['Nombre','Altura']].itertuples():
        h = row.Altura
        if isinstance(h, str):
            h_clean = re.sub(r'[^\d\.]', '', h)
            if h_clean: height_map[row.Nombre] = float(h_clean)
        elif pd.notna(h):
            height_map[row.Nombre] = float(h)
    value_map = {}
    for row in df_jugadores[['Nombre','ValorNum']].itertuples():
        v = row.ValorNum
        if pd.notna(v):
            try:    value_map[row.Nombre] = float(v)
            except:
                v_clean = re.sub(r'[^\d\.]','', str(v))
                if v_clean: value_map[row.Nombre] = float(v_clean)

    return render_template(
        'clusteringdef.html',
        jugadores_list=jugadores,
        defensas_list=defensas,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )


#@app.route("/clustering")
#def clustering():
 #   jugadores = df_jugadores["Nombre"].unique().tolist()
  #  return render_template("clustering.html", jugadores_list=jugadores)


@app.route("/mostrar_bd")
def mostrar_bd():
    try:
        nombre      = request.args.get("nombre",    default=None, type=str)
        edad        = request.args.get("edad",      type=int)
        posicion    = request.args.get("posicion",  default=None, type=str)
        partidos    = request.args.get("partidos",  default=None, type=str)
        valor_min       = request.args.get("valor_min",       type=float)
        valor_max       = request.args.get("valor",           type=float)
        goles_min       = request.args.get("goles_min",       type=float)
        goles_max       = request.args.get("goles",           type=float)
        asis_min        = request.args.get("asistencias_min", type=float)
        asis_max        = request.args.get("asistencias",     type=float)
        disparos_min = request.args.get("disparos_min", type=float)
        disparos_max = request.args.get("disparos",     type=float)
        page        = request.args.get("page",      default=1,   type=int)
        PAGE_SIZE   = 50

        df_filtrado = df_jugadores.copy()

        df_filtrado["Disparos"] = (
            df_filtrado["Disparos"]
            .replace("-", np.nan)
            .astype(float)
            .fillna(0)
            .astype(int)        # <- devuelve a enteros
        )


        if nombre:
            df_filtrado = df_filtrado[
                df_filtrado["Nombre"].str.contains(nombre, case=False, na=False)
            ]
        if edad is not None:
            df_filtrado = df_filtrado[df_filtrado["Edad"] == edad]
        if posicion:
            equivalencias = {
                "Portero": "POR",
                "Defensa": "DF",
                "Centrocampista": "MC",
                "Delantero": "DL",
                "Extremo": "ME",
                "Mediapunta": "MP",
                "Carrilero": "CR",
                "Pivote Defensivo": "MCD"
            }
            buscado = equivalencias.get(posicion, posicion)
            df_filtrado = df_filtrado[
                df_filtrado["Posición"].str.contains(rf"\b{buscado}\b", case=False, na=False)
            ]
        if partidos:
            if partidos == "0":
                df_filtrado = df_filtrado[df_filtrado["Titular"] == 0]
            elif partidos == "1-5":
                df_filtrado = df_filtrado[(df_filtrado["Titular"] >= 1) & (df_filtrado["Titular"] <= 5)]
            elif partidos == "6-15":
                df_filtrado = df_filtrado[(df_filtrado["Titular"] >= 6) & (df_filtrado["Titular"] <= 15)]
            elif partidos == "16-25":
                df_filtrado = df_filtrado[(df_filtrado["Titular"] >= 16) & (df_filtrado["Titular"] <= 25)]
            elif partidos.startswith("26"):
                df_filtrado = df_filtrado[df_filtrado["Titular"] >= 26]
        
        if valor_min is not None:
            df_filtrado = df_filtrado[df_filtrado["ValorNum"] >= valor_min]
        if valor_max is not None:
            df_filtrado = df_filtrado[df_filtrado["ValorNum"] <= valor_max]

        if goles_min is not None and "Gol" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Gol"] >= goles_min]
        if goles_max is not None and "Gol" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Gol"] <= goles_max]

        if asis_min is not None and "Asis" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Asis"] >= asis_min]
        if asis_max is not None and "Asis" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Asis"] <= asis_max]

        if disparos_min is not None and "Disparos" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Disparos"] >= disparos_min]
        if disparos_max is not None and "Disparos" in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado["Disparos"] <= disparos_max]


        if df_filtrado.empty:
            return "<h2>No se encontraron jugadores con los filtros seleccionados</h2>"

        total_rows  = len(df_filtrado)
        total_pages = (total_rows + PAGE_SIZE - 1) // PAGE_SIZE
        if page < 1 or page > total_pages:
            abort(404)

        start   = (page - 1) * PAGE_SIZE
        end     = start + PAGE_SIZE
        df_page = df_filtrado.iloc[start:end]

        table_html = df_page.to_html(
            index=False,
            classes="table table-striped table-bordered",
            border=0,
            justify="center"
        )

        filtros = request.args.to_dict()
        filtros.pop("page", None)

        return render_template(
            "BBDD_filtrada.html",
            table_html=table_html,
            page=page,
            total_pages=total_pages,
            filtros=filtros
        )

    except Exception as e:
        return f"Error al filtrar/paginar: {str(e)}", 500



def buscar_foto_wikipedia(nombre):
    url = "https://es.wikipedia.org/w/api.php"
    params = {
        "action":      "query",
        "titles":      nombre,
        "prop":        "pageimages",
        "format":      "json",
        "pithumbsize": 300
    }
    resp = requests.get(url, params=params, timeout=5)
    if not resp.ok:
        return None
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        if thumb.get("source"):
            return thumb["source"]
    return None


@app.route("/api/comparar")
def api_comparar():
    j1, j2 = request.args.get("jugador1"), request.args.get("jugador2")
    attrs = ["Media", "Gol/90", "Asis/90", "Reg/90", "% Pase"]

    p1 = df_jugadores[df_jugadores["Nombre"] == j1]
    p2 = df_jugadores[df_jugadores["Nombre"] == j2]
    if p1.empty or p2.empty:
        return jsonify({"error": "Jugador no encontrado"}), 404
    p1, p2 = p1.iloc[0], p2.iloc[0]

    def to_float(val):
        return 0.0 if str(val).strip() == "-" else float(val)

    clean  = {a: df_jugadores[a].replace("-", 0).astype(float) for a in attrs}
    statsA = [round(percentileofscore(clean[a], to_float(p1[a])), 1) for a in attrs]
    statsB = [round(percentileofscore(clean[a], to_float(p2[a])), 1) for a in attrs]

    campos_perfil = ["Nombre", "Edad", "Altura", "Peso", "Posición", "Club",
                     "ValorNum", "Sueldo", "Media", "Gol/90", "Asis/90",
                     "Reg/90", "% Pase", "Disparos", "Min/Par"]

    def construir_perfil(p, nombre):
        perfil = {}
        foto = buscar_foto_wikipedia(nombre)
        if foto:
            perfil["FotoURL"] = foto
        for c in campos_perfil:
            if c in df_jugadores.columns:
                v = p[c]
                perfil[c] = v.item() if hasattr(v, "item") else v
        return perfil

    return jsonify({
        "labels":  attrs,
        "statsA":  statsA,
        "statsB":  statsB,
        "nameA":   j1,
        "nameB":   j2,
        "perfilA": construir_perfil(p1, j1),
        "perfilB": construir_perfil(p2, j2)
    })




@app.route("/api/cluster_por")
def api_cluster_por():
    try:
        k = int(request.args.get("k", 4))
        df_por, attrs, cluster_names = cluster_goalkeepers(df_jugadores, k=k)
        return jsonify({
            "jugadores":    df_por["Nombre"].tolist(),
            "labels":       df_por["cluster"].tolist(),
            "coords2":      df_por[["x_pca", "y_pca"]].values.tolist(),
            "clusterNames": cluster_names,
            "attrs":        attrs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/cluster_def')
def api_cluster_def():
    """API que devuelve clustering de defensas en JSON."""
    k = request.args.get('k', default=4, type=int)
    df_def, attrs, names = cluster_defenders(df_jugadores, k)
    return jsonify({
        'jugadores':    df_def['Nombre'].tolist(),
        'labels':       df_def['cluster'].tolist(),
        'coords2':      df_def[['x_pca', 'y_pca']].values.tolist(),
        'clusterNames': names,
        'attrs':        attrs
    })

# Y el endpoint en app.py
@app.route("/api/cluster_mid")
def api_cluster_mid():
    try:
        k = int(request.args.get("k", 4))
        df_mid, attrs, cluster_names = cluster_midfielders(df_jugadores, k)
        return jsonify({
            "jugadores":    df_mid["Nombre"].tolist(),
            "labels":       df_mid["cluster"].tolist(),
            "coords2":      df_mid[["x_pca", "y_pca"]].values.tolist(),
            "clusterNames": cluster_names,
            "attrs":        attrs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# -- Vista para renderizar el clustering de mediocentros --
@app.route('/clustering/centrocampistas')
def clustering_centrocampistas():
    jugadores = df_jugadores['Nombre'].unique().tolist()

    # Extrae solo los mediocentros para el datalist
    midfielders = (
        df_jugadores[df_jugadores['Posición'].str.contains(r'\bMC\b', na=False)]['Nombre']
        .sort_values()
        .unique()
        .tolist()
    )

    # Mapas de filtro: edad, altura y valor
    age_map = {
        r.Nombre: int(r.Edad)
        for r in df_jugadores[['Nombre','Edad']].itertuples()
        if pd.notna(r.Edad)
    }

    height_map = {}
    for r in df_jugadores[['Nombre','Altura']].itertuples():
        h = r.Altura
        if isinstance(h, str):
            hc = re.sub(r'[^\d\.]', '', h)
            if hc: height_map[r.Nombre] = float(hc)
        elif pd.notna(h):
            height_map[r.Nombre] = float(h)

    value_map = {}
    for r in df_jugadores[['Nombre','ValorNum']].itertuples():
        v = r.ValorNum
        if pd.notna(v):
            try:
                value_map[r.Nombre] = float(v)
            except:
                vc = re.sub(r'[^\d\.]', '', str(v))
                if vc: value_map[r.Nombre] = float(vc)

    return render_template(
        'clusteringmed.html',
        jugadores_list=jugadores,
        midfielders_list=midfielders,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )



@app.route("/api/cluster_fw")
def api_cluster_fw():
    try:
        k = int(request.args.get("k", 4))
        df_fw, attrs, names = cluster_forwards(df_jugadores, k)
        return jsonify({
            "jugadores":    df_fw["Nombre"].tolist(),
            "labels":       df_fw["cluster"].tolist(),
            "coords2":      df_fw[["x_pca","y_pca"]].values.tolist(),
            "clusterNames": names,
            "attrs":        attrs
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/clustering/delanteros')
def clustering_delanteros():
    jugadores = df_jugadores["Nombre"].unique().tolist()

    delanteros = (
        df_jugadores[df_jugadores["Posición"].str.contains(r"\bDL\b", na=False)]["Nombre"]
        .sort_values()
        .unique()
        .tolist()
    )

    # ——— Construcción de mapas ———
    age_map = {
        row.Nombre: int(row.Edad)
        for row in df_jugadores[["Nombre","Edad"]].itertuples()
        if pd.notna(row.Edad)
    }

    height_map = {}
    for row in df_jugadores[["Nombre","Altura"]].itertuples():
        h = row.Altura
        if isinstance(h, str):
            h_clean = re.sub(r"[^\d\.]", "", h)
            if h_clean: height_map[row.Nombre] = float(h_clean)
        elif pd.notna(h):
            height_map[row.Nombre] = float(h)

    value_map = {}
    for row in df_jugadores[["Nombre","ValorNum"]].itertuples():
        v = row.ValorNum
        if pd.notna(v):
            try:
                value_map[row.Nombre] = float(v)
            except:
                v_clean = re.sub(r"[^\d\.]", "", str(v))
                if v_clean: value_map[row.Nombre] = float(v_clean)
    # ————————————————————————

    return render_template(
        'clusteringdel.html',      # tu plantilla para delanteros
        jugadores_list=jugadores,
        delanteros_list=delanteros,
        age_map=age_map,
        height_map=height_map,
        value_map=value_map
    )


def cluster_goalkeepers(df, k=4):
    # 1) Filtrar sólo porteros
    df_por = df[df["Posición"] == "POR"].copy()
    
    # 2) Métricas relevantes
    features = [
        "CS/90",         # porterías imbatidas por 90'
        "Enc/90",        # goles encajados por 90'
        "Rp %",          # % de paradas
        "Pen. parados",  # penaltis parados
        "BDs",           # despejes
        "Distancia"      # km recorridos
    ]
    cols = [c for c in features if c in df_por.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas de portero: {cols}")

    # 3) Limpiar y convertir a float
    for c in cols:
        s = df_por[c].astype(str)
        s = s.replace('-', np.nan)                     # '-' → NaN
        s = s.str.replace(r'[^0-9\.]', '', regex=True) # quita todo menos dígitos y punto
        df_por[c] = s.replace('', np.nan).astype(float)

    # 4) Descartar columnas muy vacías y rellenar medianas
    df_por = df_por.dropna(axis=1, thresh=len(df_por) * 0.7)
    cols   = [c for c in cols if c in df_por.columns]
    df_por[cols] = df_por[cols].fillna(df_por[cols].median())

    # 5) Escalar
    Xs = StandardScaler().fit_transform(df_por[cols].values)

    # 6) K-Means
    k_eff  = min(k, Xs.shape[0])
    km     = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(Xs)
    
    # 7) PCA para 2D
    coords = PCA(n_components=2, random_state=0).fit_transform(Xs)

    # 8) Mapeo a descripciones legibles
    pretty = {
        'CS/90':         'Porterías imbatidas / 90′',
        'Enc/90':        'Goles encajados / 90′',
        'Rp %':          'Porcentaje de paradas',
        'Pen. parados':  'Penaltis parados',
        'BDs':           'Despejes totales',
        'Distancia':     'Kilómetros recorridos'
    }
    centros       = km.cluster_centers_
    cluster_names = []
    for center in centros:
        best      = cols[np.argmax(center)]
        nice      = pretty.get(best, best)
        cluster_names.append(f"Alto en {nice}")

    # 9) Adjuntar resultados
    df_por = df_por.reset_index(drop=True)
    df_por["cluster"] = labels
    df_por["x_pca"]   = coords[:, 0]
    df_por["y_pca"]   = coords[:, 1]

    return df_por, cols, cluster_names


#####DEFENSAS CLUSTERING

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
    # 1) Filtrar defensas (etiqueta "DF")
    df_def = df[df["Posición"].str.contains(r"\bDF\b", na=False)].copy()

    # 2) Seleccionar columnas existentes
    features = ["Entr/90", "Bal aér/90", "Int/90", "Desp", "Pos Gan/90", "% Pase"]
    cols = [c for c in features if c in df_def.columns]

    # 3) Limpiar y convertir a float
    for c in cols:
        s = df_def[c].astype(str).replace('-', np.nan)
        s = s.str.replace(r'[^0-9\.]', '', regex=True)
        df_def[c] = pd.to_numeric(s, errors='coerce')

    # 4) Eliminar columnas con >30% NaN y rellenar restantes con medianas
    df_def = df_def.dropna(axis=1, thresh=len(df_def)*0.7)
    cols = [c for c in cols if c in df_def.columns]
    df_def[cols] = df_def[cols].fillna(df_def[cols].median())

    # 5) Escalar y clusterizar
    X = StandardScaler().fit_transform(df_def[cols].values)
    k_eff = min(k, X.shape[0])
    km = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 6) Reducir a 2D para visualización
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

     # 7) Nombrar clúster usando descripciones legibles
    pretty_def = {
        'Entr/90':    'Entradas ganadas por 90′',
        'Bal aér/90': 'Duelos aéreos ganados / 90′',
        'Int/90':     'Intercepciones por 90′',
        'Desp':       'Despejes totales',
        'Pos Gan/90': 'Recuperaciones de posición / 90′',
        '% Pase':     'Precisión de pase'
    }
    centers = km.cluster_centers_
    cluster_names = []
    for c in centers:
        best = cols[np.argmax(c)]
        nice = pretty_def.get(best, best)
        cluster_names.append(f"Alto en {nice}")

    # 8) Construir DataFrame de resultados
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
    # 1) Filtrar mediocentros
    df_mid = df[df["Posición"].str.contains(r"\bMC\b", na=False)].copy()

    # 2) Métricas ampliadas
    features = [
        "Reg/90",      # regates
        "Pas Clv/90",  # pases clave
        "% Pase",      # precisión de pase
        "Asis/90",     # asistencias
        "Distancia",   # km recorridos
        "Pas Prog/90", # pases progresivos (si existe)
        "Rob/90"       # robos/recuperaciones
    ]
    # Filtramos solo las que realmente estén en el DataFrame
    cols = [c for c in features if c in df_mid.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas para mediocentros: {cols}")

    # 3) Limpiar y convertir a float
    for c in cols:
        s = df_mid[c].astype(str).replace('-', np.nan)
        s = s.str.replace(r'[^0-9\.]', '', regex=True)
        df_mid[c] = pd.to_numeric(s, errors='coerce')

    # 4) Eliminar columnas con >30% NaN y rellenar medianas
    df_mid = df_mid.dropna(axis=1, thresh=len(df_mid)*0.7)
    cols   = [c for c in cols if c in df_mid.columns]
    df_mid[cols] = df_mid[cols].fillna(df_mid[cols].median())

    # 5) Escalar
    X = StandardScaler().fit_transform(df_mid[cols].values)

    # 6) K-Means
    k_eff  = min(k, X.shape[0])
    km     = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 7) Reducir a 2D con PCA
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    # 8) Generar nombres legibles
    pretty_mid = {
        'Reg/90':      'Regates completados / 90′',
        'Pas Clv/90':  'Pases clave / 90′',
        '% Pase':      'Precisión de pase',
        'Asis/90':     'Asistencias / 90′',
        'Distancia':   'Kilómetros recorridos',
        'Pas Prog/90': 'Pases progresivos / 90′',
        'Rob/90':      'Recuperaciones (robos) / 90′'
    }
    centros = km.cluster_centers_
    cluster_names = []
    for center in centros:
        best = cols[np.argmax(center)]
        nice = pretty_mid.get(best, best)
        cluster_names.append(f"Alto en {nice}")

    # 9) Devolver los resultados
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
    """
    # 1) Filtrar delanteros
    df_fw = df[df["Posición"].str.contains(r"\bDL\b", na=False)].copy()

    # 2) Seleccionar las métricas disponibles
    features = ["Gol/90", "Asis/90", "Reg/90", "% Pase", "Disparos", "Min/Par", "OC/90" ]
    cols = [c for c in features if c in df_fw.columns]
    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas para delanteros: {cols}")

    # 3) Limpiar y convertir a float
    for c in cols:
        s = df_fw[c].astype(str).replace('-', np.nan)
        s = s.str.replace(r'[^0-9\.]', '', regex=True)
        df_fw[c] = pd.to_numeric(s, errors='coerce')

    # 4) Eliminar columnas con >30% NaN y rellenar medianas
    df_fw = df_fw.dropna(axis=1, thresh=len(df_fw)*0.7)
    cols  = [c for c in cols if c in df_fw.columns]
    df_fw[cols] = df_fw[cols].fillna(df_fw[cols].median())

    # 5) Escalar
    X = StandardScaler().fit_transform(df_fw[cols].values)

    # 6) K-Means
    k_eff = min(k, X.shape[0])
    km    = KMeans(n_clusters=k_eff, random_state=0, n_init="auto")
    labels = km.fit_predict(X)

    # 7) PCA 2D
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    # 8) Nombres legibles
    pretty_fw = {
        "Gol/90":   "Goles / 90′",
        "Asis/90":  "Asistencias / 90′",
        "Reg/90":   "Regates / 90′",
        "% Pase":   "Precisión de pase",
        "Disparos": "Disparos totales",
        "Min/Par":  "Minutos por gol",
        "OC/90":    "Ocasiones creadas / 90′"    # <-- leyenda clara

    }
    centros = km.cluster_centers_
    cluster_names = []
    for center in centros:
        best = cols[np.argmax(center)]
        nice = pretty_fw.get(best, best)
        cluster_names.append(f"Alto en {nice}")

    # 9) Devolver DataFrame y metadatos
    df_out = df_fw.reset_index(drop=True)
    df_out["cluster"] = labels
    df_out["x_pca"]   = coords[:,0]
    df_out["y_pca"]   = coords[:,1]
    return df_out, cols, cluster_names






if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
