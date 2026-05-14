from flask import Flask, render_template, request, abort, jsonify,redirect, url_for, flash, render_template, make_response
import pandas as pd
import os
import re
import numpy as np
import requests
from scipy.stats import percentileofscore
import subprocess
from flask import Flask, request, redirect, url_for, flash, render_template
from preprocesar_tabla import procesar_BBDD_html
from models import db, Usuario, ArchivoHTML, Jugador, Favorito
from flask_login import LoginManager, login_required, current_user
from auth import auth
from user import user
from clustering import clustering_bp
from data_service import obtener_dataframe_actual, convertir_float, convertir_int, limpiar_json_fila

app = Flask(__name__)

app.register_blueprint(auth)
app.register_blueprint(user)
app.register_blueprint(clustering_bp)

#app.secret_key = "clave_secreta" lo he cambiado por lo de ahora abajo
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "clave_secreta")

database_url = os.environ.get("DATABASE_URL")

if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

if database_url:
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///local.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth.login"


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(Usuario, int(user_id))


with app.app_context():
    db.create_all()








# Guardar al nivel de app.py
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'BBDD.html')
ALLOWED_EXTENSIONS = {'html', 'htm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@app.route('/upload_html', methods=['POST'])
@login_required
def upload_html():
    if 'htmlFile' not in request.files:
        flash('No se encontró el archivo.')
        return redirect(url_for('index'))

    file = request.files['htmlFile']

    if file.filename == '':
        flash('No se seleccionó ningún archivo.')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Solo se permiten archivos HTML (.html, .htm).')
        return redirect(url_for('index'))

    try:
        # 1. Leer el contenido del HTML
        contenido_html = file.read().decode("utf-8", errors="ignore")

        # 2. Guardar también el archivo físico como hacías antes
        with open(SAVE_PATH, "w", encoding="utf-8") as f:
            f.write(contenido_html)

        # 3. Procesar el HTML sin modificar la demo global
        df = procesar_BBDD_html(SAVE_PATH, guardar_demo=False)

        if df is None or df.empty:
            flash('Archivo subido, pero ocurrió un error en el procesamiento.')
            return redirect(url_for('index'))

        # 4. Leer el pickle generado por tu procesamiento actual
        ######df = pd.read_pickle("jugadores.pkl")

        # 5. Guardar el HTML en PostgreSQL
        nuevo_archivo = ArchivoHTML(
            usuario_id=current_user.id,
            nombre_archivo=file.filename,
            contenido_html=contenido_html,
            jugadores_detectados=len(df),
            columnas_detectadas=len(df.columns),
            estado="procesado"
        )

        db.session.add(nuevo_archivo)
        db.session.commit()

        # 6. Guardar jugadores en PostgreSQL
        for _, fila in df.iterrows():
            jugador = Jugador(
                archivo_id=nuevo_archivo.id,
                nombre=str(fila.get("Nombre", "")),
                edad=convertir_int(fila.get("Edad")),
                posicion=str(fila.get("Posición", "")),
                club=str(fila.get("Club", "")),
                valor_traspaso=str(fila.get("Valor de traspaso", "")),
                sueldo=str(fila.get("Sueldo", "")),
                media=convertir_float(fila.get("Media")),
                goles=convertir_float(fila.get("Gol")),
                asistencias=convertir_float(fila.get("Asis")),
                minutos=convertir_float(fila.get("Min")),
                datos_json=limpiar_json_fila(fila)
                )

            db.session.add(jugador)

        db.session.commit()

        flash('Archivo HTML subido, procesado y guardado en la base de datos correctamente.')

    except Exception as e:
        db.session.rollback()
        flash(f'Ocurrió un error al procesar el archivo: {e}')

    return redirect(url_for('index'))



pickle_path = os.path.join(os.getcwd(), "jugadores.pkl")
if not os.path.exists(pickle_path):
    # Genera el pickle automáticamente si falta
    subprocess.run(["python", "preprocesar_tabla.py"], check=True)
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
    df_actual = obtener_dataframe_actual()

    jugadores = df_actual["Nombre"].dropna().unique().tolist()

    seleccionados = request.args.getlist("players[]")

    j1 = seleccionados[0] if len(seleccionados) > 0 else (jugadores[0] if jugadores else "")
    j2 = seleccionados[1] if len(seleccionados) > 1 else (jugadores[1] if len(jugadores) > 1 else j1)

    return render_template(
        "comparacion.html",
        jugadores_list=jugadores,
        selected1=j1,
        selected2=j2
    )




@app.route("/mostrar_bd")
def mostrar_bd():
    try:
        df_jugadores = obtener_dataframe_actual()

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
            .astype(int)      
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

        html = render_template(
            "BBDD_filtrada.html",
            table_html=table_html,
            page=page,
            total_pages=total_pages,
            filtros=filtros
        )
        # Envuelve en make_response para añadir cabeceras
        resp = make_response(html)
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma']        = 'no-cache'
        resp.headers['Expires']       = '0'
        return resp

    except Exception as e:
        return f"Error al filtrar/paginar: {str(e)}", 500


@app.route("/api/nombres_jugadores")
def api_nombres_jugadores():
    df = obtener_dataframe_actual()
    nombres = sorted(df["Nombre"].dropna().unique().tolist())
    return jsonify(nombres)


def buscar_foto_wikipedia(nombre):
    """
    Busca la miniatura de Wikipedia para un jugador:
    1) Query de búsqueda “<nombre> futbolista”
    2) Si no hay resultado o no hay thumbnail, busca <nombre> con opensearch
    """
    WIKI_API = "https://es.wikipedia.org/w/api.php"

    # Helper para extraer thumbnail de un título dado
    def obtener_thumbnail(titulo):
        params_img = {
            "action": "query",
            "titles": titulo,
            "prop": "pageimages",
            "format": "json",
            "pithumbsize": 300
        }
        resp2 = requests.get(WIKI_API, params=params_img, timeout=5)
        if not resp2.ok:
            return None
        pages = resp2.json().get("query", {}).get("pages", {})
        for p in pages.values():
            thumb = p.get("thumbnail", {})
            if thumb.get("source"):
                return thumb["source"]
        return None

    # 1) Búsqueda con “futbolista”
    params_search = {
        "action": "query",
        "list": "search",
        "srsearch": f"{nombre} futbolista",
        "format": "json",
        "srlimit": 1
    }
    resp = requests.get(WIKI_API, params=params_search, timeout=5)
    if resp.ok:
        results = resp.json().get("query", {}).get("search", [])
        if results:
            thumb = obtener_thumbnail(results[0]["title"])
            if thumb:
                return thumb

    # 2) Fallback: opensearch puro
    params_open = {
        "action": "opensearch",
        "search": nombre,
        "limit": 1,
        "namespace": 0,
        "format": "json"
    }
    resp3 = requests.get(WIKI_API, params=params_open, timeout=5)
    if not resp3.ok:
        return None
    data = resp3.json()
    # data[1] es lista de títulos
    if len(data) > 1 and data[1]:
        titulo2 = data[1][0]
        thumb2 = obtener_thumbnail(titulo2)
        if thumb2:
            return thumb2

    return None




@app.route("/api/comparar")
def api_comparar():
    j1 = request.args.get("jugador1")
    j2 = request.args.get("jugador2")

    df_actual = obtener_dataframe_actual()

    attrs = ["Media", "Gol/90", "Asis/90", "Reg/90", "Pas Clv/90"]

    p1 = df_actual[df_actual["Nombre"] == j1]
    p2 = df_actual[df_actual["Nombre"] == j2]

    if p1.empty or p2.empty:
        return jsonify({"error": "Jugador no encontrado"}), 404

    p1, p2 = p1.iloc[0], p2.iloc[0]

    def to_float(val):
        return 0.0 if str(val).strip() in ["-", ""] else float(re.sub(r"[^\d\.]", "", str(val)))

    clean = {}
    for a in attrs:
        clean[a] = df_actual[a] \
            .replace("-", np.nan) \
            .astype(str) \
            .str.replace(r"[^\d\.]", "", regex=True) \
            .replace("", "0") \
            .astype(float)

    statsA = [round(percentileofscore(clean[a], to_float(p1[a])), 1) for a in attrs]
    statsB = [round(percentileofscore(clean[a], to_float(p2[a])), 1) for a in attrs]

    campos_perfil = [
        "Nombre", "Edad", "Altura", "Peso", "Posición", "Club",
        "ValorNum", "Sueldo", "Media", "Gol/90", "Asis/90",
        "Reg/90", "Pas Clv/90", "Disparos", "Min/Par"
    ]

    def construir_perfil(p, nombre):
        perfil = {}

        foto = buscar_foto_wikipedia(nombre)
        if foto:
            perfil["FotoURL"] = foto

        for c in campos_perfil:
            if c in df_actual.columns:
                v = p[c]
                perfil[c] = v.item() if hasattr(v, "item") else v

        return perfil

    return jsonify({
        "labels": attrs,
        "statsA": statsA,
        "statsB": statsB,
        "nameA": j1,
        "nameB": j2,
        "perfilA": construir_perfil(p1, j1),
        "perfilB": construir_perfil(p2, j2)
    })












if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
