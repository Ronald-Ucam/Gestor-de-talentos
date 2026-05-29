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


@app.route("/api/nombres_jugadores")
def api_nombres_jugadores():
    df_actual = obtener_dataframe_actual()

    if "Nombre" not in df_actual.columns:
        return jsonify([])

    nombres = (
        df_actual["Nombre"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    nombres = sorted(nombres[nombres != ""].unique().tolist())

    return jsonify(nombres)



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

        start = (page - 1) * PAGE_SIZE
        end = start + PAGE_SIZE
        df_page = df_filtrado.iloc[start:end].copy()

        # Renombrado solo visual para que los atributos se entiendan mejor.
        # No cambia la base de datos ni los filtros, solo los nombres que se muestran en la tabla.
        mapa_columnas = {
            "% Pase": "Precisión de pase",
            "% disparos": "Precisión de disparo",
            "Asis": "Asistencias",
            "Asis/90": "Asistencias / 90 min",
            "Bal aér/90": "Balones aéreos / 90 min",
            "Desp": "Despejes",
            "Entr/90": "Entradas / 90 min",
            "Final": "Fin de contrato",
            "Gol": "Goles",
            "Gol/90": "Goles / 90 min",
            "Media": "Valoración media",
            "Min": "Minutos jugados",
            "Min/Par": "Minutos por partido",
            "Part": "Partidos",
            "Pas Clv/90": "Pases clave / 90 min",
            "Pases prog": "Pases progresivos",
            "Pases prog/90": "Pases progresivos / 90 min",
            "Pos Gan/90": "Posesiones ganadas / 90 min",
            "Pos Perd/90": "Posesiones perdidas / 90 min",
            "Reg": "Regates",
            "Reg/90": "Regates / 90 min",
            "Rob/90": "Robos / 90 min",
            "Tir/90": "Tiros / 90 min",
            "TirP/90": "Tiros a puerta / 90 min",
            "Titular": "Titularidades",
            "ValorNum": "Valor mínimo",
            "xG": "Goles esperados (xG)",
            # Columnas principales visibles en la tabla
            "Valor de traspaso": "Valor mercado",
            "Sueldo": "Sueldo anual",
            "Altura": "Altura",
            "Peso": "Peso",
            "Posición": "Posición",
            "Club": "Club",
            "Edad": "Edad",
            "Nombre": "Jugador"
        }

        df_page = df_page.rename(columns=mapa_columnas)

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


@app.route("/api/comparar")
def api_comparar():
    j1 = request.args.get("jugador1")
    j2 = request.args.get("jugador2")

    df_actual = obtener_dataframe_actual()

    p1 = df_actual[df_actual["Nombre"] == j1]
    p2 = df_actual[df_actual["Nombre"] == j2]

    if p1.empty or p2.empty:
        return jsonify({"error": "Jugador no encontrado"}), 404

    p1, p2 = p1.iloc[0], p2.iloc[0]

    def obtener_grupo_posicion(posicion):
        """
        Agrupa la posición del jugador en una categoría general:
        POR, DEF, MED o DEL.

        La prioridad está pensada para evitar comparaciones incoherentes:
        - Porteros siempre son POR.
        - Si aparece DL, se considera perfil ofensivo.
        - Si aparece DF o CR, se considera defensa.
        - Si aparece MC/MCD/MCO, se considera centrocampista.
        - ME/MP sin MC ni DF se consideran perfil ofensivo.
        """
        if pd.isna(posicion):
            return ""

        pos = str(posicion).upper()

        if "POR" in pos:
            return "POR"

        # Si puede jugar como delantero centro, lo tratamos como delantero/ofensivo
        if "DL" in pos:
            return "DEL"

        # Defensas y carrileros
        if any(p in pos for p in ["DF", "DFC", "LI", "LD", "CR"]):
            return "DEF"

        # Centrocampistas
        if any(p in pos for p in ["MC", "MCD", "MCO"]):
            return "MED"

        # Extremos y mediapuntas ofensivos
        if any(p in pos for p in ["ME", "MP", "EI", "ED"]):
            return "DEL"

        return pos.split(",")[0].strip()

    pos1 = obtener_grupo_posicion(p1.get("Posición", ""))
    pos2 = obtener_grupo_posicion(p2.get("Posición", ""))

    print("COMPARANDO:", j1, pos1, "VS", j2, pos2)

    def extraer_codigos_posicion(posicion):
        """
        Extrae códigos de posición del texto.
        Ejemplos:
        'MP (DIC), DL (C)' -> {'MP', 'DL'}
        'MC, ME (C), MP (DC)' -> {'MC', 'ME', 'MP'}
        'POR' -> {'POR'}
        """
        if pd.isna(posicion):
            return set()

        pos = str(posicion).upper()

        codigos_validos = {
            "POR", "DF", "DFC", "LI", "LD", "CR",
            "MC", "MCD", "MCO", "MP",
            "DL", "ME", "EI", "ED"
        }

        encontrados = set(re.findall(r"\b[A-Z]{2,3}\b", pos))

        return encontrados.intersection(codigos_validos)


    def grupo_desde_codigo(codigo):
        """
        Convierte un código concreto de posición a grupo general.
        """
        if codigo == "POR":
            return "POR"

        if codigo in ["DF", "DFC", "LI", "LD", "CR"]:
            return "DEF"

        if codigo in ["MC", "MCD", "MCO"]:
            return "MED"

        if codigo in ["MP", "DL", "ME", "EI", "ED"]:
            return "DEL"

        return ""


    codigos1 = extraer_codigos_posicion(p1.get("Posición", ""))
    codigos2 = extraer_codigos_posicion(p2.get("Posición", ""))

    codigos_comunes = codigos1.intersection(codigos2)

    if not codigos_comunes:
        return jsonify({
            "error": (
                f"No se pueden comparar jugadores de posiciones distintas "
                f"({j1}: {p1.get('Posición', '')} / {j2}: {p2.get('Posición', '')}). "
                "Selecciona dos jugadores que compartan al menos una posición."
            )
        }), 400


    # Elegimos la posición común más relevante para decidir las métricas
    prioridad_codigos = ["POR", "DL", "MP", "ME", "EI", "ED", "MC", "MCO", "MCD", "DF", "DFC", "LI", "LD", "CR"]

    codigo_comun = next(
        (codigo for codigo in prioridad_codigos if codigo in codigos_comunes),
        list(codigos_comunes)[0]
    )

    pos1 = grupo_desde_codigo(codigo_comun)
    pos2 = pos1

    print("COMPARANDO:", j1, codigos1, "VS", j2, codigos2, "COMÚN:", codigo_comun, "GRUPO:", pos1)

    metricas_por_grupo = {
        "POR": [
            "Media",
            "Enc/90",
            "Portería imbatida",
            "Rp %",
            "BDs",
            "BRe",
            "Pen. parados"
        ],

        "DEF": [
            "Media",
            "Entr/90",
            "Rob/90",
            "Desp",
            "Bal aér/90",
            "% Pase",
            "Pases prog/90"
        ],

        "MED": [
            "Media",
            "Asis/90",
            "Pas Clv/90",
            "% Pase",
            "Pases prog/90",
            "Reg/90",
            "Rob/90"
        ],

        "DEL": [
            "Media",
            "Gol/90",
            "xG",
            "Asis/90",
            "Disparos",
            "Tir/90",
            "Reg/90"
        ]
    }

    attrs = metricas_por_grupo.get(
        pos1,
        ["Media", "Gol/90", "Asis/90", "Reg/90", "Pas Clv/90"]
    )

    # Nos quedamos solo con columnas que existan realmente en el DataFrame
    attrs = [a for a in attrs if a in df_actual.columns]

    if len(attrs) < 2:
        return jsonify({
            "error": f"No hay suficientes métricas disponibles para comparar jugadores del grupo {pos1}."
        }), 400

    # Métricas donde un valor menor representa mejor rendimiento
    metricas_menor_mejor = ["Enc/90", "Pos Perd/90", "FC"]

    def to_float(val):
        """
        Convierte valores del dataset a float.
        Soporta:
        - "-"
        - ""
        - porcentajes como "85%"
        - unidades como "299.3 km"
        - valores con coma decimal
        """
        if pd.isna(val):
            return 0.0

        texto = str(val).strip()

        if texto in ["-", "", "nan", "None"]:
            return 0.0

        texto = texto.replace(",", ".")
        limpio = re.sub(r"[^0-9\.\-]", "", texto)

        if limpio in ["", "-", "."]:
            return 0.0

        try:
            return float(limpio)
        except ValueError:
            return 0.0

    clean = {}

    for a in attrs:
        clean[a] = (
            df_actual[a]
            .apply(to_float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    def calcular_percentil(nombre_metrica, valor):
        percentil = percentileofscore(clean[nombre_metrica], valor)

        # En estas métricas, cuanto menor es el valor, mejor.
        # Ejemplo: en Enc/90 interesa encajar menos goles por partido.
        if nombre_metrica in metricas_menor_mejor:
            percentil = 100 - percentil

        return round(percentil, 1)

    statsA = [calcular_percentil(a, to_float(p1[a])) for a in attrs]
    statsB = [calcular_percentil(a, to_float(p2[a])) for a in attrs]

    campos_perfil = [
        "Nombre", "Edad", "Altura", "Peso", "Posición", "Club",
        "Valor de traspaso", "ValorNum", "Sueldo", "Media",

        # Métricas ofensivas
        "Gol", "Gol/90", "xG", "Asis", "Asis/90",
        "Disparos", "Tir/90", "TirP/90", "% disparos",
        "Reg", "Reg/90", "Pas Clv/90",

        # Métricas de pase / creación
        "% Pase", "Pases prog", "Pases prog/90", "Ps I/90", "Ps C/90",

        # Métricas defensivas
        "Entr/90", "Rob/90", "Desp", "Bal aér/90",
        "Pos Gan/90", "Pos Perd/90",

        # Métricas de portero
        "Enc", "Enc/90", "Portería imbatida",
        "Rp %", "BDs", "BRe", "Pen. recibidos",
        "Pen. parados", "Prop. penaltis parados",

        # Otros datos
        "Min", "Min/Par", "Part", "Titular"
    ]

    def construir_perfil(p, nombre):
        perfil = {}

        foto = None

        if "buscar_foto_wikipedia" in globals():
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
        "grupo": pos1,
        "perfilA": construir_perfil(p1, j1),
        "perfilB": construir_perfil(p2, j2)
    })





if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
