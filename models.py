from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()


class Usuario(db.Model, UserMixin):
    __tablename__ = "usuarios"

    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow)

    archivos = db.relationship("ArchivoHTML", backref="usuario", lazy=True)


class ArchivoHTML(db.Model):
    __tablename__ = "archivos_html"

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=True)

    nombre_archivo = db.Column(db.String(255), nullable=False)
    contenido_html = db.Column(db.Text, nullable=False)

    fecha_subida = db.Column(db.DateTime, default=datetime.utcnow)
    jugadores_detectados = db.Column(db.Integer, default=0)
    columnas_detectadas = db.Column(db.Integer, default=0)
    estado = db.Column(db.String(50), default="procesado")

    jugadores = db.relationship(
        "Jugador",
        backref="archivo",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Jugador(db.Model):
    __tablename__ = "jugadores"

    id = db.Column(db.Integer, primary_key=True)
    archivo_id = db.Column(db.Integer, db.ForeignKey("archivos_html.id"), nullable=False)

    nombre = db.Column(db.String(150))
    edad = db.Column(db.Integer)
    posicion = db.Column(db.String(120))
    club = db.Column(db.String(150))
    valor_traspaso = db.Column(db.String(120))
    sueldo = db.Column(db.String(120))

    media = db.Column(db.Float)
    goles = db.Column(db.Float)
    asistencias = db.Column(db.Float)
    minutos = db.Column(db.Float)

    datos_json = db.Column(db.JSON)


class Favorito(db.Model):
    __tablename__ = "favoritos"

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False)
    jugador_id = db.Column(db.Integer, db.ForeignKey("jugadores.id"), nullable=False)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)