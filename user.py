from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user

from models import db, ArchivoHTML, Jugador


user = Blueprint("user", __name__)


@user.route("/mi-cuenta")
@login_required
def mi_cuenta():
    archivos = ArchivoHTML.query.filter_by(
        usuario_id=current_user.id
    ).all()

    total_archivos = len(archivos)

    total_jugadores = 0
    for archivo in archivos:
        total_jugadores += archivo.jugadores_detectados or 0

    ultimo_archivo = ArchivoHTML.query.filter_by(
        usuario_id=current_user.id
    ).order_by(
        ArchivoHTML.fecha_subida.desc()
    ).first()

    return render_template(
        "mi_cuenta.html",
        total_archivos=total_archivos,
        total_jugadores=total_jugadores,
        ultimo_archivo=ultimo_archivo
    )


@user.route("/mis-archivos")
@login_required
def mis_archivos():
    archivos = ArchivoHTML.query.filter_by(
        usuario_id=current_user.id
    ).order_by(
        ArchivoHTML.fecha_subida.desc()
    ).all()

    return render_template("mis_archivos.html", archivos=archivos)


@user.route("/eliminar-archivo/<int:archivo_id>", methods=["POST"])
@login_required
def eliminar_archivo(archivo_id):
    archivo = ArchivoHTML.query.filter_by(
        id=archivo_id,
        usuario_id=current_user.id
    ).first_or_404()

    try:
        Jugador.query.filter_by(archivo_id=archivo.id).delete()

        db.session.delete(archivo)
        db.session.commit()

        flash("Archivo eliminado correctamente.")
    except Exception as e:
        db.session.rollback()
        flash(f"Error al eliminar el archivo: {e}")

    return redirect(url_for("user.mis_archivos"))