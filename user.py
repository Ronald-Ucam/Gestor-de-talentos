from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash
from models import db, Usuario, ArchivoHTML, Jugador, Favorito
import re

user = Blueprint("user", __name__)

def password_segura(password):
    if len(password) < 6:
        return False

    tiene_mayuscula = re.search(r"[A-ZÁÉÍÓÚÑ]", password)
    tiene_numero = re.search(r"\d", password)
    tiene_signo = re.search(r"[^A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", password)

    return tiene_mayuscula and tiene_numero and tiene_signo


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

@user.route("/cambiar_password", methods=["POST"])
@login_required
def cambiar_password():
    password_actual = request.form.get("password_actual", "").strip()
    password_nueva = request.form.get("password_nueva", "").strip()
    password_confirmar = request.form.get("password_confirmar", "").strip()

    if not password_actual or not password_nueva or not password_confirmar:
        flash("Todos los campos de contraseña son obligatorios.")
        return redirect(url_for("user.mi_cuenta"))

    usuario = db.session.get(Usuario, current_user.id)

    if not usuario:
        flash("No se pudo encontrar el usuario.")
        return redirect(url_for("user.mi_cuenta"))

    if not check_password_hash(usuario.password_hash, password_actual):
        flash("La contraseña actual no es correcta.")
        return redirect(url_for("user.mi_cuenta"))

    if password_nueva != password_confirmar:
        flash("La nueva contraseña y la confirmación no coinciden.")
        return redirect(url_for("user.mi_cuenta"))

    if not password_segura(password_nueva):
        flash("La nueva contraseña debe tener al menos 6 caracteres, una mayúscula, un número y un signo.")
        return redirect(url_for("user.mi_cuenta"))

    usuario.password_hash = generate_password_hash(password_nueva)

    db.session.add(usuario)
    db.session.commit()

    logout_user()
    flash("Contraseña actualizada correctamente. Inicia sesión con la nueva contraseña.")
    return redirect(url_for("auth.login"))


@user.route("/eliminar-cuenta", methods=["POST"])
@login_required
def eliminar_cuenta():
    password_confirmacion = request.form.get("password_confirmacion", "").strip()

    if not password_confirmacion:
        flash("Debes introducir tu contraseña para eliminar la cuenta.")
        return redirect(url_for("user.mi_cuenta"))

    usuario = db.session.get(Usuario, current_user.id)

    if not usuario:
        flash("No se pudo encontrar el usuario.")
        return redirect(url_for("index"))

    if not check_password_hash(usuario.password_hash, password_confirmacion):
        flash("La contraseña introducida no es correcta.")
        return redirect(url_for("user.mi_cuenta"))

    try:
        usuario_id = usuario.id

        archivos = ArchivoHTML.query.filter_by(usuario_id=usuario_id).all()

        for archivo in archivos:
            Jugador.query.filter_by(archivo_id=archivo.id).delete()

        ArchivoHTML.query.filter_by(usuario_id=usuario_id).delete()


        db.session.delete(usuario)
        db.session.commit()

        logout_user()

        flash("Tu cuenta y todos tus datos asociados han sido eliminados correctamente.")
        return redirect(url_for("index"))

    except Exception as e:
        db.session.rollback()
        flash(f"Error al eliminar la cuenta: {e}")
        return redirect(url_for("user.mi_cuenta"))