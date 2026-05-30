from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash
import re

from models import db, Usuario


auth = Blueprint("auth", __name__)


def password_segura(password):
    if len(password) < 6:
        return False

    tiene_mayuscula = re.search(r"[A-ZÁÉÍÓÚÑ]", password)
    tiene_numero = re.search(r"\d", password)
    tiene_signo = re.search(r"[^A-Za-zÁÉÍÓÚáéíóúÑñ0-9]", password)

    return tiene_mayuscula and tiene_numero and tiene_signo


@auth.route("/registro", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        nombre = request.form.get("nombre")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        terms = request.form.get("terms")

        if not nombre or not email or not password or not confirm_password:
            flash("Todos los campos son obligatorios.")
            return redirect(url_for("auth.registro"))
        
        if not terms:
            flash("Debes aceptar los términos de uso y la política de privacidad.")
            return redirect(url_for("auth.registro"))

        if password != confirm_password:
            flash("Las contraseñas no coinciden.")
            return redirect(url_for("auth.registro"))

        if not password_segura(password):
            flash("La contraseña debe tener al menos 6 caracteres, una mayúscula, un número y un signo.")
            return redirect(url_for("auth.registro"))

        usuario_existente = Usuario.query.filter_by(email=email).first()

        if usuario_existente:
            flash("Ya existe una cuenta con ese email.")
            return redirect(url_for("auth.registro"))

        nuevo_usuario = Usuario(
            nombre=nombre,
            email=email,
            password_hash=generate_password_hash(password)
        )

        db.session.add(nuevo_usuario)
        db.session.commit()

        flash("Cuenta creada correctamente. Ahora puedes iniciar sesión.")
        return redirect(url_for("auth.login"))

    return render_template("registro.html")


@auth.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        usuario = Usuario.query.filter_by(email=email).first()

        if not usuario or not check_password_hash(usuario.password_hash, password):
            flash("Email o contraseña incorrectos.")
            return redirect(url_for("auth.login"))

        login_user(usuario)
        flash("Has iniciado sesión correctamente.")
        return redirect(url_for("index"))

    return render_template("login.html")


@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Has cerrado sesión correctamente.")
    return redirect(url_for("index"))