import re
import pandas as pd
from flask_login import current_user
from models import ArchivoHTML, Jugador



def obtener_dataframe_actual():
    """
    Devuelve el DataFrame que debe usar la aplicación.

    - Si el usuario está logueado y tiene archivos subidos:
      carga los jugadores del último archivo desde la base de datos.

    - Si no está logueado o no tiene archivos:
      carga la demo desde jugadores.pkl.
    """

    if current_user.is_authenticated:
        ultimo_archivo = ArchivoHTML.query.filter_by(
            usuario_id=current_user.id
        ).order_by(
            ArchivoHTML.fecha_subida.desc()
        ).first()

        if ultimo_archivo:
            jugadores = Jugador.query.filter_by(
                archivo_id=ultimo_archivo.id
            ).all()

            if jugadores:
                datos = []

                for jugador in jugadores:
                    if jugador.datos_json:
                        fila = dict(jugador.datos_json)
                    else:
                        fila = {}

                    fila["Nombre"] = jugador.nombre
                    fila["Edad"] = jugador.edad
                    fila["Posición"] = jugador.posicion
                    fila["Club"] = jugador.club
                    fila["Sueldo"] = jugador.sueldo
                    fila["Media"] = jugador.media
                    fila["Gol"] = jugador.goles
                    fila["Asis"] = jugador.asistencias
                    fila["Min"] = jugador.minutos

                    datos.append(fila)

                return pd.DataFrame(datos)

    return pd.read_pickle("jugadores.pkl")