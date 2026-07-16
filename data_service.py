import os
import pandas as pd
from flask_login import current_user
from models import ArchivoHTML, Jugador


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_PICKLE_PATH = os.path.join(BASE_DIR, "jugadores.pkl")


def obtener_dataframe_actual():
    """
    Devuelve el DataFrame que debe usar la aplicación.

    - Si el usuario está autenticado y tiene archivos subidos:
      carga los jugadores del último archivo desde PostgreSQL.

    - Si no está autenticado o no tiene archivos:
      carga la demo desde jugadores.pkl.
    """

    if current_user.is_authenticated:
        ultimo_archivo = (
            ArchivoHTML.query
            .filter_by(usuario_id=current_user.id)
            .order_by(ArchivoHTML.fecha_subida.desc())
            .first()
        )

        if ultimo_archivo:
            jugadores = Jugador.query.filter_by(
                archivo_id=ultimo_archivo.id
            ).all()

            if jugadores:
                datos = []

                for jugador in jugadores:
                    fila = (
                        dict(jugador.datos_json)
                        if jugador.datos_json
                        else {}
                    )

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

                df_usuario = pd.DataFrame(datos)

                print(
                    f"[DATA SERVICE] Usuario autenticado. "
                    f"Jugadores cargados desde BD: {len(df_usuario)}"
                )

                return df_usuario

    if not os.path.exists(DEMO_PICKLE_PATH):
        raise FileNotFoundError(
            f"No se encontró la demo en: {DEMO_PICKLE_PATH}"
        )

    df_demo = pd.read_pickle(DEMO_PICKLE_PATH)

    print(f"[DATA SERVICE] Ruta demo: {DEMO_PICKLE_PATH}")
    print(f"[DATA SERVICE] Jugadores demo cargados: {len(df_demo)}")

    if "Posición" in df_demo.columns:
        print(
            "[DATA SERVICE] Posiciones demo:",
            df_demo["Posición"].value_counts(dropna=False).to_dict()
        )

    return df_demo