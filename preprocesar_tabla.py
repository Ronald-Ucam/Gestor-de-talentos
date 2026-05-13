import pandas as pd
from bs4 import BeautifulSoup
import re
import os


def extraer_dataframe_desde_html(html_path=None):
    """
    Lee un archivo HTML exportado desde Football Manager,
    extrae la tabla principal y devuelve un DataFrame limpio.
    No guarda archivos físicos. Solo procesa y devuelve datos.
    """

    if html_path is None:
        html_path = os.path.join(os.getcwd(), "BBDD.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"No se encontró el archivo HTML: {html_path}")

    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    tabla = soup.find("table")

    if tabla is None:
        raise ValueError("No se encontró ninguna <table> en el archivo HTML.")

    headers = [th.text.strip() for th in tabla.find_all("th")]

    rows = []
    for fila in tabla.find_all("tr")[1:]:
        cols = [td.text.strip() for td in fila.find_all("td")]
        if len(cols) == len(headers):
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)

    # Limpieza básica para que la app pueda filtrar y analizar mejor
    if "Edad" in df.columns:
        df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce").fillna(0).astype(int)

    if "Titular" in df.columns:
        def limpiar_titular(val):
            try:
                match = re.match(r"\d+", str(val))
                return int(match.group()) if match else 0
            except Exception:
                return 0

        df["Titular"] = df["Titular"].apply(limpiar_titular)

    for col in ["Gol", "Asis", "% Pase"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "Valor de traspaso" in df.columns:
        def extraer_valor_min(texto):
            try:
                texto_clean = str(texto).replace(",", "")
                match = re.search(r"(\d+\.?\d*)", texto_clean)
                return float(match.group(1)) if match else 0.0
            except Exception:
                return 0.0

        df["ValorNum"] = df["Valor de traspaso"].apply(extraer_valor_min)

    return df


def guardar_dataframe_demo(df):
    """
    Guarda un DataFrame como demo global de la aplicación.
    Esta función solo debería usarse para generar/actualizar la demo.
    """

    df.to_pickle("jugadores.pkl")
    print("✔ jugadores.pkl generado correctamente.")

    full_html = df.to_html(index=False, classes="table table-striped")
    with open("full_table.html", "w", encoding="utf-8") as f:
        f.write(full_html)

    print("✔ full_table.html generado correctamente.")


def procesar_BBDD_html(html_path=None, guardar_demo=True):
    """
    Función compatible con el sistema anterior.

    Si guardar_demo=True:
        procesa el HTML y actualiza jugadores.pkl y full_table.html.

    Si guardar_demo=False:
        procesa el HTML y devuelve el DataFrame sin modificar la demo.
    """

    try:
        df = extraer_dataframe_desde_html(html_path)

        if guardar_demo:
            guardar_dataframe_demo(df)

        return df

    except Exception as e:
        print(f"Error procesando HTML: {e}")
        return None


if __name__ == "__main__":
    procesar_BBDD_html(guardar_demo=True)