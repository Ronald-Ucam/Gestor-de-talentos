import pandas as pd
from bs4 import BeautifulSoup
import re
import os

def procesar_BBDD_html(html_path=None):
    try:
        # 1️ Ruta al archivo HTML con la tabla original
        if html_path is None:
            html_path = os.path.join(os.getcwd(), "BBDD.html")
        if not os.path.exists(html_path):
            raise FileNotFoundError("No se encontró BBDD.html en la raíz del proyecto.")

        # 2️ Leer el HTML y extraer la tabla con BeautifulSoup
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        tabla = soup.find("table")
        if tabla is None:
            raise ValueError("No se encontró ninguna <table> en BBDD.html")

        # Sacar encabezados (th) y filas (td)
        headers = [th.text.strip() for th in tabla.find_all("th")]
        rows = []
        for fila in tabla.find_all("tr")[1:]:
            cols = [td.text.strip() for td in fila.find_all("td")]
            if len(cols) == len(headers):
                rows.append(cols)

        # 3️ Construir DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # 4️ Preprocesar columnas para filtrado rápido:
        df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce").fillna(0).astype(int)

        def limpiar_titular(val):
            try:
                return int(re.match(r"\d+", val).group())
            except:
                return 0

        df["Titular"] = df["Titular"].apply(limpiar_titular)

        for col in ["Gol", "Asis", "% Pase"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        def extraer_valor_min(texto):
            try:
                texto_clean = texto.replace(",", "")
                match = re.search(r"(\d+\.?\d*)", texto_clean)
                return float(match.group(1)) if match else 0.0
            except:
                return 0.0

        df["ValorNum"] = df["Valor de traspaso"].apply(extraer_valor_min)

        #Antes de guardar, borrar versiones antiguas
        if os.path.exists("jugadores.pkl"):
            os.remove("jugadores.pkl")
        if os.path.exists("full_table.html"):
            os.remove("full_table.html")

        # 5️Guardar el DataFrame preprocesado en un pickle
        df.to_pickle("jugadores.pkl")
        print("✔ jugadores.pkl generado correctamente.")

        # 6️ Generar el HTML completo de la tabla 
        full_html = df.to_html(index=False, classes="table table-striped")
        with open("full_table.html", "w", encoding="utf-8") as f:
            f.write(full_html)
        print("✔ full_table.html generado correctamente.")

        return True  

    except Exception as e:
        print(f"Error procesando BBDD.html: {e}")
        return False  

if __name__ == "__main__":
    procesar_BBDD_html()
