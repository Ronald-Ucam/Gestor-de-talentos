# preprocesar_tabla.py

import pandas as pd
from bs4 import BeautifulSoup
import re
import os

# 1️⃣ Ruta al archivo HTML con la tabla original
html_path = os.path.join(os.getcwd(), "BBDD.html")
if not os.path.exists(html_path):
    raise FileNotFoundError("No se encontró BBDD.html en la raíz del proyecto.")

# 2️⃣ Leer el HTML y extraer la tabla con BeautifulSoup
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

# 3️⃣ Construir DataFrame
df = pd.DataFrame(rows, columns=headers)

# 4️⃣ Preprocesar columnas para filtrado rápido:

# ● Convertir “Edad” a entero
df["Edad"] = pd.to_numeric(df["Edad"], errors="coerce").fillna(0).astype(int)

# ● Limpiar “Titular” extrayendo solo el número
def limpiar_titular(val):
    try:
        return int(re.match(r"\d+", val).group())
    except:
        return 0

df["Titular"] = df["Titular"].apply(limpiar_titular)

# ● Convertir “Gol”, “Asis” y “% Pase” a float
for col in ["Gol", "Asis", "% Pase"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

# ● Extraer valor mínimo de “Valor de traspaso” (por ejemplo: “34M € - 41M €” → 34.0)
def extraer_valor_min(texto):
    try:
        texto_clean = texto.replace(",", "")
        match = re.search(r"(\d+\.?\d*)", texto_clean)
        return float(match.group(1)) if match else 0.0
    except:
        return 0.0

df["ValorNum"] = df["Valor de traspaso"].apply(extraer_valor_min)

# 5️⃣ Guardar el DataFrame preprocesado en un pickle (carga muy rápida luego)
df.to_pickle("jugadores.pkl")
print("✔ jugadores.pkl generado correctamente.")

# 6️⃣ Generar el HTML completo de la tabla (sin filtros) y guardarlo
full_html = df.to_html(index=False, classes="table table-striped")
with open("full_table.html", "w", encoding="utf-8") as f:
    f.write(full_html)

print("✔ full_table.html generado correctamente.")
