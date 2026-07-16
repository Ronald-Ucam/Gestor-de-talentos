import os
import re
import pandas as pd
from bs4 import BeautifulSoup


TEXT_COLUMNS = [
    "Inf", "Nombre", "Valor de traspaso", "Sueldo", "Final", "Posición",
    "Club", "Cedido por", "Procedencia", "Nac", "2ª Nac",
    "Pierna buena", "Cláus. Resc.", "Part"
]

PERCENT_COLUMNS = [
    "% disparos", "Pen %", "% Pase", "Cen.C/I", "Rcg %",
    "Ent P", "Pep %", "Rp %", "Prop. penaltis parados"
]

DECIMAL_COLUMNS = [
    "Min/Par", "Media", "xG", "xG- SP", "xG- HR", "Gol/90", "xA",
    "Asis/90", "TirP/90", "Tir/90", "Oc C/90", "Pas Clv/90",
    "Ps I/90", "Ps C/90", "Pases prog/90", "Reg/90", "Bal aér/90",
    "Cab G/90", "Entr/90", "Rob/90", "Pos Gan/90", "Pos Perd/90",
    "Esprints/90", "Enc/90"
]

INTEGER_COLUMNS = [
    "Edad", "Titular", "Min", "Gol", "Asis", "Disparos", "TaP",
    "Pen", "Pen M", "Fdj", "OCG", "Pas Cl", "Pas I", "Pas C",
    "Pases prog", "Cen.In", "Cen.Com", "Reg", "Cab Int", "Cab",
    "Ent Cl", "Ent C", "FR", "FC", "Ama", "Roj.", "Rob", "Desp",
    "Enc", "Portería imbatida", "BAt", "BDs", "BRe", "Pen. recibidos",
    "Pen. parados", "JPar"
]


def texto_limpio(valor):
    """
    Normaliza textos exportados desde Football Manager.
    Elimina espacios raros, saltos y caracteres NBSP.
    """
    if valor is None:
        return ""

    texto = str(valor).replace("\xa0", " ")
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def es_vacio(valor):
    """
    Detecta valores vacíos o no informativos.
    """
    texto = texto_limpio(valor).lower()
    return texto in ["", "-", "nan", "none", "sin datos", "no disponible"]


def numero_desde_texto(valor, default=0.0, miles=False):
    """
    Convierte un texto en número.

    miles=True se usa en columnas como:
    Min = 4.620 -> 4620
    Pas I = 1.137 -> 1137

    miles=False se usa en columnas decimales como:
    Media = 6.93
    Reg/90 = 3.71
    """
    if es_vacio(valor):
        return default

    texto = texto_limpio(valor).replace(",", ".")
    texto = texto.replace("%", "")
    texto = texto.replace("€", "")
    texto = texto.replace("km", "")
    texto = texto.replace("cm", "")
    texto = texto.replace("kg", "")
    texto = texto.replace("p/a", "")

    limpio = re.sub(r"[^0-9.\-]", "", texto)

    if limpio in ["", "-", ".", "-."]:
        return default

    if miles:
        limpio = limpio.replace(".", "")
    else:
        if limpio.count(".") > 1:
            partes = limpio.split(".")
            limpio = "".join(partes[:-1]) + "." + partes[-1]

    try:
        return float(limpio)
    except ValueError:
        return default


def convertir_float(valor, default=0.0):
    """
    Conversión general a float.
    """
    return float(numero_desde_texto(valor, default=default, miles=False))


def convertir_int(valor, default=0):
    """
    Conversión general a int.
    Trata los puntos como separadores de miles.
    """
    return int(round(numero_desde_texto(valor, default=default, miles=True)))


def convertir_porcentaje(valor, default=0.0):
    """
    Convierte porcentajes como '79%' en 79.0.
    """
    return convertir_float(valor, default=default)


def convertir_distancia(valor, default=0.0):
    """
    Convierte distancias como '299.3 km' en 299.3.
    """
    return convertir_float(valor, default=default)


def extraer_numero_dinero_token(token):
    """
    Convierte valores económicos a millones.

    Ejemplos:
    75M €       -> 75.0
    18.5M €     -> 18.5
    900mil €    -> 0.9
    700m €      -> 0.7
    3.674.000 € -> 3.674
    """
    token = texto_limpio(token)

    if not token:
        return 0.0

    millones = re.search(r"(\d+(?:[.,]\d+)?)\s*M\b", token)
    if millones:
        return float(millones.group(1).replace(",", "."))

    miles = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:mil|m)\b", token, flags=re.IGNORECASE)
    if miles:
        return float(miles.group(1).replace(",", ".")) / 1000

    euros = re.search(r"\d[\d.]*", token)
    if euros:
        bruto = euros.group(0).replace(".", "")
        try:
            return float(bruto) / 1_000_000
        except Exception:
            return 0.0

    return 0.0


def convertir_valor_mercado(valor):
    """
    Convierte el valor de traspaso a millones.

    Si hay rango:
    75M € - 88M € -> 75.0

    Se usa el primer valor porque representa el mínimo estimado,
    que encaja con el filtro ValorNum de la aplicación.
    """
    if es_vacio(valor):
        return 0.0

    texto = texto_limpio(valor)

    if "no a la venta" in texto.lower():
        return 0.0

    primero = texto.split("-")[0].strip()
    return extraer_numero_dinero_token(primero)


def convertir_dinero_euros(valor):
    """
    Convierte sueldo o cantidades económicas a euros.

    Ejemplo:
    3.674.000 € p/a -> 3674000.0
    """
    if es_vacio(valor):
        return 0.0

    texto = texto_limpio(valor)

    if "no a la venta" in texto.lower():
        return 0.0

    valor_millones = extraer_numero_dinero_token(texto)
    return valor_millones * 1_000_000


def limpiar_partidos(valor):
    """
    Limpia la columna Part.

    Ejemplos:
    27 (2) -> total 29, titular 27, suplente 2
    51     -> total 51, titular 51, suplente 0
    -      -> total 0, titular 0, suplente 0
    """
    texto = texto_limpio(valor)

    if es_vacio(texto):
        return 0, 0, 0

    match = re.match(r"^(\d+)(?:\s*\((\d+)\))?", texto)

    if not match:
        return 0, 0, 0

    titular = int(match.group(1))
    suplente = int(match.group(2)) if match.group(2) else 0
    total = titular + suplente

    return total, titular, suplente


def extraer_tabla_html(html_path=None):
    """
    Lee un archivo HTML exportado desde Football Manager
    y extrae la tabla principal sin limpiar todavía los datos.
    """
    if html_path is None:
        html_path = os.path.join(os.getcwd(), "BBDD.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"No se encontró el archivo HTML: {html_path}")

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    tabla = soup.find("table")

    if tabla is None:
        raise ValueError("No se encontró ninguna tabla en el archivo HTML.")

    headers = [
        texto_limpio(th.get_text(" ", strip=True))
        for th in tabla.find_all("th")
    ]

    rows = []

    for fila in tabla.find_all("tr")[1:]:
        cols = [
            texto_limpio(td.get_text(" ", strip=True))
            for td in fila.find_all("td")
        ]

        if len(cols) == len(headers):
            rows.append(cols)

    if not rows:
        raise ValueError("La tabla HTML no contiene filas válidas.")

    return pd.DataFrame(rows, columns=headers)


def limpiar_dataframe_jugadores(df):
    """
    Pipeline centralizado de limpieza.

    Esta función es el único punto donde se transforman los datos:
    - texto
    - números
    - porcentajes
    - importes económicos
    - altura/peso
    - minutos
    - partidos
    - métricas deportivas
    """
    df = df.copy()

    df.columns = [texto_limpio(col) for col in df.columns]

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(texto_limpio)

    for col in TEXT_COLUMNS:
        if col in df.columns and col not in [
            "Valor de traspaso", "Sueldo", "Cláus. Resc.", "Part"
        ]:
            df[col] = df[col].apply(
                lambda x: "" if es_vacio(x) else texto_limpio(x)
            )

    if "Valor de traspaso" in df.columns:
        df["ValorNum"] = df["Valor de traspaso"].apply(convertir_valor_mercado)

    if "Sueldo" in df.columns:
        df["SueldoNum"] = df["Sueldo"].apply(convertir_dinero_euros)

    if "Cláus. Resc." in df.columns:
        df["ClausulaNum"] = df["Cláus. Resc."].apply(convertir_valor_mercado)

    if "Altura" in df.columns:
        df["Altura"] = df["Altura"].apply(convertir_int)

    if "Peso" in df.columns:
        df["Peso"] = df["Peso"].apply(convertir_int)

    if "Part" in df.columns:
        partidos = df["Part"].apply(limpiar_partidos)
        df["PartidosTotal"] = partidos.apply(lambda x: x[0])
        df["PartidosTitular"] = partidos.apply(lambda x: x[1])
        df["PartidosSuplente"] = partidos.apply(lambda x: x[2])

    for col in INTEGER_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(convertir_int)

    for col in DECIMAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(convertir_float)

    for col in PERCENT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(convertir_porcentaje)

    if "Distancia" in df.columns:
        df["Distancia"] = df["Distancia"].apply(convertir_distancia)

    return df


def validar_dataframe_jugadores(df):
    """
    Comprueba que el DataFrame tiene las columnas mínimas
    que necesita la aplicación.
    """
    columnas_obligatorias = ["Nombre", "Edad", "Posición", "Club"]

    faltantes = [
        col for col in columnas_obligatorias
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas obligatorias en el HTML: {faltantes}"
        )

    if df.empty:
        raise ValueError("El DataFrame está vacío después de procesar el HTML.")

    return True


def limpiar_json_fila(fila):
    """
    Prepara una fila del DataFrame para guardarla en JSON dentro de la BD.
    Convierte tipos de NumPy/Pandas a tipos normales de Python.
    """
    datos = {}

    for clave, valor in fila.fillna("").to_dict().items():
        if hasattr(valor, "item"):
            valor = valor.item()

        if pd.isna(valor):
            valor = ""

        datos[str(clave)] = valor

    return datos


def extraer_dataframe_desde_html(html_path=None):
    """
    Función principal de extracción + limpieza.
    Devuelve siempre un DataFrame ya limpio.
    """
    df = extraer_tabla_html(html_path)
    df = limpiar_dataframe_jugadores(df)
    validar_dataframe_jugadores(df)
    return df


def guardar_dataframe_demo(df):
    """
    Guarda el DataFrame limpio como demo global de la aplicación.
    """
    df.to_pickle("jugadores.pkl")
    print("✔ jugadores.pkl generado correctamente.")

    full_html = df.to_html(index=False, classes="table table-striped")

    with open("full_table.html", "w", encoding="utf-8") as f:
        f.write(full_html)

    print("✔ full_table.html generado correctamente.")


def procesar_BBDD_html(html_path=None, guardar_demo=True):
    """
    Procesa el HTML exportado desde Football Manager.

    Si guardar_demo=True:
        genera jugadores.pkl y full_table.html.

    Si guardar_demo=False:
        devuelve el DataFrame limpio sin modificar la demo global.
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