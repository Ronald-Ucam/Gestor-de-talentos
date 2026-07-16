import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def normalizar_columnas(df):
    """
    Normaliza nombres de columnas para trabajar tanto con jugadores.pkl
    como con BBDD.html.
    """

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " ".join([str(x) for x in col if str(x) != "nan"]).strip()
            for col in df.columns
        ]

    df.columns = df.columns.astype(str).str.strip()

    posibles_posicion = [
        "Pos",
        "POS",
        "Pos.",
        "Posicion",
        "posición",
        "posicion"
    ]

    for col in df.columns:
        if col in posibles_posicion:
            df = df.rename(columns={col: "Posición"})

    if "Posición" not in df.columns:
        for col in df.columns:
            if "pos" in col.lower():
                df = df.rename(columns={col: "Posición"})
                break

    return df


def cargar_dataframe():
    """
    Carga primero jugadores.pkl.
    Si no existe, intenta leer BBDD.html.
    """

    if os.path.exists("jugadoress.pkl"):
        print("Datos cargados desde jugadoress.pkl.")
        df = pd.read_pickle("jugadoress.pkl")
        return normalizar_columnas(df)

    if os.path.exists("BBDD.html"):
        print("Datos cargados desde BBDD.html.")
        tablas = pd.read_html("BBDD.html")

        # Se elige la tabla más grande, normalmente la de jugadores
        df = max(tablas, key=lambda t: t.shape[0] * t.shape[1])
        return normalizar_columnas(df)

    raise FileNotFoundError(
        "No se encontró jugadores.pkl ni BBDD.html en la carpeta del proyecto."
    )


def limpiar_numero(serie):
    """
    Convierte valores como '-', '75%', '1,25' o texto en números.
    """

    s = serie.astype(str)
    s = s.replace("-", np.nan)
    s = s.str.replace(",", ".", regex=False)
    s = s.str.replace(r"[^0-9\.]", "", regex=True)
    s = s.replace("", np.nan)

    return pd.to_numeric(s, errors="coerce")


def preparar_delanteros(df):
    """
    Prepara los delanteros para comparar algoritmos.

    Pasos:
    1. Filtra jugadores DL.
    2. Selecciona variables ofensivas.
    3. Limpia valores numéricos.
    4. Crea métricas derivadas.
    5. Elimina columnas con demasiados nulos.
    6. Rellena nulos con mediana.
    7. Normaliza con StandardScaler.
    """

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if "Posición" not in df.columns:
        print("\nColumnas encontradas en el DataFrame:")
        print(df.columns.tolist())
        raise ValueError(
            "No existe la columna 'Posición'. Revisa el nombre real de la columna."
        )

    df_fw = df[
        df["Posición"].astype(str).str.contains(r"\bDL\b", na=False)
    ].copy()

    if df_fw.empty:
        raise ValueError("No se han encontrado delanteros con posición DL.")

    columnas_base = [
        "Gol/90",
        "Asis/90",
        "Reg/90",
        "% Pase",
        "Disparos",
        "Min/Par",
        "OC/90"
    ]

    cols = [c for c in columnas_base if c in df_fw.columns]

    if len(cols) < 2:
        raise ValueError(f"No hay suficientes columnas para delanteros: {cols}")

    for c in cols:
        df_fw[c] = limpiar_numero(df_fw[c])

    extras = []

    # Efectividad: goles por disparo
    if "Gol/90" in df_fw.columns and "Disparos" in df_fw.columns:
        df_fw["Conv%"] = df_fw["Gol/90"] / df_fw["Disparos"].replace(0, np.nan)
        df_fw["Conv%"] = df_fw["Conv%"].replace([np.inf, -np.inf], np.nan)
        extras.append("Conv%")

    # Goles esperados por 90 si existe xG
    if "xG" in df_fw.columns and "Min/Par" in df_fw.columns:
        df_fw["xG"] = limpiar_numero(df_fw["xG"])
        df_fw["xG/90"] = df_fw["xG"] / df_fw["Min/Par"].replace(0, np.nan)
        df_fw["xG/90"] = df_fw["xG/90"].replace([np.inf, -np.inf], np.nan)
        extras.append("xG/90")

    cols = [c for c in cols + extras if c in df_fw.columns]

    # Mantener columnas con al menos 70% de datos válidos
    cols = [
        c for c in cols
        if df_fw[c].notna().sum() >= len(df_fw) * 0.7
    ]

    if len(cols) < 2:
        raise ValueError(f"No quedan suficientes columnas válidas: {cols}")

    df_fw[cols] = df_fw[cols].replace([np.inf, -np.inf], np.nan)
    df_fw[cols] = df_fw[cols].fillna(df_fw[cols].median(numeric_only=True))
    df_fw[cols] = df_fw[cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_fw[cols].astype(float).values)

    return df_fw.reset_index(drop=True), cols, X


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(X, labels, excluir_ruido=True):
    """
    Calcula métricas internas para clustering.

    En DBSCAN, la etiqueta -1 representa ruido.
    Si excluir_ruido=True, las métricas se calculan sin esos puntos.
    """

    labels = np.asarray(labels)

    n_total = len(labels)
    n_ruido = int(np.sum(labels == -1))
    ruido_pct = round((n_ruido / n_total) * 100, 2)

    if excluir_ruido and n_ruido > 0:
        mascara = labels != -1
        X_eval = X[mascara]
        labels_eval = labels[mascara]
    else:
        X_eval = X
        labels_eval = labels

    etiquetas = set(labels_eval.tolist())
    n_clusters = len(etiquetas)

    if len(X_eval) < 3 or n_clusters < 2 or n_clusters >= len(X_eval):
        return {
            "valido": False,
            "clusters": n_clusters,
            "ruido": n_ruido,
            "ruido_pct": ruido_pct,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "motivo": "No hay suficientes clústeres válidos para calcular métricas"
        }

    return {
        "valido": True,
        "clusters": n_clusters,
        "ruido": n_ruido,
        "ruido_pct": ruido_pct,
        "silhouette": round(float(silhouette_score(X_eval, labels_eval)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X_eval, labels_eval)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_eval, labels_eval)), 4),
        "motivo": "OK"
    }


def agregar_resultado(resultados, modelo, tipo, parametros, X, labels, tiempo, extra=None):
    """
    Añade una fila de resultado a la evaluación.
    """

    metricas = calcular_metricas(X, labels)

    fila = {
        "modelo": modelo,
        "tipo": tipo,
        "parametros": parametros,
        "tiempo_segundos": round(float(tiempo), 5),
        **metricas
    }

    if extra:
        fila.update(extra)

    resultados.append(fila)


# ============================================================
# EVALUACIÓN DE MODELOS
# ============================================================

def evaluar_modelos(X):
    """
    Evalúa varios modelos sobre el mismo conjunto de datos normalizado.
    """

    resultados = []

    max_k = min(8, len(X) - 1)

    if max_k < 2:
        raise ValueError("No hay suficientes jugadores para probar clustering.")

    rango_k = range(2, max_k + 1)

    # --------------------------------------------------------
    # 1. K-MEANS
    # --------------------------------------------------------
    for k in rango_k:
        inicio = time.time()

        modelo = KMeans(
            n_clusters=k,
            random_state=0,
            n_init="auto"
        )

        labels = modelo.fit_predict(X)
        tiempo = time.time() - inicio

        agregar_resultado(
            resultados=resultados,
            modelo=f"K-Means k={k}",
            tipo="particional",
            parametros=f"k={k}",
            X=X,
            labels=labels,
            tiempo=tiempo,
            extra={
                "inertia": round(float(modelo.inertia_), 4)
            }
        )

    # --------------------------------------------------------
    # 2. AGGLOMERATIVE CLUSTERING
    # --------------------------------------------------------
    for k in rango_k:
        inicio = time.time()

        modelo = AgglomerativeClustering(n_clusters=k)
        labels = modelo.fit_predict(X)

        tiempo = time.time() - inicio

        agregar_resultado(
            resultados=resultados,
            modelo=f"Agglomerative k={k}",
            tipo="jerarquico",
            parametros=f"k={k}",
            X=X,
            labels=labels,
            tiempo=tiempo
        )

    # --------------------------------------------------------
    # 3. GAUSSIAN MIXTURE
    # --------------------------------------------------------
    for k in rango_k:
        inicio = time.time()

        modelo = GaussianMixture(
            n_components=k,
            random_state=0
        )

        labels = modelo.fit_predict(X)
        tiempo = time.time() - inicio

        agregar_resultado(
            resultados=resultados,
            modelo=f"Gaussian Mixture k={k}",
            tipo="probabilistico",
            parametros=f"k={k}",
            X=X,
            labels=labels,
            tiempo=tiempo
        )

    # --------------------------------------------------------
    # 4. DBSCAN
    # --------------------------------------------------------
    for eps in [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
        for min_samples in [3, 5, 8]:
            inicio = time.time()

            modelo = DBSCAN(
                eps=eps,
                min_samples=min_samples
            )

            labels = modelo.fit_predict(X)
            tiempo = time.time() - inicio

            agregar_resultado(
                resultados=resultados,
                modelo=f"DBSCAN eps={eps}, min_samples={min_samples}",
                tipo="densidad",
                parametros=f"eps={eps}, min_samples={min_samples}",
                X=X,
                labels=labels,
                tiempo=tiempo
            )

    df_resultados = pd.DataFrame(resultados)

    df_validos = df_resultados[df_resultados["valido"] == True].copy()

    if df_validos.empty:
        return df_resultados, None

    # Ranking: cuanto menor sea la puntuación final, mejor.
    df_validos["rank_silhouette"] = df_validos["silhouette"].rank(
        ascending=False,
        method="min"
    )
    df_validos["rank_davies"] = df_validos["davies_bouldin"].rank(
        ascending=True,
        method="min"
    )
    df_validos["rank_calinski"] = df_validos["calinski_harabasz"].rank(
        ascending=False,
        method="min"
    )
    df_validos["rank_ruido"] = df_validos["ruido_pct"].rank(
        ascending=True,
        method="min"
    )
    df_validos["rank_tiempo"] = df_validos["tiempo_segundos"].rank(
        ascending=True,
        method="min"
    )

    df_validos["puntuacion_final"] = (
        df_validos["rank_silhouette"] * 0.35 +
        df_validos["rank_davies"] * 0.25 +
        df_validos["rank_calinski"] * 0.25 +
        df_validos["rank_ruido"] * 0.10 +
        df_validos["rank_tiempo"] * 0.05
    )

    df_validos = df_validos.sort_values("puntuacion_final", ascending=True)

    mejor = df_validos.iloc[0].to_dict()

    df_resultados = df_resultados.merge(
        df_validos[["modelo", "puntuacion_final"]],
        on="modelo",
        how="left"
    )

    df_resultados = df_resultados.sort_values(
        by=["puntuacion_final", "silhouette"],
        ascending=[True, False],
        na_position="last"
    )

    return df_resultados, mejor


# ============================================================
# GRÁFICOS
# ============================================================
def graficar_comparacion_algoritmos(df_resultados):
    """
    Compara los mejores resultados de cada tipo de algoritmo usando
    la puntuación final. Cuanto menor sea la puntuación, mejor.
    """

    df_validos = df_resultados[df_resultados["valido"] == True].copy()

    if df_validos.empty:
        print("No hay resultados válidos para graficar comparación de algoritmos.")
        return

    mejores = []

    for tipo in df_validos["tipo"].unique():
        df_tipo = df_validos[df_validos["tipo"] == tipo].copy()
        df_tipo = df_tipo.sort_values("puntuacion_final", ascending=True)
        mejores.append(df_tipo.iloc[0])

    df_mejores = pd.DataFrame(mejores)

    plt.figure(figsize=(10, 6))
    plt.bar(df_mejores["modelo"], df_mejores["puntuacion_final"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Puntuación final")
    plt.title("Comparación de algoritmos según puntuación final")

    for i, row in df_mejores.reset_index(drop=True).iterrows():
        texto = f"Ruido: {row['ruido_pct']}%"
        plt.text(
            i,
            row["puntuacion_final"],
            texto,
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("grafico_comparacion_algoritmos.png", dpi=160)
    plt.close()

    print("Gráfico guardado: grafico_comparacion_algoritmos.png")


def graficar_kmeans_por_k(df_resultados):
    """
    Genera:
    1. Silhouette de K-Means según k.
    2. Inercia de K-Means según k.
    """

    df_kmeans = df_resultados[
        df_resultados["modelo"].astype(str).str.contains("K-Means")
    ].copy()

    if df_kmeans.empty:
        print("No hay resultados de K-Means para graficar.")
        return

    df_kmeans["k"] = df_kmeans["modelo"].str.extract(r"k=(\d+)").astype(int)
    df_kmeans = df_kmeans.sort_values("k")

    # Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(df_kmeans["k"], df_kmeans["silhouette"], marker="o")
    plt.xlabel("Número de clústeres (k)")
    plt.ylabel("Silhouette Score")
    plt.title("K-Means: calidad del clustering según k")
    plt.xticks(df_kmeans["k"])
    plt.tight_layout()
    plt.savefig("grafico_kmeans_silhouette.png", dpi=160)
    plt.close()

    print("Gráfico guardado: grafico_kmeans_silhouette.png")

    # Inercia
    if "inertia" in df_kmeans.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df_kmeans["k"], df_kmeans["inertia"], marker="o")
        plt.xlabel("Número de clústeres (k)")
        plt.ylabel("Inercia")
        plt.title("K-Means: método del codo")
        plt.xticks(df_kmeans["k"])
        plt.tight_layout()
        plt.savefig("grafico_kmeans_codo.png", dpi=160)
        plt.close()

        print("Gráfico guardado: grafico_kmeans_codo.png")


def graficar_dbscan_ruido(df_resultados):
    """
    Grafica el porcentaje de ruido generado por DBSCAN para sus configuraciones.
    """

    df_dbscan = df_resultados[
        df_resultados["modelo"].astype(str).str.contains("DBSCAN")
    ].copy()

    if df_dbscan.empty:
        print("No hay resultados de DBSCAN para graficar.")
        return

    df_dbscan = df_dbscan.sort_values("ruido_pct", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(df_dbscan["modelo"], df_dbscan["ruido_pct"])
    plt.xticks(rotation=90)
    plt.ylabel("Porcentaje de ruido (%)")
    plt.title("DBSCAN: porcentaje de jugadores marcados como ruido")
    plt.tight_layout()
    plt.savefig("grafico_dbscan_ruido.png", dpi=160)
    plt.close()

    print("Gráfico guardado: grafico_dbscan_ruido.png")


def graficar_pca_kmeans_delanteros(df_fw, cols, k=6):
    """
    Aplica K-Means a delanteros y representa los clústeres en 2D usando PCA.
    """

    X = df_fw[cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.astype(float).values)

    km = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=22)
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.title(f"Delanteros agrupados con K-Means (k={k})")
    plt.colorbar(scatter, label="Clúster")
    plt.tight_layout()
    plt.savefig("grafico_pca_kmeans_delanteros.png", dpi=160)
    plt.close()

    print("Gráfico guardado: grafico_pca_kmeans_delanteros.png")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    print("\n==============================================")
    print(" EVALUACIÓN DE IA PARA DELANTEROS")
    print("==============================================\n")

    df = cargar_dataframe()

    df_fw, cols, X = preparar_delanteros(df)

    print(f"Jugadores delanteros evaluados: {len(df_fw)}")
    print(f"Columnas utilizadas: {cols}\n")

    df_resultados, mejor = evaluar_modelos(X)

    columnas_mostrar = [
        "modelo",
        "tipo",
        "clusters",
        "ruido",
        "ruido_pct",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "tiempo_segundos",
        "puntuacion_final",
        "motivo"
    ]

    columnas_mostrar = [
        c for c in columnas_mostrar
        if c in df_resultados.columns
    ]

    print("RESULTADOS ORDENADOS:\n")
    print(df_resultados[columnas_mostrar].to_string(index=False))

    salida = "resultados_ia_delanteros.csv"
    df_resultados.to_csv(salida, index=False, encoding="utf-8-sig")

    print(f"\nResultados guardados en: {salida}")

    # Generar gráficos
    graficar_comparacion_algoritmos(df_resultados)
    graficar_kmeans_por_k(df_resultados)
    graficar_dbscan_ruido(df_resultados)
    graficar_pca_kmeans_delanteros(df_fw, cols, k=6)

    if mejor:
        print("\n==============================================")
        print(" MEJOR MODELO SEGÚN ESTA EVALUACIÓN")
        print("==============================================")
        print(f"Modelo: {mejor['modelo']}")
        print(f"Tipo: {mejor['tipo']}")
        print(f"Clusters: {mejor['clusters']}")
        print(f"Ruido: {mejor['ruido']} jugadores ({mejor['ruido_pct']}%)")
        print(f"Silhouette: {mejor['silhouette']}")
        print(f"Davies-Bouldin: {mejor['davies_bouldin']}")
        print(f"Calinski-Harabasz: {mejor['calinski_harabasz']}")
        print(f"Tiempo: {mejor['tiempo_segundos']} segundos")
        print(f"Puntuación final: {round(mejor['puntuacion_final'], 4)}")
    else:
        print("\nNo se encontró ningún modelo válido.")


if __name__ == "__main__":
    main()