"""Nodos del pipeline de entrenamiento: división de datos y modelos."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def dividir_datos(
    features: pd.DataFrame, parametros: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos en entrenamiento y prueba, de forma estratificada.

    La estratificación conserva la proporción de churn (26.5 %) en ambos
    conjuntos. La llave del cliente queda como índice para poder identificar
    a cada cliente en las predicciones.

    Args:
        features: Tabla de variables con llave y objetivo.
        parametros: ``llave``, ``objetivo``, ``test_size`` y ``random_state``.

    Returns:
        ``X_train``, ``X_test``, ``y_train`` y ``y_test``.
    """
    datos = features.set_index(parametros["llave"])
    X = datos.drop(columns=parametros["objetivo"])
    y = datos[parametros["objetivo"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=parametros["test_size"],
        stratify=y,
        random_state=parametros["random_state"],
    )
    return X_train, X_test, y_train, y_test


def crear_preprocesador(
    X: pd.DataFrame, columnas_excluidas: list[str]
) -> ColumnTransformer:
    """Crea el preprocesamiento para modelos lineales.

    - Categóricas (texto): *one-hot encoding*. Categorías no vistas en
      entrenamiento se ignoran en lugar de causar un error.
    - Numéricas: estandarización (media 0, desviación 1).
    - ``columnas_excluidas``: se eliminan (por ejemplo, totales acumulados muy
      correlacionados con la antigüedad).

    Se ajusta dentro del modelo, solo con datos de entrenamiento, para no
    filtrar información del conjunto de prueba.

    Args:
        X: Datos de entrada, para identificar el tipo de cada columna.
        columnas_excluidas: Columnas que el modelo no usa.

    Returns:
        ``ColumnTransformer`` sin ajustar.
    """
    columnas = X.drop(columns=columnas_excluidas)
    categoricas = columnas.select_dtypes(exclude="number").columns.tolist()
    numericas = columnas.select_dtypes("number").columns.tolist()
    return ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ("numericas", StandardScaler(), numericas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def entrenar_regresion_logistica(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    columnas_excluidas: list[str],
    hiperparametros: dict[str, Any],
) -> Pipeline:
    """Entrena una regresión logística con su preprocesamiento.

    Es el primer modelo: simple, rápido e interpretable (cada coeficiente indica
    si una variable aumenta o reduce el riesgo de churn).

    Args:
        X_train: Variables de entrenamiento.
        y_train: Objetivo de entrenamiento.
        columnas_excluidas: Columnas que el modelo no usa.
        hiperparametros: Argumentos para ``LogisticRegression``.

    Returns:
        Pipeline de scikit-learn (preprocesamiento + modelo) ya entrenado.
    """
    modelo = Pipeline(
        [
            ("preprocesamiento", crear_preprocesador(X_train, columnas_excluidas)),
            ("clasificador", LogisticRegression(**hiperparametros)),
        ]
    )
    return modelo.fit(X_train, y_train)
