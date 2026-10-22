"""Nodos del pipeline de entrenamiento: división de datos y modelos."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Algoritmos disponibles para los modelos definidos en parameters_model_training.yml
ALGORITMOS = {
    "regresion_logistica": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": HistGradientBoostingClassifier,
}


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
    X: pd.DataFrame, columnas_excluidas: list[str], escalar_numericas: bool
) -> ColumnTransformer:
    """Crea el preprocesamiento de las variables.

    - Categóricas (texto): *one-hot encoding*. Categorías no vistas en
      entrenamiento se ignoran en lugar de causar un error.
    - Numéricas: estandarización (media 0, desviación 1) si
      ``escalar_numericas`` es verdadero (modelos lineales); los modelos de
      árboles las usan sin cambios.
    - ``columnas_excluidas``: se eliminan (por ejemplo, totales acumulados muy
      correlacionados con la antigüedad).

    Se ajusta dentro del modelo, solo con datos de entrenamiento, para no
    filtrar información del conjunto de prueba.

    Args:
        X: Datos de entrada, para identificar el tipo de cada columna.
        columnas_excluidas: Columnas que el modelo no usa.
        escalar_numericas: Si se estandarizan las variables numéricas.

    Returns:
        ``ColumnTransformer`` sin ajustar.
    """
    columnas = X.drop(columns=columnas_excluidas)
    categoricas = columnas.select_dtypes(exclude="number").columns.tolist()
    numericas = columnas.select_dtypes("number").columns.tolist()
    return ColumnTransformer(
        [
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
            (
                "numericas",
                StandardScaler() if escalar_numericas else "passthrough",
                numericas,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    configuracion: dict[str, Any],
    validacion_cruzada: dict[str, Any],
) -> tuple[Pipeline, dict[str, Any]]:
    """Entrena un modelo ajustando sus hiperparámetros con GridSearchCV.

    Prueba todas las combinaciones de la ``rejilla`` con validación cruzada
    estratificada sobre los datos de entrenamiento, elige la mejor según la
    métrica configurada y reentrena con todos los datos de entrenamiento. El
    conjunto de prueba no se usa aquí.

    Args:
        X_train: Variables de entrenamiento.
        y_train: Objetivo de entrenamiento.
        configuracion: ``nombre``, ``algoritmo``, ``escalar_numericas``,
            ``columnas_excluidas``, ``hiperparametros`` y ``rejilla``.
        validacion_cruzada: ``n_splits``, ``scoring`` y ``random_state``.

    Returns:
        El mejor modelo (preprocesamiento + clasificador) ya entrenado, y un
        resumen de la búsqueda (nombre, mejores hiperparámetros y puntaje).
    """
    algoritmo = ALGORITMOS[configuracion["algoritmo"]]
    modelo = Pipeline(
        [
            (
                "preprocesamiento",
                crear_preprocesador(
                    X_train,
                    configuracion["columnas_excluidas"],
                    configuracion["escalar_numericas"],
                ),
            ),
            ("clasificador", algoritmo(**configuracion["hiperparametros"])),
        ]
    )
    busqueda = GridSearchCV(
        modelo,
        param_grid={
            f"clasificador__{param}": valores
            for param, valores in configuracion["rejilla"].items()
        },
        scoring=validacion_cruzada["scoring"],
        cv=StratifiedKFold(
            n_splits=validacion_cruzada["n_splits"],
            shuffle=True,
            random_state=validacion_cruzada["random_state"],
        ),
        n_jobs=-1,
    ).fit(X_train, y_train)

    resumen = {
        "modelo": configuracion["nombre"],
        "mejores_hiperparametros": {
            param.removeprefix("clasificador__"): valor
            for param, valor in busqueda.best_params_.items()
        },
        f"{validacion_cruzada['scoring']}_validacion_cruzada": busqueda.best_score_,
    }
    return busqueda.best_estimator_, resumen
