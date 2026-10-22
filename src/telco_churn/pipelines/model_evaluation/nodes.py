"""Nodos del pipeline de evaluación: líneas base, métricas y comparación."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def calcular_metricas(
    y_real: pd.Series, y_predicho: pd.Series, puntaje: pd.Series
) -> dict[str, float]:
    """Calcula las métricas de clasificación del proyecto.

    Args:
        y_real: Objetivo real (0/1).
        y_predicho: Predicción (0/1).
        puntaje: Probabilidad o puntaje de churn (mayor = más riesgo).

    Returns:
        Recall, precisión, F1, ROC-AUC, PR-AUC y *accuracy*.
    """
    return {
        "recall": recall_score(y_real, y_predicho, zero_division=0),
        "precision": precision_score(y_real, y_predicho, zero_division=0),
        "f1": f1_score(y_real, y_predicho, zero_division=0),
        "roc_auc": roc_auc_score(y_real, puntaje),
        "pr_auc": average_precision_score(y_real, puntaje),
        "accuracy": accuracy_score(y_real, y_predicho),
    }


def evaluar_lineas_base(
    X_test: pd.DataFrame, y_test: pd.Series, regla_negocio: dict[str, str]
) -> pd.DataFrame:
    """Evalúa las líneas base que cualquier modelo debe superar.

    - **Clase mayoritaria:** predice que nadie se va.
    - **Regla de negocio:** predice churn para todos los clientes que cumplen
      una condición (por defecto, contrato mes a mes).

    Args:
        X_test: Variables de prueba.
        y_test: Objetivo de prueba.
        regla_negocio: ``columna`` y ``valor`` que definen la regla.

    Returns:
        Tabla con una fila de métricas por línea base.
    """
    nadie_se_va = pd.Series(0, index=y_test.index)
    regla = X_test[regla_negocio["columna"]].eq(regla_negocio["valor"]).astype(int)
    return pd.DataFrame(
        {
            "Línea base: clase mayoritaria": calcular_metricas(
                y_test, nadie_se_va, nadie_se_va
            ),
            f"Línea base: {regla_negocio['columna']} = {regla_negocio['valor']}": (
                calcular_metricas(y_test, regla, regla)
            ),
        }
    ).T


def evaluar_modelo(
    modelo: Pipeline,
    busqueda: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    umbral: float,
) -> dict[str, Any]:
    """Evalúa un modelo entrenado en el conjunto de prueba.

    Args:
        modelo: Modelo de scikit-learn con ``predict_proba``.
        busqueda: Resumen de GridSearchCV (nombre, mejores hiperparámetros y
            puntaje de validación cruzada).
        X_test: Variables de prueba.
        y_test: Objetivo de prueba.
        umbral: Probabilidad a partir de la cual se predice churn.

    Returns:
        Nombre del modelo, métricas en prueba y resumen de la búsqueda.
    """
    probabilidad = pd.Series(modelo.predict_proba(X_test)[:, 1], index=y_test.index)
    metricas = calcular_metricas(
        y_test, (probabilidad >= umbral).astype(int), probabilidad
    )
    return {
        **metricas,
        **busqueda,
        "mejores_hiperparametros": str(busqueda["mejores_hiperparametros"]),
    }


def comparar_modelos(
    lineas_base: pd.DataFrame, *evaluaciones: dict[str, Any]
) -> pd.DataFrame:
    """Reúne las métricas de las líneas base y los modelos en una tabla.

    Args:
        lineas_base: Métricas de las líneas base (una fila por línea base).
        *evaluaciones: Resultado de ``evaluar_modelo`` para cada modelo.

    Returns:
        Tabla de comparación, una fila por modelo, con las métricas
        redondeadas a 3 decimales.
    """
    modelos = pd.DataFrame(list(evaluaciones)).set_index("modelo")
    tabla = pd.concat([lineas_base, modelos]).rename_axis("modelo").reset_index()
    return tabla.round(3)
