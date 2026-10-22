"""Pruebas de los nodos del pipeline ``model_evaluation``."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.pipelines.model_evaluation.nodes import (
    calcular_metricas,
    comparar_modelos,
    evaluar_lineas_base,
    evaluar_modelo,
)


class ModeloFijo:
    """Modelo de prueba que devuelve probabilidades conocidas."""

    def __init__(self, probabilidades: list[float]):
        self.probabilidades = np.array(probabilidades)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve las probabilidades en el formato de scikit-learn."""
        return np.column_stack([1 - self.probabilidades, self.probabilidades])


@pytest.fixture
def datos_prueba() -> tuple[pd.DataFrame, pd.Series]:
    """Cuatro clientes: dos se van (mes a mes) y dos se quedan."""
    indice = pd.Index(["A", "B", "C", "D"], name="Customer ID")
    X_test = pd.DataFrame(
        {
            "Contract": [
                "Month-to-Month",
                "Month-to-Month",
                "Month-to-Month",
                "Two Year",
            ]
        },
        index=indice,
    )
    y_test = pd.Series([1, 1, 0, 0], index=indice, name="Churn Value")
    return X_test, y_test


def test_calcular_metricas_prediccion_perfecta():
    y = pd.Series([1, 0, 1, 0])

    metricas = calcular_metricas(y, y, y)

    assert all(valor == 1.0 for valor in metricas.values())


def test_evaluar_lineas_base(datos_prueba):
    X_test, y_test = datos_prueba

    resultado = evaluar_lineas_base(
        X_test, y_test, {"columna": "Contract", "valor": "Month-to-Month"}
    )

    mayoritaria, regla = resultado.iloc[0], resultado.iloc[1]
    assert mayoritaria["recall"] == 0
    assert mayoritaria["accuracy"] == pytest.approx(0.5)
    assert regla["recall"] == 1
    assert regla["precision"] == pytest.approx(2 / 3)


def test_evaluar_modelo_aplica_el_umbral(datos_prueba):
    X_test, y_test = datos_prueba
    modelo = ModeloFijo([0.9, 0.4, 0.6, 0.1])

    con_umbral_alto = evaluar_modelo(modelo, X_test, y_test, umbral=0.5)
    con_umbral_bajo = evaluar_modelo(modelo, X_test, y_test, umbral=0.3)

    assert con_umbral_alto["recall"] == pytest.approx(0.5)
    assert con_umbral_bajo["recall"] == 1
    assert con_umbral_alto["roc_auc"] == con_umbral_bajo["roc_auc"]


def test_comparar_modelos_une_lineas_base_y_modelos():
    metricas = {"recall": 0.8, "precision": 0.6}
    lineas_base = pd.DataFrame({"Base": {"recall": 0.0, "precision": 0.0}}).T

    resultado = comparar_modelos(lineas_base, metricas)

    assert resultado["modelo"].tolist() == ["Base", "Regresión logística"]
