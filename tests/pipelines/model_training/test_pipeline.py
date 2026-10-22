"""Pruebas de los nodos del pipeline ``model_training``."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.pipelines.model_training.nodes import (
    crear_preprocesador,
    dividir_datos,
    entrenar_regresion_logistica,
)

CLIENTES = 100
CLIENTES_PRUEBA = 20
PARAMETROS_DIVISION = {
    "llave": "Customer ID",
    "objetivo": "Churn Value",
    "test_size": CLIENTES_PRUEBA / CLIENTES,
    "random_state": 0,
}


@pytest.fixture
def features() -> pd.DataFrame:
    """Tabla de variables sintética: 100 clientes, 25 % churn."""
    rng = np.random.default_rng(0)
    n = CLIENTES
    churn = np.array([1] * 25 + [0] * 75)
    return pd.DataFrame(
        {
            "Customer ID": [f"C{i:03d}" for i in range(n)],
            "Tenure in Months": np.where(churn == 1, 3, 40) + rng.integers(0, 5, n),
            "Total Charges": rng.uniform(20, 5000, n),
            "Contract": np.where(churn == 1, "Month-to-Month", "Two Year"),
            "Churn Value": churn,
        }
    )


def test_dividir_datos_es_estratificada_y_sin_fugas(features):
    X_train, X_test, y_train, y_test = dividir_datos(features, PARAMETROS_DIVISION)

    assert len(X_test) == CLIENTES_PRUEBA
    assert len(X_train) == CLIENTES - CLIENTES_PRUEBA
    assert y_train.mean() == pytest.approx(y_test.mean())
    assert "Churn Value" not in X_train.columns
    assert X_train.index.name == "Customer ID"
    assert not set(X_train.index) & set(X_test.index)


def test_crear_preprocesador_codifica_escala_y_excluye(features):
    X = features.drop(columns=["Customer ID", "Churn Value"])

    transformado = crear_preprocesador(X, ["Total Charges"]).fit(X)

    columnas = list(transformado.get_feature_names_out())
    assert "Total Charges" not in columnas
    assert "Contract_Month-to-Month" in columnas
    assert "Tenure in Months" in columnas


def test_entrenar_regresion_logistica_predice_probabilidades(features):
    X_train, X_test, y_train, _ = dividir_datos(features, PARAMETROS_DIVISION)

    modelo = entrenar_regresion_logistica(
        X_train, y_train, ["Total Charges"], {"max_iter": 1000}
    )

    probabilidad = modelo.predict_proba(X_test)[:, 1]
    assert probabilidad.shape == (len(X_test),)
    assert ((probabilidad >= 0) & (probabilidad <= 1)).all()
