"""Pruebas de los nodos del pipeline ``model_training``."""

import numpy as np
import pandas as pd
import pytest

from telco_churn.pipelines.model_training.nodes import (
    crear_preprocesador,
    dividir_datos,
    entrenar_modelo,
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


VALIDACION_CRUZADA = {"n_splits": 3, "scoring": "average_precision", "random_state": 0}


@pytest.mark.parametrize("escalar", [True, False])
def test_crear_preprocesador_codifica_y_excluye(features, escalar):
    X = features.drop(columns=["Customer ID", "Churn Value"])

    resultado = crear_preprocesador(X, ["Total Charges"], escalar).fit_transform(X)

    columnas = list(
        crear_preprocesador(X, ["Total Charges"], escalar)
        .fit(X)
        .get_feature_names_out()
    )
    assert "Total Charges" not in columnas
    assert "Contract_Month-to-Month" in columnas
    antiguedad = resultado[:, columnas.index("Tenure in Months")]
    if escalar:
        assert antiguedad.mean() == pytest.approx(0, abs=1e-9)
    else:
        assert antiguedad.tolist() == X["Tenure in Months"].tolist()


@pytest.mark.parametrize(
    ("algoritmo", "escalar", "rejilla"),
    [
        ("regresion_logistica", True, {"C": [0.1, 1]}),
        ("random_forest", False, {"max_depth": [2, None]}),
        ("gradient_boosting", False, {"max_depth": [2, 3]}),
    ],
)
def test_entrenar_modelo_elige_hiperparametros_de_la_rejilla(
    features, algoritmo, escalar, rejilla
):
    X_train, X_test, y_train, _ = dividir_datos(features, PARAMETROS_DIVISION)
    configuracion = {
        "nombre": algoritmo,
        "algoritmo": algoritmo,
        "escalar_numericas": escalar,
        "columnas_excluidas": ["Total Charges"],
        "hiperparametros": {"random_state": 0},
        "rejilla": rejilla,
    }

    modelo, resumen = entrenar_modelo(
        X_train, y_train, configuracion, VALIDACION_CRUZADA
    )

    probabilidad = modelo.predict_proba(X_test)[:, 1]
    assert ((probabilidad >= 0) & (probabilidad <= 1)).all()
    assert resumen["modelo"] == algoritmo
    ((parametro, valores),) = rejilla.items()
    assert resumen["mejores_hiperparametros"][parametro] in valores
    assert 0 <= resumen["average_precision_validacion_cruzada"] <= 1
