"""Pruebas de los nodos del pipeline ``feature_engineering``."""

import pandas as pd
import pytest

from telco_churn.pipelines.feature_engineering.nodes import (
    codificar_si_no,
    crear_variables,
)


@pytest.fixture
def clientes() -> pd.DataFrame:
    """Tabla limpia mínima con tres clientes."""
    return pd.DataFrame(
        {
            "Customer ID": ["0001-A", "0002-B", "0003-C"],
            "Tenure in Months": [3, 6, 40],
            "Phone Service": ["Yes", "Yes", "No"],
            "Online Security": ["No", "Yes", "Yes"],
            "Premium Tech Support": ["No", "Yes", "No"],
            "Contract": ["Month-to-Month", "One Year", "Two Year"],
            "Churn Value": [1, 0, 0],
        }
    )


def test_crear_variables(clientes):
    resultado = crear_variables(
        clientes,
        meses_cliente_nuevo=6,
        servicios=["Phone Service", "Online Security", "Premium Tech Support"],
        servicios_proteccion=["Online Security", "Premium Tech Support"],
    )

    assert resultado["cliente_nuevo"].tolist() == [1, 1, 0]
    assert resultado["num_servicios"].tolist() == [1, 3, 1]
    assert resultado["servicios_proteccion"].tolist() == [0, 2, 1]


def test_codificar_si_no_convierte_solo_columnas_si_no(clientes):
    resultado = codificar_si_no(clientes)

    assert resultado["Phone Service"].tolist() == [1, 1, 0]
    assert resultado["Online Security"].tolist() == [0, 1, 1]
    assert resultado["Contract"].equals(clientes["Contract"])
    assert resultado["Customer ID"].equals(clientes["Customer ID"])
