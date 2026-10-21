"""Pruebas de los nodos del pipeline ``data_processing``."""

import pandas as pd
import pytest

from telco_churn.pipelines.data_processing.nodes import (
    preparar_poblacion,
    unir_tablas,
)


@pytest.fixture
def tablas_crudas() -> dict[str, pd.DataFrame]:
    """Versión mínima de las tablas crudas con dos clientes."""
    ids = ["0001-A", "0002-B"]
    return {
        "demographics": pd.DataFrame(
            {"Customer ID": ids, "Count": [1, 1], "Age": [30, 70]}
        ),
        "location": pd.DataFrame(
            {"Customer ID": ids, "Count": [1, 1], "Zip Code": [90001, 90002]}
        ),
        "services": pd.DataFrame(
            {
                "Customer ID": ids,
                "Count": [1, 1],
                "Quarter": ["Q3", "Q3"],
                "Tenure in Months": [1, 40],
            }
        ),
        "status": pd.DataFrame(
            {
                "Customer ID": ids,
                "Count": [1, 1],
                "Quarter": ["Q3", "Q3"],
                "Churn Value": [1, 0],
            }
        ),
        "population": pd.DataFrame(
            {"Zip Code": [90001, 90002], "Population": [54492, 101215]}
        ),
    }


def test_preparar_poblacion_convierte_formatos_con_comas():
    population = pd.DataFrame(
        {
            "ID": [1, 2, 3],
            "Zip Code": [1, 2, 3],
            "Population": ["11", "54,492", "1,01,215"],
        }
    )

    resultado = preparar_poblacion(population)

    assert list(resultado.columns) == ["Zip Code", "Population"]
    assert resultado["Population"].tolist() == [11, 54492, 101215]


def test_unir_tablas_una_fila_por_cliente_sin_columnas_repetidas(tablas_crudas):
    resultado = unir_tablas(**tablas_crudas)

    assert len(resultado) == len(tablas_crudas["demographics"])
    assert resultado["Customer ID"].is_unique
    assert resultado.columns.is_unique
    assert not any(c.endswith(("_x", "_y")) for c in resultado.columns)
    assert resultado["Population"].tolist() == [54492, 101215]


def test_unir_tablas_falla_si_un_cliente_esta_duplicado(tablas_crudas):
    status = tablas_crudas["status"]
    tablas_crudas["status"] = pd.concat([status, status.iloc[[0]]])

    with pytest.raises(pd.errors.MergeError):
        unir_tablas(**tablas_crudas)


def test_unir_tablas_falla_si_se_pierden_clientes(tablas_crudas):
    tablas_crudas["services"] = tablas_crudas["services"].iloc[[0]]

    with pytest.raises(ValueError, match="perdió clientes"):
        unir_tablas(**tablas_crudas)
