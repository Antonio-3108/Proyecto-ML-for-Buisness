"""
This is a boilerplate test file for pipeline 'business_impact'
generated using Kedro 1.7.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
import pytest

# ruff: noqa: PLR2004
from telco_churn.pipelines.business_impact.nodes import (
    calcular_impacto_economico,
    comparar_impacto,
)


@pytest.fixture
def sample_y_test():
    """Objetivo de prueba simple."""
    return pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], index=range(10))


@pytest.fixture
def sample_probabilidad():
    """Probabilidades de churn simples."""
    return pd.Series(
        [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5], index=range(10)
    )


@pytest.fixture
def sample_supuestos():
    """Supuestos de negocio para pruebas."""
    return {
        "ingreso_anual_promedio": 100,
        "costo_por_cliente": 10,
        "tasa_exito_retencion": 0.5,
        "clientes_a_contactar": 5,
    }


def test_calcular_impacto_economico(
    sample_y_test, sample_probabilidad, sample_supuestos
):
    """Prueba el cálculo de impacto económico."""
    resultado = calcular_impacto_economico(
        sample_y_test, sample_probabilidad, sample_supuestos, "Test"
    )

    assert resultado["estrategia"] == "Test"
    assert "clientes_detectados" in resultado
    assert "clientes_retenidos" in resultado
    assert "ingreso_retenido" in resultado
    assert "costo_campana" in resultado
    assert "ahorro_neto" in resultado

    # Verificar cálculos básicos
    assert resultado["costo_campana"] == 5 * 10
    assert resultado["clientes_detectados"] >= 0


def test_comparar_impacto(sample_supuestos):
    """Prueba la comparación de impactos."""
    impacto1 = calcular_impacto_economico(
        pd.Series([0, 1]), pd.Series([0.5, 0.5]), sample_supuestos, "Estrategia 1"
    )
    impacto2 = calcular_impacto_economico(
        pd.Series([1, 0]), pd.Series([0.5, 0.5]), sample_supuestos, "Estrategia 2"
    )

    lineas_base = [impacto1]
    resultado = comparar_impacto(lineas_base, impacto2)

    assert isinstance(resultado, pd.DataFrame)
    assert len(resultado) == 2
    assert "Estrategia 1" in resultado.index
    assert "Estrategia 2" in resultado.index
    assert "ahorro_neto" in resultado.columns
