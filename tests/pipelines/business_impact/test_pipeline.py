"""Pruebas de los nodos del pipeline ``business_impact``."""

import pandas as pd
import pytest

from telco_churn.pipelines.business_impact.nodes import (
    calcular_impacto_economico,
    comparar_impacto,
    evaluar_impacto_lineas_base,
)

SUPUESTOS = {
    "ingreso_anual_promedio": 100,
    "costo_por_cliente": 10,
    "tasa_exito_retencion": 0.5,
    "clientes_a_contactar": 5,
    "total_clientes": 10,
}
REGLA = {"columna": "Contract", "valor": "Month-to-Month"}


@pytest.fixture
def datos_prueba() -> tuple[pd.DataFrame, pd.Series]:
    """Diez clientes: 4 se van; 3 de ellos tienen contrato mes a mes."""
    contrato = ["Month-to-Month"] * 5 + ["Two Year"] * 5
    churn = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    indice = pd.Index([f"C{i}" for i in range(10)], name="Customer ID")
    return (
        pd.DataFrame({"Contract": contrato}, index=indice),
        pd.Series(churn, index=indice, name="Churn Value"),
    )


def test_calcular_impacto_contacta_a_los_de_mayor_riesgo(datos_prueba):
    _, y = datos_prueba
    # El modelo pone primero a los 4 que se van
    probabilidad = pd.Series([0.9, 0.8, 0.7, 0.1, 0.2, 0.6, 0.3, 0.1, 0.1, 0.1])
    probabilidad.index = y.index

    resultado = calcular_impacto_economico(y, probabilidad, SUPUESTOS, "Modelo")

    # Contacta a 5 clientes: los 4 que se van + 1 que se queda
    assert resultado["clientes_detectados"] == pytest.approx(4)
    assert resultado["clientes_retenidos"] == pytest.approx(2)
    assert resultado["ingreso_retenido"] == pytest.approx(200)
    assert resultado["costo_campana"] == pytest.approx(50)
    assert resultado["ahorro_neto"] == pytest.approx(150)


def test_lineas_base_usan_el_valor_esperado(datos_prueba):
    X, y = datos_prueba

    azar, regla = evaluar_impacto_lineas_base(X, y, REGLA, SUPUESTOS)

    # Azar: 5 contactados x 40 % de churn = 2 detectados
    assert azar["clientes_detectados"] == pytest.approx(2)
    # Regla: 5 contactados x 60 % de churn entre los de mes a mes = 3
    assert regla["clientes_detectados"] == pytest.approx(3)


def test_lineas_base_no_dependen_del_orden_de_las_filas(datos_prueba):
    X, y = datos_prueba
    orden = y.sample(frac=1, random_state=0).index

    original = evaluar_impacto_lineas_base(X, y, REGLA, SUPUESTOS)
    reordenado = evaluar_impacto_lineas_base(
        X.loc[orden], y.loc[orden], REGLA, SUPUESTOS
    )

    assert original == reordenado


def test_comparar_impacto_une_lineas_base_y_modelos(datos_prueba):
    X, y = datos_prueba
    lineas_base = evaluar_impacto_lineas_base(X, y, REGLA, SUPUESTOS)
    modelo = calcular_impacto_economico(y, y.astype(float), SUPUESTOS, "Modelo")

    resultado = comparar_impacto(lineas_base, modelo)

    assert resultado.index.tolist() == [
        "Contactar al azar",
        "Regla: Contract = Month-to-Month",
        "Modelo",
    ]
    assert "ahorro_neto" in resultado.columns
