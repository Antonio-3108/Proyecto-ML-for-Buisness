"""Pruebas de los nodos del pipeline ``data_processing``."""

import pandas as pd
import pytest

from telco_churn.pipelines.data_processing.nodes import (
    eliminar_columnas,
    preparar_poblacion,
    rellenar_faltantes,
    unir_tablas,
    validar_datos_limpios,
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


@pytest.fixture
def clientes() -> pd.DataFrame:
    """Tabla de clientes ya unida, con faltantes de negocio."""
    return pd.DataFrame(
        {
            "Customer ID": ["0001-A", "0002-B", "0003-C"],
            "Offer": ["Offer A", None, None],
            "Internet Type": ["DSL", None, "Cable"],
            "Churn Score": [90, 20, 50],
            "Count": [1, 1, 1],
            "Churn Value": [1, 0, 0],
        }
    )


def test_rellenar_faltantes_usa_la_categoria_configurada(clientes):
    resultado = rellenar_faltantes(
        clientes, {"Offer": "None", "Internet Type": "No Internet"}
    )

    assert resultado["Offer"].tolist() == ["Offer A", "None", "None"]
    assert resultado["Internet Type"].tolist() == ["DSL", "No Internet", "Cable"]


def test_rellenar_faltantes_falla_si_la_columna_no_existe(clientes):
    with pytest.raises(KeyError, match="no encontradas"):
        rellenar_faltantes(clientes, {"Columna Inexistente": "x"})


def test_eliminar_columnas_quita_todos_los_grupos(clientes):
    resultado = eliminar_columnas(
        clientes, {"fuga_de_datos": ["Churn Score"], "constante": ["Count"]}
    )

    assert "Churn Score" not in resultado.columns
    assert "Count" not in resultado.columns
    assert "Churn Value" in resultado.columns


def test_validar_datos_limpios_acepta_datos_correctos(clientes):
    limpios = clientes.fillna("None")

    resultado = validar_datos_limpios(limpios, "Customer ID", "Churn Value")

    pd.testing.assert_frame_equal(resultado, limpios)


@pytest.mark.parametrize(
    ("modificar", "mensaje"),
    [
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicados"),
        (lambda df: df, "valores faltantes"),
        (lambda df: df.fillna("None").assign(**{"Churn Value": [1, 0, 2]}), "binaria"),
    ],
    ids=["llave_duplicada", "faltantes", "objetivo_no_binario"],
)
def test_validar_datos_limpios_detecta_errores(clientes, modificar, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_datos_limpios(modificar(clientes), "Customer ID", "Churn Value")
