"""Prueba de integración: el pipeline completo corre de principio a fin."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# ruff: noqa: PLR2004

CLIENTES_EN_DATOS_CRUDOS = 7043
COLUMNAS_LIMPIAS = 35  # 33 predictores + llave + objetivo
VARIABLES_NUEVAS = 3  # cliente_nuevo, num_servicios, servicios_proteccion
MODELOS_CANDIDATOS = 3


def test_kedro_run_genera_tablas_de_clientes():
    raiz = Path.cwd()
    bootstrap_project(raiz)

    with KedroSession.create(project_path=raiz) as session:
        session.run()
        catalog = session.load_context().catalog
        unidos = catalog.load("clientes_unidos")
        limpios = catalog.load("clientes_limpios")
        features = catalog.load("features_clientes")
        comparacion = catalog.load("comparacion_modelos").set_index("modelo")
        fuga_de_datos = catalog.load("params:data_processing")["columnas_a_eliminar"][
            "fuga_de_datos"
        ]

    assert len(unidos) == CLIENTES_EN_DATOS_CRUDOS
    assert unidos["Customer ID"].is_unique

    assert limpios.shape == (CLIENTES_EN_DATOS_CRUDOS, COLUMNAS_LIMPIAS)
    assert limpios.notna().all().all()
    assert not set(fuga_de_datos) & set(limpios.columns)

    assert features.shape == (
        CLIENTES_EN_DATOS_CRUDOS,
        COLUMNAS_LIMPIAS + VARIABLES_NUEVAS,
    )
    assert features.drop(columns="Customer ID").select_dtypes(
        exclude="number"
    ).columns.tolist() == [
        "Gender",
        "Offer",
        "Internet Type",
        "Contract",
        "Payment Method",
    ]

    # Cada modelo debe superar a las líneas base en la métrica principal (PR-AUC)
    es_linea_base = comparacion.index.str.startswith("Línea base")
    lineas_base = comparacion[es_linea_base]
    modelos = comparacion[~es_linea_base]
    assert len(modelos) == MODELOS_CANDIDATOS
    assert (modelos["pr_auc"] > lineas_base["pr_auc"].max()).all()

    # Verificar que el pipeline de impacto de negocio genera resultados
    comparacion_impacto = catalog.load("comparacion_impacto")
    assert len(comparacion_impacto) == 5  # 2 líneas base + 3 modelos
    assert "ahorro_neto" in comparacion_impacto.columns
