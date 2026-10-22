"""Prueba de integración: el pipeline completo corre de principio a fin."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

CLIENTES_EN_DATOS_CRUDOS = 7043
COLUMNAS_LIMPIAS = 35  # 33 predictores + llave + objetivo
VARIABLES_NUEVAS = 3  # cliente_nuevo, num_servicios, servicios_proteccion


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

    # El modelo debe superar a las líneas base en la métrica principal (PR-AUC)
    regresion_logistica = comparacion.loc["Regresión logística"]
    lineas_base = comparacion.drop(index="Regresión logística")
    assert (regresion_logistica["pr_auc"] > lineas_base["pr_auc"]).all()
