"""Prueba de integración: el pipeline completo corre de principio a fin."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

CLIENTES_EN_DATOS_CRUDOS = 7043
COLUMNAS_LIMPIAS = 35  # 33 predictores + llave + objetivo


def test_kedro_run_genera_tablas_de_clientes():
    raiz = Path.cwd()
    bootstrap_project(raiz)

    with KedroSession.create(project_path=raiz) as session:
        session.run()
        catalog = session.load_context().catalog
        unidos = catalog.load("clientes_unidos")
        limpios = catalog.load("clientes_limpios")
        fuga_de_datos = catalog.load("params:data_processing")["columnas_a_eliminar"][
            "fuga_de_datos"
        ]

    assert len(unidos) == CLIENTES_EN_DATOS_CRUDOS
    assert unidos["Customer ID"].is_unique

    assert limpios.shape == (CLIENTES_EN_DATOS_CRUDOS, COLUMNAS_LIMPIAS)
    assert limpios.notna().all().all()
    assert not set(fuga_de_datos) & set(limpios.columns)
