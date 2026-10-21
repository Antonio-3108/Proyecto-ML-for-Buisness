"""Prueba de integración: el pipeline completo corre de principio a fin."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

CLIENTES_EN_DATOS_CRUDOS = 7043


def test_kedro_run_genera_tabla_de_clientes():
    raiz = Path.cwd()
    bootstrap_project(raiz)

    with KedroSession.create(project_path=raiz) as session:
        session.run()
        catalog = session.load_context().catalog
        clientes = catalog.load("clientes_unidos")

    assert len(clientes) == CLIENTES_EN_DATOS_CRUDOS
    assert clientes["Customer ID"].is_unique
