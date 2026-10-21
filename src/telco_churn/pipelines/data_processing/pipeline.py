"""Pipeline de procesamiento de datos: carga y unión de las tablas crudas."""

from kedro.pipeline import Node, Pipeline

from .nodes import preparar_poblacion, unir_tablas


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de procesamiento de datos.

    Returns:
        Pipeline que prepara la población y une las tablas de clientes.
    """
    return Pipeline(
        [
            Node(
                func=preparar_poblacion,
                inputs="raw_population",
                outputs="poblacion_preparada",
                name="preparar_poblacion",
            ),
            Node(
                func=unir_tablas,
                inputs=[
                    "raw_demographics",
                    "raw_location",
                    "raw_services",
                    "raw_status",
                    "poblacion_preparada",
                ],
                outputs="clientes_unidos",
                name="unir_tablas",
            ),
        ]
    )
