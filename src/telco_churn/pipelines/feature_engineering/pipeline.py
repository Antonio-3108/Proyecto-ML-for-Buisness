"""Pipeline de *feature engineering*: variables nuevas y codificación Sí/No."""

from kedro.pipeline import Node, Pipeline

from .nodes import codificar_si_no, crear_variables


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de *feature engineering*.

    Returns:
        Pipeline que genera la tabla de variables a partir de la tabla limpia.
    """
    return Pipeline(
        [
            Node(
                func=crear_variables,
                inputs=[
                    "clientes_limpios",
                    "params:feature_engineering.meses_cliente_nuevo",
                    "params:feature_engineering.servicios",
                    "params:feature_engineering.servicios_proteccion",
                ],
                outputs="clientes_con_variables",
                name="crear_variables",
            ),
            Node(
                func=codificar_si_no,
                inputs="clientes_con_variables",
                outputs="features_clientes",
                name="codificar_si_no",
            ),
        ]
    )
