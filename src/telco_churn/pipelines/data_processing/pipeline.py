"""Pipeline de procesamiento de datos: carga, unión y limpieza de las tablas crudas."""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    eliminar_columnas,
    preparar_poblacion,
    rellenar_faltantes,
    unir_tablas,
    validar_datos_limpios,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de procesamiento de datos.

    Returns:
        Pipeline que une las tablas de clientes y genera la tabla limpia.
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
            Node(
                func=rellenar_faltantes,
                inputs=[
                    "clientes_unidos",
                    "params:data_processing.valores_faltantes",
                ],
                outputs="clientes_sin_faltantes",
                name="rellenar_faltantes",
            ),
            Node(
                func=eliminar_columnas,
                inputs=[
                    "clientes_sin_faltantes",
                    "params:data_processing.columnas_a_eliminar",
                ],
                outputs="clientes_filtrados",
                name="eliminar_columnas",
            ),
            Node(
                func=validar_datos_limpios,
                inputs=[
                    "clientes_filtrados",
                    "params:data_processing.llave",
                    "params:data_processing.objetivo",
                ],
                outputs="clientes_limpios",
                name="validar_datos_limpios",
            ),
        ]
    )
