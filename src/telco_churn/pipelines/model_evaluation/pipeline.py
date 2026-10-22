"""Pipeline de evaluación: líneas base, métricas de los modelos y comparación."""

from kedro.pipeline import Node, Pipeline

from .nodes import comparar_modelos, evaluar_lineas_base, evaluar_modelo


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de evaluación.

    Returns:
        Pipeline que evalúa los modelos en el conjunto de prueba.
    """
    return Pipeline(
        [
            Node(
                func=evaluar_lineas_base,
                inputs=[
                    "X_test",
                    "y_test",
                    "params:model_evaluation.regla_negocio",
                ],
                outputs="metricas_lineas_base",
                name="evaluar_lineas_base",
            ),
            Node(
                func=evaluar_modelo,
                inputs=[
                    "modelo_regresion_logistica",
                    "X_test",
                    "y_test",
                    "params:model_evaluation.umbral",
                ],
                outputs="metricas_regresion_logistica",
                name="evaluar_regresion_logistica",
            ),
            Node(
                func=comparar_modelos,
                inputs=["metricas_lineas_base", "metricas_regresion_logistica"],
                outputs="comparacion_modelos",
                name="comparar_modelos",
            ),
        ]
    )
