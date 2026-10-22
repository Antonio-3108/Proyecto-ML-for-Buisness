"""Pipeline de evaluación: líneas base, métricas de los modelos y comparación."""

from kedro.pipeline import Node, Pipeline

from telco_churn.pipelines.model_training.pipeline import MODELOS

from .nodes import comparar_modelos, evaluar_lineas_base, evaluar_modelo


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de evaluación.

    Returns:
        Pipeline que evalúa las líneas base y cada modelo en el conjunto de
        prueba, y los reúne en una tabla de comparación.
    """
    lineas_base = Node(
        func=evaluar_lineas_base,
        inputs=["X_test", "y_test", "params:model_evaluation.regla_negocio"],
        outputs="metricas_lineas_base",
        name="evaluar_lineas_base",
    )
    evaluar = [
        Node(
            func=evaluar_modelo,
            inputs=[
                f"modelo_{modelo}",
                f"busqueda_{modelo}",
                "X_test",
                "y_test",
                "params:model_evaluation.umbral",
            ],
            outputs=f"evaluacion_{modelo}",
            name=f"evaluar_{modelo}",
        )
        for modelo in MODELOS
    ]
    comparar = Node(
        func=comparar_modelos,
        inputs=["metricas_lineas_base", *[f"evaluacion_{m}" for m in MODELOS]],
        outputs="comparacion_modelos",
        name="comparar_modelos",
    )
    return Pipeline([lineas_base, *evaluar, comparar])
