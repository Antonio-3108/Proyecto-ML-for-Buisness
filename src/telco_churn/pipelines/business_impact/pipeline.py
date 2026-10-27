"""
This is a boilerplate pipeline 'business_impact'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline

from telco_churn.pipelines.model_training.pipeline import MODELOS

from .nodes import comparar_impacto, evaluar_impacto_lineas_base, evaluar_impacto_modelo


def create_pipeline(**kwargs) -> Pipeline:
    lineas_base = Node(
        func=evaluar_impacto_lineas_base,
        inputs=[
            "X_test",
            "y_test",
            "params:model_evaluation.regla_negocio",
            "params:business_impact.supuestos",
        ],
        outputs="impacto_lineas_base",
        name="evaluar_impacto_lineas_base",
    )

    evaluar = [
        Node(
            func=evaluar_impacto_modelo,
            inputs=[
                f"modelo_{modelo}",
                f"busqueda_{modelo}",
                "X_test",
                "y_test",
                "params:business_impact.supuestos",
            ],
            outputs=f"impacto_{modelo}",
            name=f"evaluar_impacto_{modelo}",
        )
        for modelo in MODELOS
    ]

    comparar = Node(
        func=comparar_impacto,
        inputs=[
            "impacto_lineas_base",
            *[f"impacto_{m}" for m in MODELOS],
        ],
        outputs="comparacion_impacto",
        name="comparar_impacto",
    )

    return Pipeline([lineas_base, *evaluar, comparar])
