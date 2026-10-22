"""Pipeline de entrenamiento: división train/test y entrenamiento de modelos."""

from kedro.pipeline import Node, Pipeline

from .nodes import dividir_datos, entrenar_regresion_logistica


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de entrenamiento.

    Returns:
        Pipeline que divide los datos y entrena los modelos.
    """
    return Pipeline(
        [
            Node(
                func=dividir_datos,
                inputs=["features_clientes", "params:model_training.division"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="dividir_datos",
            ),
            Node(
                func=entrenar_regresion_logistica,
                inputs=[
                    "X_train",
                    "y_train",
                    "params:model_training.columnas_excluidas_lineal",
                    "params:model_training.regresion_logistica",
                ],
                outputs="modelo_regresion_logistica",
                name="entrenar_regresion_logistica",
            ),
        ]
    )
