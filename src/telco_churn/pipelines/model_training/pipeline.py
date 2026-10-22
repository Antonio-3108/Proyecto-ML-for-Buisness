"""Pipeline de entrenamiento: división train/test y entrenamiento de modelos."""

from kedro.pipeline import Node, Pipeline

from .nodes import dividir_datos, entrenar_modelo

# Modelos candidatos; cada uno se configura en parameters_model_training.yml
MODELOS = ["regresion_logistica", "random_forest", "gradient_boosting"]


def create_pipeline(**kwargs) -> Pipeline:
    """Crea el pipeline de entrenamiento.

    Returns:
        Pipeline que divide los datos y entrena cada modelo candidato con
        GridSearchCV.
    """
    dividir = Node(
        func=dividir_datos,
        inputs=["features_clientes", "params:model_training.division"],
        outputs=["X_train", "X_test", "y_train", "y_test"],
        name="dividir_datos",
    )
    entrenar = [
        Node(
            func=entrenar_modelo,
            inputs=[
                "X_train",
                "y_train",
                f"params:model_training.modelos.{modelo}",
                "params:model_training.validacion_cruzada",
            ],
            outputs=[f"modelo_{modelo}", f"busqueda_{modelo}"],
            name=f"entrenar_{modelo}",
        )
        for modelo in MODELOS
    ]
    return Pipeline([dividir, *entrenar])
