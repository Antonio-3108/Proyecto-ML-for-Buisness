"""Nodos del pipeline de *feature engineering*.

Crean variables nuevas a partir de la tabla limpia y codifican las columnas
Sí/No. La codificación *one-hot* y el escalado se hacen en el pipeline de
modelado, ajustados solo con los datos de entrenamiento.
"""

import pandas as pd

VALORES_SI_NO = {"Yes", "No"}


def crear_variables(
    clientes: pd.DataFrame,
    meses_cliente_nuevo: int,
    servicios: list[str],
    servicios_proteccion: list[str],
) -> pd.DataFrame:
    """Agrega variables derivadas que resumen el perfil del cliente.

    - ``cliente_nuevo``: 1 si la antigüedad es de ``meses_cliente_nuevo`` o menos.
    - ``num_servicios``: cantidad de servicios contratados.
    - ``servicios_proteccion``: cantidad de servicios de seguridad y soporte.

    Args:
        clientes: Tabla limpia con columnas Sí/No todavía en texto.
        meses_cliente_nuevo: Antigüedad máxima (meses) para considerar nuevo
            a un cliente.
        servicios: Columnas Sí/No que representan un servicio contratado.
        servicios_proteccion: Subconjunto de servicios de seguridad y soporte.

    Returns:
        Tabla con las tres variables nuevas.
    """
    return clientes.assign(
        cliente_nuevo=(clientes["Tenure in Months"] <= meses_cliente_nuevo).astype(int),
        num_servicios=clientes[servicios].eq("Yes").sum(axis=1),
        servicios_proteccion=clientes[servicios_proteccion].eq("Yes").sum(axis=1),
    )


def codificar_si_no(clientes: pd.DataFrame) -> pd.DataFrame:
    """Convierte a 1/0 todas las columnas cuyos valores son solo ``Yes``/``No``.

    Las columnas se detectan automáticamente, así que una columna Sí/No nueva
    en los datos se codifica sin cambiar el código.

    Args:
        clientes: Tabla de clientes.

    Returns:
        Tabla con las columnas Sí/No como enteros (1 = Yes, 0 = No).
    """
    columnas_si_no = [
        col
        for col in clientes.select_dtypes(exclude="number").columns
        if set(clientes[col].unique()) <= VALORES_SI_NO
    ]
    return clientes.assign(
        **{col: clientes[col].eq("Yes").astype(int) for col in columnas_si_no}
    )
