"""Nodos del pipeline de procesamiento de datos.

Implementan las decisiones documentadas en ``notebooks/01_exploracion.ipynb``.
"""

import pandas as pd

LLAVE_CLIENTE = "Customer ID"
LLAVE_POBLACION = "Zip Code"


def preparar_poblacion(population: pd.DataFrame) -> pd.DataFrame:
    """Convierte ``Population`` a número y elimina el identificador interno.

    La población viene como texto con comas como separador de miles, en algunos
    casos con el formato de agrupación de la India (``"1,01,215"``). Quitar todas
    las comas da el número correcto en ambos formatos.

    Args:
        population: Tabla cruda de población por código postal.

    Returns:
        Tabla con las columnas ``Zip Code`` y ``Population`` (entero).
    """
    return population.drop(columns=["ID"]).assign(
        Population=lambda df: pd.to_numeric(df["Population"].str.replace(",", ""))
    )


def _quitar_columnas_repetidas(
    base: pd.DataFrame, nueva: pd.DataFrame, llave: str
) -> pd.DataFrame:
    """Quita de ``nueva`` las columnas que ya existen en ``base``, salvo la llave."""
    repetidas = [c for c in nueva.columns if c in base.columns and c != llave]
    return nueva.drop(columns=repetidas)


def unir_tablas(
    demographics: pd.DataFrame,
    location: pd.DataFrame,
    services: pd.DataFrame,
    status: pd.DataFrame,
    population: pd.DataFrame,
) -> pd.DataFrame:
    """Une las tablas de clientes en una sola tabla con una fila por cliente.

    Las tablas de clientes se unen por ``Customer ID`` (uno a uno) y la población
    por ``Zip Code`` (muchos a uno). Las columnas repetidas entre tablas
    (``Count``, ``Quarter``) se conservan una sola vez. ``validate`` hace que
    pandas lance un error si alguna llave está duplicada.

    Args:
        demographics: Datos demográficos de cada cliente.
        location: Ubicación de cada cliente.
        services: Servicios contratados y facturación.
        status: Estado del cliente y variable objetivo.
        population: Población por código postal, ya preparada.

    Returns:
        Tabla unida con una fila por cliente.
    """
    clientes = demographics
    for tabla in (location, services, status):
        clientes = clientes.merge(
            _quitar_columnas_repetidas(clientes, tabla, LLAVE_CLIENTE),
            on=LLAVE_CLIENTE,
            how="inner",
            validate="one_to_one",
        )

    filas_esperadas = len(demographics)
    if len(clientes) != filas_esperadas:
        raise ValueError(
            f"La unión perdió clientes: {len(clientes)} de {filas_esperadas}."
        )

    return clientes.merge(
        population, on=LLAVE_POBLACION, how="left", validate="many_to_one"
    )
