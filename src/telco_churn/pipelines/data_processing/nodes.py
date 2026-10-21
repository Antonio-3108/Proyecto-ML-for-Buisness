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


def rellenar_faltantes(
    clientes: pd.DataFrame, valores_faltantes: dict[str, str]
) -> pd.DataFrame:
    """Reemplaza los faltantes que tienen significado de negocio por una categoría.

    Por ejemplo, ``Offer`` vacío significa que el cliente no aceptó ninguna
    oferta. Se usa una categoría propia en lugar de imputar con media o moda,
    para no inventar información.

    Args:
        clientes: Tabla de clientes unida.
        valores_faltantes: Columna -> valor con el que se reemplazan sus vacíos.

    Returns:
        Tabla con esas columnas sin valores faltantes.

    Raises:
        KeyError: Si alguna columna configurada no existe en la tabla.
    """
    no_encontradas = set(valores_faltantes) - set(clientes.columns)
    if no_encontradas:
        raise KeyError(f"Columnas no encontradas: {sorted(no_encontradas)}")
    return clientes.fillna(valores_faltantes)


def eliminar_columnas(
    clientes: pd.DataFrame, columnas_a_eliminar: dict[str, list[str]]
) -> pd.DataFrame:
    """Elimina las columnas que no se usan como predictores.

    Args:
        clientes: Tabla de clientes.
        columnas_a_eliminar: Listas de columnas agrupadas por motivo
            (identificador, constante, redundante, fuga de datos, etc.).

    Returns:
        Tabla sin esas columnas.

    Raises:
        KeyError: Si alguna columna configurada no existe en la tabla.
    """
    columnas = [c for grupo in columnas_a_eliminar.values() for c in grupo]
    return clientes.drop(columns=columnas)


def validar_datos_limpios(
    clientes: pd.DataFrame, llave: str, objetivo: str
) -> pd.DataFrame:
    """Verifica la calidad de la tabla limpia antes de guardarla.

    Detiene el pipeline si los datos cambian y dejan de cumplir lo que se
    observó en la exploración, en lugar de entrenar con datos incorrectos.

    Args:
        clientes: Tabla limpia.
        llave: Columna que identifica a cada cliente.
        objetivo: Columna de la variable objetivo.

    Returns:
        La misma tabla, sin cambios, si pasa todas las validaciones.

    Raises:
        ValueError: Si hay llaves duplicadas, valores faltantes o el objetivo
            no es binario (0/1).
    """
    errores = []
    if clientes[llave].duplicated().any():
        errores.append(f"'{llave}' tiene valores duplicados")
    faltantes = clientes.columns[clientes.isna().any()].tolist()
    if faltantes:
        errores.append(f"columnas con valores faltantes: {faltantes}")
    if not set(clientes[objetivo].unique()) <= {0, 1}:
        errores.append(f"'{objetivo}' no es binaria (0/1)")

    if errores:
        raise ValueError("Datos limpios inválidos: " + "; ".join(errores))
    return clientes
