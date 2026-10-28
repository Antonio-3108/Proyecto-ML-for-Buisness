"""Nodos del pipeline de impacto de negocio.

Estiman el ahorro neto de una campaña de retención que contacta a un número
fijo de clientes (el presupuesto), según cómo se eligen esos clientes: al azar,
con una regla de negocio o con el ranking de riesgo de cada modelo.
"""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def _resumir_impacto(
    nombre: str, clientes_detectados: float, supuestos: dict[str, Any]
) -> dict[str, Any]:
    """Convierte los clientes que se iban y fueron contactados en dinero.

    Args:
        nombre: Nombre de la estrategia.
        clientes_detectados: Clientes contactados que realmente se iban
            (escalado a toda la base de clientes).
        supuestos: Supuestos de negocio de ``parameters_business_impact.yml``.

    Returns:
        Clientes detectados y retenidos, ingreso retenido, costo y ahorro neto.
    """
    clientes_retenidos = clientes_detectados * supuestos["tasa_exito_retencion"]
    ingreso_retenido = clientes_retenidos * supuestos["ingreso_anual_promedio"]
    costo_campana = supuestos["clientes_a_contactar"] * supuestos["costo_por_cliente"]
    return {
        "estrategia": nombre,
        "clientes_contactados": supuestos["clientes_a_contactar"],
        "clientes_detectados": clientes_detectados,
        "clientes_retenidos": clientes_retenidos,
        "ingreso_retenido": ingreso_retenido,
        "costo_campana": costo_campana,
        "ahorro_neto": ingreso_retenido - costo_campana,
    }


def calcular_impacto_economico(
    y_real: pd.Series,
    probabilidad: pd.Series,
    supuestos: dict[str, Any],
    nombre: str,
) -> dict[str, Any]:
    """Estima el ahorro de contactar a los clientes con mayor riesgo.

    Se contacta en el conjunto de prueba la misma proporción de clientes que
    la campaña contacta en toda la base (2,000 de 7,043 = 28.4 %), elegidos de
    mayor a menor probabilidad. Los clientes que se iban entre los contactados
    se escalan a toda la base.

    Args:
        y_real: Objetivo real del conjunto de prueba (0/1).
        probabilidad: Probabilidad de churn de cada cliente de prueba.
        supuestos: Supuestos de negocio de ``parameters_business_impact.yml``.
        nombre: Nombre de la estrategia.

    Returns:
        Resumen del impacto económico de la estrategia.
    """
    proporcion_a_contactar = (
        supuestos["clientes_a_contactar"] / supuestos["total_clientes"]
    )
    n_contactados = max(int(len(y_real) * proporcion_a_contactar), 1)
    contactados = probabilidad.sort_values(ascending=False).head(n_contactados)

    detectados_en_prueba = y_real[contactados.index].sum()
    factor_escala = supuestos["total_clientes"] / len(y_real)
    return _resumir_impacto(nombre, detectados_en_prueba * factor_escala, supuestos)


def evaluar_impacto_lineas_base(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    regla_negocio: dict[str, str],
    supuestos: dict[str, Any],
) -> list[dict[str, Any]]:
    """Estima el ahorro esperado de las estrategias sin modelo.

    Ninguna de las dos ordena a los clientes, así que se usa el **valor
    esperado** (clientes contactados x tasa de churn del grupo) en lugar de
    tomar los primeros clientes de la tabla, cuyo resultado dependería del
    orden de las filas.

    - **Contactar al azar:** tasa de churn de todos los clientes.
    - **Regla de negocio:** tasa de churn de los clientes que cumplen la regla
      (por defecto, contrato mes a mes).

    Args:
        X_test: Variables de prueba.
        y_test: Objetivo de prueba.
        regla_negocio: ``columna`` y ``valor`` que definen la regla.
        supuestos: Supuestos de negocio de ``parameters_business_impact.yml``.

    Returns:
        Resumen del impacto de cada línea base.
    """
    cumple_regla = X_test[regla_negocio["columna"]].eq(regla_negocio["valor"])
    tasas = {
        "Contactar al azar": y_test.mean(),
        f"Regla: {regla_negocio['columna']} = {regla_negocio['valor']}": y_test[
            cumple_regla
        ].mean(),
    }
    return [
        _resumir_impacto(nombre, supuestos["clientes_a_contactar"] * tasa, supuestos)
        for nombre, tasa in tasas.items()
    ]


def evaluar_impacto_modelo(
    modelo: Pipeline,
    busqueda: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    supuestos: dict[str, Any],
) -> dict[str, Any]:
    """Estima el ahorro de contactar a los clientes que el modelo ve más riesgosos.

    Args:
        modelo: Modelo entrenado con ``predict_proba``.
        busqueda: Resumen de GridSearchCV (se usa el nombre del modelo).
        X_test: Variables de prueba.
        y_test: Objetivo de prueba.
        supuestos: Supuestos de negocio de ``parameters_business_impact.yml``.

    Returns:
        Resumen del impacto económico del modelo.
    """
    probabilidad = pd.Series(modelo.predict_proba(X_test)[:, 1], index=y_test.index)
    return calcular_impacto_economico(
        y_test, probabilidad, supuestos, busqueda["modelo"]
    )


def comparar_impacto(
    lineas_base: list[dict[str, Any]],
    *impactos_modelos: dict[str, Any],
) -> pd.DataFrame:
    """Reúne el impacto de las líneas base y los modelos en una tabla.

    Args:
        lineas_base: Impacto de cada línea base.
        *impactos_modelos: Impacto de cada modelo.

    Returns:
        Tabla con una fila por estrategia, redondeada a 2 decimales.
    """
    tabla = pd.DataFrame([*lineas_base, *impactos_modelos]).set_index("estrategia")
    return tabla.round(2)
