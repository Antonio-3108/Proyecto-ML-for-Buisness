"""
This is a boilerplate pipeline 'business_impact'
generated using Kedro 1.7.0
"""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline


def calcular_impacto_economico(
    y_real: pd.Series,
    probabilidad: pd.Series,
    supuestos: dict[str, Any],
    nombre: str,
) -> dict[str, float]:

    # Ordenar clientes por probabilidad de churn (mayor riesgo primero)
    ranking = probabilidad.sort_values(ascending=False)

    # Calcular la proporción de clientes que contactaríamos en la población total
    # (asumiendo que el conjunto de prueba es representativo)
    proporcion_a_contactar = supuestos["clientes_a_contactar"] / 7043  # 7043 = total de clientes

    # Aplicar esa proporción al conjunto de prueba
    n_clientes_test = int(len(ranking) * proporcion_a_contactar)
    n_clientes_test = max(n_clientes_test, 1)  # al menos 1 cliente

    # Contactar a los N clientes con mayor riesgo en el conjunto de prueba
    clientes_a_contactar = ranking.head(n_clientes_test)

    # Entre los contactados, ¿cuántos realmente se iban?
    churners_detectados = y_real[clientes_a_contactar.index].sum()

    # Escalar a la población total
    factor_escala = 7043 / len(y_real)
    churners_detectados_total = churners_detectados * factor_escala

    # De esos churners detectados, ¿cuántos se retienen con la oferta?
    clientes_retenidos = churners_detectados_total * supuestos["tasa_exito_retencion"]

    # Ingreso anual retenido
    ingreso_retenido = clientes_retenidos * supuestos["ingreso_anual_promedio"]

    # Costo de la campaña (según el presupuesto original)
    costo_campana = supuestos["clientes_a_contactar"] * supuestos["costo_por_cliente"]

    # Ahorro neto
    ahorro_neto = ingreso_retenido - costo_campana

    return {
        "estrategia": nombre,
        "clientes_contactados": supuestos["clientes_a_contactar"],
        "clientes_detectados": churners_detectados_total,
        "clientes_retenidos": clientes_retenidos,
        "ingreso_retenido": ingreso_retenido,
        "costo_campana": costo_campana,
        "ahorro_neto": ahorro_neto,
    }

def evaluar_impacto_lineas_base(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    regla_negocio: dict[str, str],
    supuestos: dict[str, Any],
) -> list[dict[str, float]]:
    probabilidad_azar = pd.Series(0.5, index=y_test.index)
    impacto_azar = calcular_impacto_economico(
        y_test, probabilidad_azar, supuestos, "Contactar al azar"
    )
    cumple_regla = (
        X_test[regla_negocio["columna"]].eq(regla_negocio["valor"]).astype(int)
    )
    probabilidad_regla = cumple_regla.astype(float)
    impacto_regla = calcular_impacto_economico(
        y_test,
        probabilidad_regla,
        supuestos,
        f"Regla: {regla_negocio['columna']} = {regla_negocio['valor']}",
    )

    return [impacto_azar, impacto_regla]


def evaluar_impacto_modelo(
    modelo: Pipeline,
    busqueda: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    supuestos: dict[str, Any],
) -> dict[str, float]:
    probabilidad = pd.Series(modelo.predict_proba(X_test)[:, 1], index=y_test.index)
    impacto = calcular_impacto_economico(
        y_test, probabilidad, supuestos, busqueda["modelo"]
    )
    return impacto


def comparar_impacto(
    lineas_base: list[dict[str, float]],
    *impactos_modelos: dict[str, float],
) -> pd.DataFrame:
    todos_los_impactos = lineas_base + list(impactos_modelos)
    tabla = pd.DataFrame(todos_los_impactos).set_index("estrategia")
    return tabla.round(2)
