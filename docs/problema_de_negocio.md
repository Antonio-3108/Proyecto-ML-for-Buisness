# Problema de negocio

Propuesta del problema de negocio (fase *Business Understanding* de CRISP-DM).

## Contexto

Una empresa de telecomunicaciones en California atiende a 7,043 clientes con
servicios de telefonía e internet. En el trimestre analizado (Q3),
**1,869 clientes cancelaron su servicio (26.5 %)**. Esos clientes pagaban en
promedio **$74 al mes**, lo que representa cerca de **$1.67 millones al año en
ingresos recurrentes perdidos**.

El principal motivo de cancelación es **la competencia** (841 clientes, ~45 % de
las bajas), seguido de actitud del personal, insatisfacción y precio. El churn
se concentra en clientes con **contrato mes a mes (46 %)**, frente a 11 % en
contratos de un año y 2.5 % en contratos de dos años.

Retener a un cliente cuesta menos que adquirir uno nuevo, pero el presupuesto
de retención es limitado: no es viable ofrecer descuentos o llamar a todos los
clientes. Hoy la empresa no tiene una forma sistemática de saber a quién
contactar primero.

## Stakeholder

**Gerencia de Retención de Clientes (CRM).** Necesita decidir, cada trimestre,
**a qué clientes dirigir la campaña de retención** (llamada, descuento,
migración a contrato anual, mejora de servicio) con un presupuesto limitado.

## Pregunta de negocio

> **¿Qué clientes tienen mayor probabilidad de cancelar su servicio, para que el
> equipo de retención los contacte primero?**

## Objetivo analítico

**Clasificación binaria supervisada.** Predecir, para cada cliente activo, la
probabilidad de churn (`Churn Value` = 1) a partir de su perfil demográfico,
ubicación, servicios contratados, tipo de contrato y facturación.

Con esa probabilidad se ordena a los clientes de mayor a menor riesgo y se
define el grupo a contactar según el presupuesto de la campaña.

**Variables excluidas por fuga de información (*data leakage*).** Las siguientes
columnas solo se conocen después de que el cliente se va o son un resultado de
otro modelo, por lo que no se usarán como predictores: `Customer Status`,
`Churn Label`, `Churn Score`, `Churn Category`, `Churn Reason` y
`Satisfaction Score` (se revisará en la exploración). `CLTV` se usará solo
para estimar el impacto económico, no como predictor.

## Métrica de éxito

**Técnica**

- **Métrica principal: Recall de la clase churn** con un umbral definido por
  negocio. Es más costoso no detectar a un cliente que se va (perder ~$890 al
  año) que contactar a uno que se iba a quedar (costo de la oferta).
- **Métricas de apoyo:** PR-AUC (adecuada con clases desbalanceadas, 26.5 % de
  positivos), ROC-AUC, precisión y F1.
- **Línea base:** el modelo debe superar a (1) un clasificador que siempre
  predice la clase mayoritaria y (2) una regla simple de negocio: "contactar a
  todos los clientes con contrato mes a mes".
- **Meta:** recall ≥ 0.75 manteniendo una precisión ≥ 0.50.

**De negocio**

- **Ahorro neto de la campaña** = ingreso retenido − costo de la campaña,
  comparado contra contactar clientes al azar o con la regla simple.
- **Captura en el top 20 %:** porcentaje de los clientes que se van que quedan
  dentro del 20 % de clientes con mayor riesgo según el modelo.

## Estimación de impacto

Estimación ilustrativa. Los supuestos se configurarán en `parameters.yml` y se
recalcularán con los resultados reales del modelo.

| Supuesto                                         | Valor        |
| ------------------------------------------------ | ------------ |
| Ingreso anual promedio de un cliente que se va   | ~$890        |
| Costo de la acción de retención por cliente      | $50          |
| Tasa de éxito de la retención (cliente contactado que se queda) | 30 % |
| Clientes contactados por campaña                 | 2,000        |

| Escenario                     | Churners detectados | Retenidos | Ingreso retenido | Costo    | Ahorro neto |
| ----------------------------- | ------------------- | --------- | ---------------- | -------- | ----------- |
| Contactar al azar             | ~530 (26.5 %)       | ~159      | ~$142K           | $100K    | **~$42K**   |
| Contactar según el modelo (recall 70 %) | ~1,308    | ~392      | ~$350K           | $100K    | **~$250K**  |

Con el mismo presupuesto, priorizar con el modelo podría multiplicar por ~6 el
ahorro neto de la campaña. Además, el análisis de las variables más
importantes permitirá recomendar acciones concretas (por ejemplo, incentivar
la migración a contratos anuales o revisar la Offer E, que tiene 53 % de churn).
