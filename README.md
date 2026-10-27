# Predicción de churn en telecomunicaciones

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Proyecto 1 de *Machine Learning for Business* (2026). Aplicamos un modelo de
Machine Learning para identificar clientes con riesgo de abandonar (churn) una
empresa de telecomunicaciones, siguiendo la metodología CRISP-DM e implementado
como un proyecto Kedro reproducible.

> **Estado:** en desarrollo. Las secciones de resultados y recomendaciones se
> completan conforme avanza el proyecto.

## Problema de negocio

**Pregunta:** ¿qué clientes tienen mayor probabilidad de cancelar su servicio,
para que el equipo de retención los contacte primero?

- **Stakeholder:** Gerencia de Retención de Clientes (CRM).
- **Objetivo analítico:** clasificación binaria (churn sí / no).
- **Métrica de éxito:** recall de la clase churn y PR-AUC, más el ahorro neto
  de la campaña frente a contactar clientes al azar.

Detalle completo en [docs/problema_de_negocio.md](docs/problema_de_negocio.md).

## Dataset

[Telecommunications Industry Customer Churn Dataset](https://www.kaggle.com/datasets/aadityabansalcodes/telecommunications-industry-customer-churn-dataset)
(Kaggle, basado en el dataset *Telco Customer Churn* de IBM). 7,043 clientes de
una telefónica ficticia en California.

Los archivos originales están en `data/01_raw/` y se registran en
`conf/base/catalog.yml`:

| Dataset (catálogo)   | Archivo                                  | Contenido                                        |
| -------------------- | ---------------------------------------- | ------------------------------------------------ |
| `raw_demographics`   | `Telco_customer_churn_demographics.csv`  | Edad, género, estado civil, dependientes         |
| `raw_location`       | `Telco_customer_churn_location.csv`      | Ciudad, código postal, coordenadas               |
| `raw_population`     | `Telco_customer_churn_population.csv`    | Población por código postal                      |
| `raw_services`       | `Telco_customer_churn_services.csv`      | Servicios contratados, contrato, cargos, ingresos |
| `raw_status`         | `Telco_customer_churn_status.csv`        | Satisfacción, estado del cliente, churn          |
| `raw_telco_churn`    | `Telco_customer_churn.csv`               | Versión consolidada (referencia)                 |
| `raw_customer_churn` | `CustomerChurn.csv`                      | Versión reducida (referencia)                    |

Las tablas se unen por `Customer ID`; la población se une por `Zip Code`.

## Estructura del proyecto

```
conf/base/          catalog.yml (datos) y parameters.yml (parámetros)
data/01_raw/        datos originales (versionados en git)
data/02_.. 08_/     salidas intermedias generadas por los pipelines (no versionadas)
notebooks/          exploración (EDA)
src/telco_churn/    código del proyecto: pipelines y nodos
tests/              pruebas automatizadas
```

## Instalación

Requiere Python 3.10 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Ejecución

```bash
kedro run
```

Para explorar los datos desde Jupyter con el catálogo ya cargado:

```bash
kedro jupyter lab
```

## Calidad de código y pruebas

```bash
ruff check .
ruff format --check .
pytest
```

## Autores

- Cynthia Santos
- Antonio Valdez

## Fuentes

- IBM Cognos Analytics, *Telco Customer Churn* (versión 11.1.3+).
- Kaggle: aadityabansalcodes, *Telecommunications Industry Customer Churn Dataset*.
- [Documentación de Kedro](https://docs.kedro.org).


---

| estrategia | clientes_contactados | clientes_detectados | clientes_retenidos | ingreso_retenido | costo_campana | ahorro_neto |
| --- | --- | --- | --- | --- | --- | --- |
| Contactar al azar | 2000 | 539.85 | 161.95 | 144139.07 | 100000 | 44139.07 |
| Regla: Contract = Month-to-Month | 2000 | 889.75 | 266.92 | 237562.54 | 100000 | 137562.54 |
| Regresión logística | 2000 | 1374.61 | 412.38 | 367020.78 | 100000 | 267020.78 |
| Random Forest | 2000 | 1364.61 | 409.38 | 364351.54 | 100000 | 264351.54 |
| Gradient Boosting | 2000 | 1394.60 | 418.38 | 372359.26 | 100000 | 272359.26 |

Gradient Boosting es el mejor modelo:

Métricas técnicas: Recall 0.850, PR-AUC 0.800, ROC-AUC 0.914
Impacto económico: Ahorro neto de $272,359
Detecta 1,395 churners de los 1,869 totales (~75%)
Comparación de estrategias (contactando 2,000 clientes con $100,000 de presupuesto):

Contactar al azar: $44,139 ahorro neto (detecta 540 churners)
Regla de negocio (Month-to-Month): $137,563 ahorro neto (detecta 890 churners)
Gradient Boosting: $272,359 ahorro neto (detecta 1,395 churners)
El modelo mejora el ahorro neto en ~6x comparado con contactar al azar y ~2x comparado con la regla de negocio.