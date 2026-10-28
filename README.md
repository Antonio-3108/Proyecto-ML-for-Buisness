# Predicción de churn en telecomunicaciones

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Proyecto 1 de *Machine Learning for Business* (2026). Aplicamos un modelo de
Machine Learning para identificar clientes con riesgo de abandonar (churn) una
empresa de telecomunicaciones, siguiendo la metodología CRISP-DM e implementado
como un proyecto Kedro reproducible.

> Las figuras de resultados están en `docs/figuras/` y se generan con
> `notebooks/02_resultados.ipynb` después de `kedro run`.

## Problema de negocio

**Pregunta:** ¿qué clientes tienen mayor probabilidad de cancelar su servicio,
para que el equipo de retención los contacte primero?

- **Stakeholder:** Gerencia de Retención de Clientes (CRM).
- **Objetivo analítico:** clasificación binaria (churn sí / no).
- **Métrica de éxito:** recall de la clase churn y PR-AUC, más el ahorro neto
  de la campaña frente a contactar clientes al azar.

Detalle completo en [docs/problema_de_negocio.md](docs/problema_de_negocio.md).

## Resultados

Modelo final: **Gradient Boosting** (tasa de aprendizaje 0.05, profundidad 3,
200 iteraciones), elegido con GridSearchCV y validación cruzada de 5 particiones.

| Modelo | Recall | Precisión | ROC-AUC | PR-AUC | Ahorro neto (2,000 contactos) |
|---|---|---|---|---|---|
| Línea base: contactar al azar | — | — | — | — | $41,743 |
| Línea base: contrato mes a mes | 0.874 | 0.446 | 0.741 | 0.423 | $138,224 |
| Regresión logística | 0.861 | 0.583 | 0.903 | 0.766 | $267,021 |
| Random Forest | 0.799 | 0.624 | 0.899 | 0.754 | $264,352 |
| **Gradient Boosting** | **0.850** | **0.612** | **0.914** | **0.800** | **$272,359** |

Contactando a los 2,000 clientes de mayor riesgo se alcanzan ~1,395 de los
1,869 que se van (75 %), con 6.5 veces el ahorro de contactar al azar.
Métricas en `data/08_reporting/comparacion_modelos.csv` y
`data/08_reporting/comparacion_impacto.csv` después de `kedro run`.

## Pipelines

| Pipeline | Entrada → salida |
|---|---|
| `data_processing` | `raw_*` → `clientes_unidos` → `clientes_limpios` |
| `feature_engineering` | `clientes_limpios` → `features_clientes` |
| `model_training` | `features_clientes` → `X_train`, `X_test`, `y_train`, `y_test`, `modelo_*` |
| `model_evaluation` | modelos + prueba → `comparacion_modelos` |
| `business_impact` | modelos + prueba → `comparacion_impacto` |

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
