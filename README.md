# Predicción de churn en telecomunicaciones

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Proyecto 1 de *Machine Learning for Business* (2026). Aplicamos un modelo de
Machine Learning para identificar clientes con riesgo de abandonar (churn) una
empresa de telecomunicaciones, siguiendo la metodología CRISP-DM e implementado
como un proyecto Kedro reproducible.

> **Estado:** en desarrollo. Las secciones de problema de negocio, resultados y
> recomendaciones se completan conforme avanza el proyecto.

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

- Michelle Santos
- Antonio

## Fuentes

- IBM Cognos Analytics, *Telco Customer Churn* (versión 11.1.3+).
- Kaggle: aadityabansalcodes, *Telecommunications Industry Customer Churn Dataset*.
- [Documentación de Kedro](https://docs.kedro.org).
