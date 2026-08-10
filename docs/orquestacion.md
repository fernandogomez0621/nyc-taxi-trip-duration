# Orquestación del pipeline

Reto bonus: transformación parametrizada que simula la operación en producción.

## Grafo de tareas

```
                    ┌──────────────────┐
                    │      tests       │
                    └────────┬─────────┘
                    ┌────────┴─────────┐
                    ▼                  ▼
          ┌──────────────────┐  ┌──────────────────┐
          │  bronze_train    │  │ bronze_scoring   │
          │ p_origen=train   │  │ p_origen=scoring │
          └────────┬─────────┘  └────────┬─────────┘
                   ▼                     │
          ┌──────────────────┐           │
          │      silver      │           │
          └────────┬─────────┘           │
                   ▼                     │
          ┌──────────────────┐           │
          │       gold       │           │
          └────────┬─────────┘           │
                   └──────────┬──────────┘
                              ▼
                    ┌──────────────────┐
                    │ monitoreo_drift  │
                    └──────────────────┘
```

## Decisiones de diseño

**Las dos ingestas corren en paralelo.** El conjunto de entrenamiento y el lote
de scoring son independientes, y declararlo así reduce el tiempo total. Es el
mismo notebook invocado dos veces con distinto parámetro, lo que evita duplicar
lógica y elimina de raíz el riesgo de que ambas ingestas diverjan.

**Las pruebas van primero.** Cuestan menos de un segundo y detienen el flujo si
un cambio rompió la lógica pura, antes de gastar cómputo procesando datos con
código defectuoso.

**El monitoreo cierra el flujo.** Cada lote se evalúa contra la referencia en el
momento de llegar, no en una ejecución aparte que podría olvidarse.

**Sin reintentos automáticos.** Las transformaciones son idempotentes —cada capa
reescribe su tabla completa— pero un fallo suele indicar un problema en el dato
de origen. Reintentar sin revisarlo retrasa el diagnóstico en lugar de
resolverlo.

**Sin notificación de éxito.** Una alerta que llega cada semana sin excepción
deja de leerse, y con ella se pierde la que sí importaba.

## Sobre el parámetro de fecha

`p_fecha_proceso` se declara pero no se usa en esta implementación, porque el
conjunto de datos corresponde a una carga histórica completa de seis meses.

Está declarado porque es el parámetro que gobernaría una ejecución incremental.
Al recibir lotes semanales, cada corrida filtraría por esa fecha y procesaría
solo la partición correspondiente, en lugar de reconstruir el histórico. El
diseño de las capas ya lo admite: Bronze escribiría en modo `append` con la fecha
como partición, y Silver y Gold procesarían únicamente las particiones nuevas.

Dejarlo declarado documenta cómo escalaría el diseño sin exigir reescribir la
orquestación.

## Despliegue

Con la CLI de Databricks:

```bash
databricks bundle deploy --target dev
databricks bundle run nyc_taxi_pipeline
```

Alternativamente puede replicarse desde la interfaz: **Jobs & Pipelines → Create
job**, añadiendo cada tarea como notebook y declarando las dependencias del
grafo anterior.

## Equivalencia con Azure Data Factory

El enunciado menciona ADF como alternativa. La correspondencia es directa:

| Databricks Workflows | Azure Data Factory |
|---|---|
| Task | Activity |
| `depends_on` | Dependencia de actividad (`Success`) |
| `base_parameters` | Parámetros de pipeline |
| `schedule` | Schedule trigger |
| `email_notifications.on_failure` | Ruta de fallo hacia Web/Logic App |
| `health.rules` | Alerta de monitor sobre duración |

La diferencia relevante es que ADF orquesta pero no ejecuta la transformación:
invocaría estos mismos notebooks mediante la actividad *Databricks Notebook*. La
lógica no cambiaría en absoluto.
