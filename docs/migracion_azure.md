# Migración a Azure Databricks

Notas para reproducir el proyecto en un espacio de trabajo con metastore propio.

## 1. Crear la estructura

Ejecutar `notebooks/00_setup` con estos parámetros:

| Parámetro | Valor |
|---|---|
| `p_catalogo` | `nyc_taxi` |
| `p_crear_catalogo` | `si` |

Equivalente en SQL, si se prefiere hacerlo a mano:

```sql
CREATE CATALOG IF NOT EXISTS nyc_taxi;
USE CATALOG nyc_taxi;

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;

CREATE VOLUME IF NOT EXISTS bronze.landing;
CREATE VOLUME IF NOT EXISTS gold.artifacts;
```

## 2. Apuntar el código al catálogo nuevo

Una sola línea, en la primera celda de cada notebook o en el clúster:

```python
import os
os.environ["NYC_TAXI_CATALOG"] = "nyc_taxi"
```

Alternativa permanente: cambiar el valor por defecto en `src/nyc_taxi/config.py`.

Para que aplique a todo el clúster, se puede definir como variable de entorno en
la configuración de cómputo: `NYC_TAXI_CATALOG=nyc_taxi`.

## 3. Subir los datos

Por interfaz: Catalog Explorer → volumen `landing` → Upload.

Por CLI, que es más fiable para 191 MB:

```bash
databricks configure --host https://adb-XXXX.azuredatabricks.net
databricks fs cp train.csv dbfs:/Volumes/nyc_taxi/bronze/landing/train.csv
databricks fs cp test.csv  dbfs:/Volumes/nyc_taxi/bronze/landing/test.csv
```

## 4. Diferencias respecto a Free Edition

**Cómputo clásico disponible.** Con un clúster tradicional en lugar de
serverless desaparecen dos restricciones que condicionaron el diseño:

- `cache()` vuelve a funcionar. Las tablas intermedias con prefijo `_` siguen
  siendo válidas y no estorban, pero dejan de ser necesarias.
- El límite de 1 GB en la caché de modelos de Spark Connect no aplica. Se puede
  ampliar la rejilla de hiperparámetros en el notebook 05 y elevar el número de
  ventanas de validación temporal.

**Control de costes.** El crédito de la cuenta gratuita se consume por horas de
máquina virtual, no por consultas. Conviene:

- Configurar terminación automática del clúster a los 10 o 15 minutos.
- Usar el tipo de nodo más pequeño disponible; el volumen de datos no justifica
  más.
- Apagar el clúster manualmente al terminar cada sesión.

**Jobs disponibles sin restricción**, lo que habilita el reto bonus de
orquestación con el encadenamiento completo de tareas.

## 5. Recomendación

Mantener Free Edition como entorno principal del entregable y usar el espacio de
Azure únicamente para las pruebas que allí no son posibles: la orquestación
completa y una búsqueda de hiperparámetros más amplia. Migrar el proyecto entero
a mitad de camino implica volver a subir los datos y reconstruir todas las
tablas, sin ganancia sobre los criterios de evaluación.
