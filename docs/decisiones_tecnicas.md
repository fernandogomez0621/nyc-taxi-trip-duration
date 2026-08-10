# Nota de decisiones técnicas y supuestos


Documento de respaldo de la prueba técnica. Recoge las decisiones de diseño, los
supuestos asumidos y la evidencia que los sustenta.

---

## 1. Plataforma

La solución se implementó sobre **Databricks Free Edition** con cómputo
serverless, Unity Catalog y Delta Lake.

El cómputo serverless impuso una restricción que condicionó el diseño: **no
admite `cache()`**. El patrón adoptado en su reemplazo es la materialización en
tablas Delta intermedias con prefijo `_`, eliminadas al final del notebook.
Cumple el mismo propósito —evitar recomputar un DataFrame en cada acción— por un
mecanismo distinto.

---

## 2. Arquitectura de capas

| Capa | Tablas | Responsabilidad |
|---|---|---|
| Bronze | `raw_trips`, `raw_scoring` | Espejo fiel del archivo, sin alterar valores |
| Silver | `clean_trips`, `quarantine_trips`, `dq_metrics` | Tipado, validación, aislamiento de lo inválido |
| Gold | `trips_analytics`, `agg_demanda_zona_hora`, `trips_features` | Una tabla por consumidor |

**Gold se dividió en tres tablas y no en una.** El criterio no es estético sino
funcional: cada tabla tiene un consumidor identificado con un contrato distinto.
`trips_analytics` entrega detalle fila a fila para exploración,
`agg_demanda_zona_hora` entrega agregados precalculados para visualización, y
`trips_features` entrega una fila ancha por observación para el modelo.

**No se aplicó modelado dimensional (estrella o copo de nieve).** Un esquema
dimensional resuelve la repetición de atributos compartidos entre muchos hechos,
que no es el problema aquí: cada viaje es único y los únicos candidatos a
dimensión (`vendor_id`, `store_and_fwd_flag`) tienen dos valores cada uno.
Normalizarlos añadiría uniones sin ahorrar nada. Además, el consumidor principal
es un modelo predictivo, que requiere una fila por observación con todas sus
variables al lado; una estrella obligaría a reconstruir esa fila con uniones
antes de cada entrenamiento. El patrón correcto en Gold para aprendizaje
automático es la tabla ancha desnormalizada.

---

## 3. Ingesta (Bronze)

El archivo se lee con **esquema explícito de tipo texto** en lugar de
`inferSchema`. Evita una pasada completa adicional sobre los 191,3 MB solo para
adivinar tipos, y garantiza que la capa no altere ningún valor: una inferencia
automática podría, por ejemplo, convertir un identificador con ceros a la
izquierda en número.

Se anexan cuatro columnas técnicas: marca temporal de ingesta, archivo de origen,
origen lógico y un **hash SHA-256 de la fila** calculado solo sobre columnas de
negocio. El hash detecta duplicados exactos entre cargas y da idempotencia a los
reprocesos.

Se usa el modo `FAILFAST`: por defecto Spark asigna nulos a lo que no puede
parsear y continúa, lo que escondería un archivo mal formado hasta varias etapas
después.

### Verificación de carga

| Conjunto | Filas cargadas | Filas esperadas | Duplicados por hash | IDs repetidos |
|---|---|---|---|---|
| `raw_trips` | 1.458.644 | 1.458.644 | 0 | 0 |
| `raw_scoring` | 625.134 | 625.134 | 0 | 0 |

Los duplicados en cero confirman que la fuente ya venía depurada. La regla de
deduplicación queda operando de forma **preventiva**, no correctiva.

---

## 4. Tipado (Silver)

| Decisión | Justificación |
|---|---|
| Coordenadas a `double`, no `float` | Un `float` de 32 bits ofrece ~7 dígitos significativos; el dato original tiene más. El redondeo desplaza el punto decenas de metros y contamina la distancia calculada, que es la variable más influyente del modelo |
| `id` y `vendor_id` permanecen en texto | Son identificadores, no cantidades. Tratarlos como número invita a operaciones sin sentido |
| `trip_duration` a entero | Son segundos exactos; un decimal sugeriría precisión que la fuente no tiene |
| `store_and_fwd_flag` a 0/1 | Mapeo de dominio. Un valor fuera de `Y`/`N` produce nulo en lugar de asumir cero, para que un dato inesperado se haga visible |

**Verificación:** un cast fallido en Spark no lanza excepción, produce nulos en
silencio. Se comparó el conteo de nulos por columna antes y después de convertir:
**cero nulos nuevos en las once columnas de negocio**.

---

## 5. Reglas de calidad y supuestos de limpieza

Cada regla se materializa como una columna booleana independiente en lugar de
encadenar filtros. Eso permite contar los fallos de cada regla por separado y,
sobre todo, detectar un umbral mal calibrado: si una sola regla rechaza un
volumen alto, el sospechoso es el criterio y no el dato.

| Regla | Umbral | Supuesto |
|---|---|---|
| `v_duracion` | 1 s a 21.600 s (6 h) | Un viaje de 0 segundos no ocurrió; más de 6 horas en el área metropolitana indica taxímetro abierto |
| `v_pasajeros` | 1 a 6 | 0 es un registro sin ocupante; más de 6 excede la capacidad legal de un taxi amarillo |
| `v_coords_pickup` | Caja de NYC | Latitud 40,50–41,00 y longitud −74,30 a −73,70. Cubre los cinco condados más JFK, LaGuardia y Newark |
| `v_coords_dropoff` | Caja de NYC | Ídem |
| `v_completitud` | No nulos | `id`, `pickup_datetime` y `trip_duration` |

### Resultado sobre la base real

| Regla | Filas rechazadas | % del total |
|---|---|---|
| `v_duracion` | 2.061 | 0,141 % |
| `v_coords_dropoff` | 985 | 0,068 % |
| `v_coords_pickup` | 256 | 0,018 % |
| `v_pasajeros` | 65 | 0,004 % |
| `v_completitud` | 0 | 0,000 % |
| **Total inválidas** | **3.118** | **0,214 %** |

`clean_trips`: 1.455.526 filas. `quarantine_trips`: 3.118. Retención del 99,79 %.

**Principio aplicado:** no se borra el dato inválido, se aísla. Borrar oculta el
problema; derivar a cuarentena lo hace contable, auditable y devolvible a la
fuente.

---

## 6. Validación del umbral de duración

El corte de 6 horas es el supuesto más fuerte del pipeline, y se verificó antes
de aceptarlo. Se aislaron los 2.057 registros que fallan la regla de duración
teniendo ambas coordenadas válidas —los candidatos a ser viajes largos
legítimos— y se calculó su velocidad implícita.

| Métrica | Valor |
|---|---|
| Velocidad implícita máxima del conjunto | 3,2 km/h |
| Registros por debajo de 1 km/h | 2.015 (98,0 %) |
| Registros por debajo de 5 km/h | 2.057 (100 %) |
| Duración mediana | 23,8 h |
| Registros por encima de 23 h | 1.841 (89,5 %) |
| Duración máxima | 979,5 h (41 días) |
| Distancia mediana recorrida | 2,39 km |

**Conclusión:** ninguno corresponde a un trayecto real. La velocidad más alta de
todo el conjunto está por debajo de la marcha de un peatón. El 89 % se agrupa
justo por debajo de las 24 horas, patrón característico de un taxímetro que
permanece abierto hasta el cierre de día del sistema, no de una distribución
natural de viajes largos.

La decisión es además **robusta al umbral exacto**: entre las 3 y las 6 horas no
existe ningún registro legítimo, de modo que cualquier corte en ese rango produce
el mismo resultado.

---

## 7. Patrón de errores identificado

Combinaciones de reglas que fallan simultáneamente:

| Reglas que fallan | Filas | Causa probable |
|---|---|---|
| Solo duración | 2.056 | Taxímetro que quedó abierto |
| Solo coordenada de destino | 739 | GPS sin señal al finalizar el trayecto |
| Ambas coordenadas | 242 | GPS inoperante durante todo el viaje |
| Solo pasajeros | 63 | Cero digitado por el conductor |
| Solo coordenada de recogida | 12 | GPS sin señal al iniciar |
| Combinaciones múltiples | 6 | Casos aislados |

**Los errores son independientes entre sí**: solo 6 de 3.118 registros fallan más
de una regla. Eso confirma que las cinco validaciones miden dimensiones distintas
y no se solapan.

**Hallazgo destacable:** la asimetría entre 985 destinos inválidos y 256
recogidas inválidas —una razón de casi cuatro a uno— tiene explicación mecánica.
La recogida siempre ocurre donde el vehículo ya circula con señal; el destino se
registra tras un trayecto durante el cual el GPS pudo perderla, por túneles o por
la altura de las edificaciones.

---

## 8. Prevención de fugas de datos

Esta es la decisión de diseño central del proyecto.

### Evidencia documental

El archivo `test.csv` publicado por la fuente contiene **9 columnas frente a las
11 de `train.csv`**. Las dos ausentes son `dropoff_datetime` y `trip_duration`.
Esa asimetría no es casual: la organización retiró deliberadamente la hora de
llegada porque permite derivar el target mediante una simple resta. Es la
confirmación de qué información **no** está disponible en el instante en que se
debe predecir.

### El caso de la velocidad promedio

La velocidad no aparece en ninguno de los dos archivos, por lo que una
comparación de columnas no la detecta. Sin embargo se calcula como distancia
dividida entre duración, es decir, **deriva del target**. Entregarla al modelo
equivale a entregarle la duración con un paso adicional de aritmética: conocidas
distancia y velocidad, el tiempo se despeja de forma exacta.

Es una variable legítima para describir el fenómeno en el análisis exploratorio,
e inválida como variable de entrada.

### Mecanismos aplicados

1. **Separación física.** La velocidad y `dropoff_datetime` viven en
   `trips_analytics` y no existen en `trips_features`. La arquitectura garantiza
   la distinción en lugar de depender de recordarla al momento de modelar.
2. **Lista blanca, no descarte.** `trips_features` se construye seleccionando las
   columnas declaradas en `config.FEATURES_MODELO`. Con un `drop()` se omite una
   columna por descuido y nadie se entera; con lista blanca, todo lo no declarado
   queda fuera por defecto.
3. **Verificación automática.** El notebook 03 falla si alguna columna prohibida
   alcanza la tabla de features.
4. **Ajuste del agrupamiento solo sobre entrenamiento.** El modelo de zonas se
   entrena exclusivamente con la partición de entrenamiento. Ajustarlo sobre el
   conjunto completo definiría los centroides usando también los viajes del
   período de evaluación, introduciendo información futura en las variables de
   entrada. Es una fuga sutil, porque no involucra al target directamente, y por
   eso mismo fácil de pasar por alto.

### Columnas finales del modelo

Doce variables de entrada: `vendor_id`, `passenger_count`, `store_and_fwd_flag`,
`pickup_hour`, `pickup_dayofweek`, `pickup_month`, `is_weekend`,
`distancia_haversine_km`, `distancia_manhattan_km`, `bearing_deg`,
`pickup_cluster` y `dropoff_cluster`. Más el identificador, el target en sus dos
formas y la marca de partición.

---

## 9. Ingeniería de variables

| Variable | Justificación |
|---|---|
| `distancia_haversine_km` | Distancia en línea recta sobre la superficie terrestre |
| `distancia_manhattan_km` | Suma de los componentes norte-sur y este-oeste. Aproxima mejor el recorrido real porque la ciudad está trazada como retícula y un vehículo no puede cortar en diagonal a través de las manzanas |
| `bearing_deg` | Rumbo del trayecto. La congestión no es simétrica: entrar a Manhattan y salir de ella tienen perfiles distintos aunque la distancia sea idéntica |
| `pickup_cluster` / `dropoff_cluster` | Agrupamiento en 20 zonas. Las coordenadas crudas obligarían a un modelo de árboles a aprender los límites de cada barrio mediante cortes sucesivos |
| `log_trip_duration` | El target está fuertemente sesgado a la derecha. Sin transformar, el error cuadrático queda dominado por los trayectos más largos y el modelo optimiza el caso raro a costa del habitual. Además alinea el entrenamiento con la métrica logarítmica de la competencia |

**Se ajustó un solo modelo de zonas y se aplicó a ambos extremos** del trayecto.
Con dos modelos independientes, la zona *n* significaría lugares distintos según
la columna y el modelo no podría relacionarlas.

**No se emplearon UDFs de Python** en ningún cálculo geográfico. Todo se expresa
con funciones nativas (`radians`, `sin`, `cos`, `atan2`), que se ejecutan dentro
de la JVM sin serializar cada fila hacia un intérprete de Python. Las
definiciones equivalentes en Python puro existen en `features/definitions.py`
únicamente como referencia legible y como base de las pruebas automáticas.

---

## 10. Particionamiento

| Tabla | Partición | Justificación |
|---|---|---|
| `trips_analytics` | `pickup_month` | Seis particiones de ~240 mil filas |
| `trips_features` | `split_flag` | Cada corrida de entrenamiento lee su partición sin recorrer la tabla completa |
| `agg_demanda_zona_hora` | Sin partición | 3.360 filas; particionar sería contraproducente |

**Declaración honesta:** con 191 MB y 1,45 millones de filas, el particionamiento
es más demostrativo que necesario. El criterio real en producción sería un tamaño
de partición objetivo de entre 128 MB y 1 GB. Se documenta explícitamente porque
el error más común al aplicar este patrón es particionar por una columna de alta
cardinalidad —la hora del día, por ejemplo—, lo que generaría 24 carpetas de
archivos diminutos y **degradaría** el rendimiento en lugar de mejorarlo.

Particionar `trips_features` por mes no aportaría nada, porque el modelado nunca
consulta por mes.

---

## 11. Estrategia de validación

**Split temporal, no aleatorio.** El corte es el 17 de junio de 2016, reservando
las últimas dos semanas del período para evaluación.

| Partición | Filas | Proporción |
|---|---|---|
| `train` | 1.349.135 | 92,7 % |
| `test` | 106.391 | 7,3 % |

El uso real del modelo es estimar viajes futuros. Un reparto aleatorio dejaría
trayectos del mismo día, la misma hora y las mismas condiciones de tráfico a
ambos lados de la partición, de modo que la métrica mediría memorización de
condiciones concretas y no capacidad de generalizar.

La proporción de evaluación (7,3 %) es inferior al 15-20 % convencional, y es una
decisión consciente: dos semanas completas capturan el ciclo semanal íntegro, que
es la estacionalidad relevante en este fenómeno. Con 106.391 observaciones las
métricas resultan estadísticamente estables.

**El split se materializa como columna** en lugar de calcularse al vuelo, de modo
que cualquier notebook que lea la tabla obtenga la misma partición y el criterio
quede auditable en el propio dato.

---

## 12. Pruebas automáticas

El proyecto incluye **56 pruebas unitarias** que cubren la lógica pura: geometría
de los cálculos de distancia y rumbo, reglas de validación, tipado, asignación
del split, métricas de evaluación y detección de deriva. No requieren sesión de
Spark y se ejecutan en menos de un segundo.

Se ejecutan desde `notebooks/00_tests` y fallan el notebook ante cualquier error,
lo que permite encadenarlas como primera tarea del job y detener el pipeline
antes de tocar el dato.

Ejemplo de lo que protegen: existe una prueba que fija la convención de día de la
semana. `dayofweek` de Spark usa 1 = domingo, mientras que `isoweekday()` de
Python usa 1 = lunes. Confundirlas desplaza la bandera de fin de semana dos días
sin producir ningún error visible; el modelo simplemente aprendería peor.

---

## 13. Hallazgos del análisis exploratorio

El análisis se apoya en **tamaños de efecto** y no en pruebas de significancia.
Con 1,45 millones de observaciones, contrastes como Kolmogorov-Smirnov o
Shapiro-Wilk rechazan cualquier hipótesis nula de forma prácticamente
automática: detectan diferencias irrelevantes y su valor p mide el tamaño de la
muestra más que el fenómeno. Todos los estadísticos se calcularon en Spark sobre
la población completa; la muestra de 200.000 registros se empleó únicamente para
las figuras.

### Distribución del objetivo

| Estadístico | Escala original | Escala logarítmica |
|---|---|---|
| Asimetría | 2,58 | −0,63 |
| Media | 13,9 min | — |
| Mediana | 11,0 min | — |

La cola derecha pesada justifica entrenar sobre `log1p(trip_duration)`: sin la
transformación, el error cuadrático queda dominado por los trayectos extremos y
el modelo optimizaría el caso raro a costa del habitual.

### Efectos marginales frente a parciales

La distancia explica por sí sola el **59,3 %** de la varianza de la duración en
escala logarítmica. El resto de factores se midió antes y después de descontar
ese efecto dominante, y la diferencia entre ambas medidas es el hallazgo
metodológico central del análisis.

| Factor | Efecto marginal | Efecto parcial |
|---|---|---|
| `pickup_hour` | 0,0109 | **0,0844** |
| `dropoff_cluster` | 0,0964 | 0,0598 |
| `pickup_cluster` | 0,0945 | 0,0289 |
| `pickup_dayofweek` | 0,0047 | 0,0210 |
| `es_aeropuerto` | **0,1256** | 0,0118 |
| `pickup_month` | 0,0024 | 0,0027 |
| `passenger_count` | 0,0012 | 0,0006 |
| `vendor_id` | 0,0001 | 0,0001 |

Dos inversiones destacan:

**La hora del día pasa del último lugar al primero.** Medida de forma marginal
explica el 1 % de la varianza, cifra que invitaría a descartarla. Descontada la
distancia, asciende al 8,4 % y se convierte en el factor más influyente. La causa
es un problema de confusión: los viajes de hora punta tienden a ser cortos
—desplazamientos internos en Manhattan— mientras que los de madrugada incluyen
los trayectos largos a los aeropuertos, de modo que ambos efectos se cancelan
parcialmente y ocultan la relación real.

Un análisis basado en correlación de Pearson habría concluido lo contrario: el
coeficiente entre hora y duración es de apenas 0,04, porque Pearson mide
asociación lineal y la hora es cíclica —las 23 y las 0 son contiguas en el
fenómeno pero opuestas en la escala numérica—. La razón de correlación no impone
forma funcional y por eso detecta el efecto.

**La bandera de aeropuerto recorre el camino inverso.** Su efecto marginal es el
mayor de todos los factores evaluados, pero el parcial resulta reducido: los
viajes de aeropuerto tardan más fundamentalmente porque son más largos, no porque
tengan una dinámica propia. Se incorporó como variable con la expectativa
explícita de que su importancia en el modelo fuera baja, y así resultó.

**La zona de destino pesa aproximadamente el doble que la de origen.** El
resultado es coherente con la dinámica del tráfico: el destino determina si el
trayecto termina internándose en la congestión del centro o saliendo hacia la
periferia, mientras que el punto de partida condiciona solo los primeros minutos.

### Hallazgos operativos

- **La congestión es un fenómeno de jornada completa, no de hora punta.** La
  velocidad mediana mínima se registra entre las 11 y las 15 horas (10,4 km/h) y
  la máxima a las 5 de la mañana (21,2 km/h): un trayecto idéntico puede tardar
  el doble según la hora de salida. La demanda, en cambio, alcanza su pico entre
  las 18 y las 19 horas. Demanda y congestión no coinciden en el tiempo porque
  esta última depende del tráfico total de la ciudad.
- **Régimen diferenciado de aeropuerto.** El 6,6 % de los trayectos toca JFK o
  LaGuardia, con duración mediana de 32 minutos frente a 10,4 del resto y
  distancia mediana de 11,3 km frente a 2,0.
- **Concentración de la demanda.** Tres de las veinte zonas absorben el 30,1 % de
  los viajes, y el 16,6 % de los trayectos comienza y termina en la misma zona.
- **Estructura temporal marcada.** El día de la semana explica el 28 % de la
  varianza del volumen diario. El 23 de enero de 2016 registra 1.643 viajes
  frente a una mediana de 8.074 —una caída del 80 %— coincidiendo con la tormenta
  de nieve que paralizó la ciudad. No es un error de datos y se conserva.

### Multicolinealidad

La correlación entre las dos medidas de distancia supera 0,99. La consecuencia
difiere según la familia de modelos: en una regresión lineal vuelve inestables
los coeficientes, mientras que los modelos de árboles no ven afectada su
capacidad predictiva aunque reparten arbitrariamente la importancia entre ambas.
Se conservaron las dos, dejando constancia de que su importancia individual debe
interpretarse en conjunto.

### Verificación del split

El PSI entre las particiones de entrenamiento y evaluación resulta inferior a
0,005 en todas las variables, incluido el objetivo. El corte temporal no
introdujo un desplazamiento de distribución que contamine la métrica, y el
resultado establece además la línea base del detector de deriva previsto para
producción: sobre datos estables no genera falsos positivos.

---

## 14. Modelado

### Estrategia de cómputo

El pipeline de datos se ejecuta íntegramente en PySpark de forma distribuida. El
**entrenamiento**, en cambio, se realiza en un único nodo con scikit-learn, y la
decisión merece justificación porque va a contracorriente de la expectativa
habitual en la plataforma.

Las trece variables del modelo sobre el conjunto de entrenamiento ocupan **82 MB
en memoria**. A ese volumen, el entrenamiento distribuido paga el coste de
coordinar ejecutores y serializar particiones sin obtener nada a cambio: MLlib se
justifica cuando los datos no caben en un nodo, y forzarlo cuando sí caben es
usar la herramienta por reflejo. El ajuste completo sobre 1,35 millones de filas
toma 25 segundos.

A ello se suma una restricción concreta de la plataforma. El cómputo serverless
ejecuta MLlib a través de Spark Connect, que mantiene los modelos ajustados en
una caché de sesión limitada a 1 GB. Un modelo de boosting sobre este volumen la
desborda, y una búsqueda de hiperparámetros lo hace de forma inevitable. La
restricción confirmó una decisión que el volumen ya recomendaba.

### Progresión de modelos

| Modelo | RMSE (min) | MAE (min) | R² | RMSLE |
|---|---|---|---|---|
| Baseline calendario | 12,44 | 7,81 | −0,066 | 0,783 |
| Baseline distancia × hora | 7,14 | 4,39 | 0,648 | 0,495 |
| Regresión lineal (recortada) | 13,03 | 5,39 | −0,169 | 0,534 |
| GBT por defecto | 5,89 | 3,34 | 0,761 | 0,378 |
| **GBT ajustado** | **5,75** | **3,27** | **0,772** | **0,374** |

**Los baselines se eligieron para ser exigentes.** Medir la mejora frente a una
regresión trivial no dice nada útil; medirla frente a lo que hoy produciría un
analista con una hoja de cálculo sí. El primer baseline —duración mediana según
hora y día— obtiene un R² negativo: predice peor que responder siempre la mediana
global. No es un error de cálculo sino la confirmación operativa de que la
distancia explica el 59 % de la varianza, de modo que una regla que la ignora
está condenada. El segundo baseline incorpora la distancia y alcanza un R² de
0,648, estableciendo el listón real que el modelo debe superar. La mejora final
en RMSLE frente a esa heurística es del **24,4 %**.

**El modelo lineal falla por razones estructurales**, y se conserva en el informe
precisamente por eso. Genera 49 predicciones superiores a seis horas, con un
máximo de 69,4 horas, cuando el viaje real más largo del conjunto de evaluación
dura 5,6. Dos causas concurren: el efecto de la hora sobre la duración es
multiplicativo sobre la distancia y un modelo aditivo no puede representarlo sin
declarar la interacción término a término; y al entrenar en escala logarítmica,
un error moderado se amplifica exponencialmente al invertir la transformación.
Documenta por qué la elección de un modelo de árboles no es preferencia sino
consecuencia de la estructura del fenómeno.

**Recorte de las predicciones al dominio válido.** Toda predicción se restringe
al rango declarado válido en la capa Silver (1 segundo a 6 horas). No es un
ajuste para mejorar la métrica sino la aplicación de la misma regla de negocio en
el otro extremo del pipeline: si un viaje de más de seis horas se consideró
inválido al limpiar, predecir diez horas tampoco es admisible.

### Selección de hiperparámetros

**No se empleó validación cruzada por particiones aleatorias.** El K-fold habitual
mezcla períodos y permitiría entrenar con datos de junio para validar sobre
marzo, que es exactamente el problema que el split temporal existe para evitar;
seleccionar hiperparámetros con ese criterio elegiría la configuración que mejor
memoriza condiciones concretas.

En su lugar se validó hacia adelante sobre tres ventanas sucesivas dentro del
conjunto de entrenamiento, cada una entrenando con todo lo anterior a un corte y
validando sobre el período siguiente. El conjunto de evaluación final no
intervino en ningún momento de la búsqueda. Se evaluaron cuatro configuraciones,
doce ajustes en total.

Configuración seleccionada: profundidad sin límite, 300 iteraciones, tasa de
aprendizaje 0,1. La diferencia en RMSLE entre la mejor y la peor configuración es
de 0,0045, lo que indica que el modelo es poco sensible a estos parámetros en el
rango explorado.

### Diagnóstico de sobreajuste

| Métrica | Entrenamiento | Evaluación | Brecha |
|---|---|---|---|
| RMSE (min) | 5,20 | 5,75 | 10,5 % |
| MAE (min) | 3,03 | 3,27 | 8,1 % |
| RMSLE | 0,353 | 0,374 | 6,0 % |

Se aporta evidencia por dos vías independientes. Primero, la brecha entre ambos
conjuntos es reducida: un modelo sobreajustado mostraría un error notablemente
menor sobre los datos que memorizó. Segundo, el RMSLE se mantiene estable entre
las tres ventanas temporales de validación (0,3625, 0,3627 y 0,3793, desviación
de 0,0079), lo que indica que el desempeño no depende del período elegido.

Conviene añadir que el conjunto de evaluación corresponde a dos semanas
**posteriores** a todo lo visto en entrenamiento. Que el error se sostenga sobre
un período futuro es una evidencia más exigente que la que aportaría una
partición aleatoria.

### Análisis de residuos

La media de los residuos es de −0,36 minutos y la mediana de +0,46, con una
asimetría de −8,18. La discrepancia entre ambas medidas revela dos
comportamientos distintos: el modelo **sobreestima levemente el viaje habitual y
subestima gravemente unos pocos trayectos excepcionalmente lentos**. El sesgo por
tramo de distancia lo confirma, pasando de +0,39 minutos en recorridos cortos a
−1,59 en los de más de veinte kilómetros.

La cuantificación de esa cola es el dato más relevante del diagnóstico:

| | |
|---|---|
| Viajes subestimados en más de 15 minutos | 2.037 (1,91 % del total) |
| Duración real mediana de esos viajes | 47,2 min |
| Duración real mediana del resto | 11,1 min |
| **Contribución de esa cola al error cuadrático total** | **53,0 %** |

Menos del 2 % de las observaciones genera más de la mitad del error. Son viajes
que tardaron mucho más de lo que su distancia, hora y zona permitirían anticipar,
lo que apunta a incidentes —accidentes, cierres de vía, condiciones
meteorológicas— no representados en ninguna variable disponible. Es el mismo
fenómeno que la tormenta del 23 de enero produjo en la serie diaria.

La consecuencia operativa es que **la ventana comunicada al pasajero debe ser
asimétrica**: el riesgo de tardar mucho más de lo previsto supera con creces el
de llegar antes, y un intervalo simétrico subestimaría sistemáticamente el peor
caso.

No se aplicó ninguna prueba formal de normalidad sobre los residuos, por la misma
razón expuesta en la sección 13: a este tamaño muestral el resultado estaría
determinado por el número de observaciones. Se reportan la asimetría y el
histograma, que describen la magnitud de la desviación. Conviene además recordar
que la normalidad de los residuos no es requisito para la calidad predictiva:
interviene únicamente en la construcción de intervalos de confianza paramétricos,
que aquí no se emplean.

### Importancia de variables

Se empleó importancia por permutación y no la interna del modelo. La segunda mide
cuántas veces se usó cada variable para dividir un nodo, lo que favorece a las de
alta cardinalidad; la primera mide la degradación real del error al desordenar
cada variable.

| Variable | Importancia |
|---|---|
| `distancia_haversine_km` | 1,4597 |
| `pickup_hour` | 0,1465 |
| `bearing_deg` | 0,0704 |
| `dropoff_cluster` | 0,0445 |
| `pickup_dayofweek` | 0,0331 |
| `pickup_cluster` | 0,0221 |
| `vendor_id` | 0,0197 |
| `distancia_manhattan_km` | 0,0039 |
| `passenger_count` | 0,0036 |
| `es_aeropuerto` | 0,0024 |
| `is_weekend` | 0,0009 |
| `pickup_month` | 0,0000 |
| `store_and_fwd_flag` | −0,0001 |

**Contraste con el análisis exploratorio.** La correlación de rangos entre el
orden previsto por los efectos parciales y el obtenido por el modelo es de
**0,730**. El modelo aprendió la estructura que el análisis identificó y no un
atajo espurio.

Las dos discrepancias mayores estaban anticipadas:

- **`es_aeropuerto` cae de la quinta posición prevista a la décima.** Confirma
  con más fuerza el razonamiento del análisis: su efecto marginal era el mayor de
  todos, pero el parcial resultó reducido, y el modelo muestra que la distancia
  absorbe por completo ese efecto.
- **`distancia_manhattan_km` aparece con aporte marginal** pese a ser una de las
  variables más informativas. Es un artefacto conocido de la importancia por
  permutación ante variables correlacionadas: al permutar una, la otra compensa y
  el error apenas se degrada. Con una correlación de 0,99, ambas deben
  interpretarse conjuntamente.

En sentido contrario, que una variable sin efecto medido apareciera entre las más
importantes habría obligado a revisar si filtraba información del objetivo. No
ocurrió.

---

## 15. Qué quedó pendiente y por qué

**Datos externos de clima e incidentes de tráfico.** Es la carencia de mayor
impacto y está cuantificada: el 1,91 % de viajes que concentra el 53 % del error
corresponde a trayectos anormalmente lentos que ninguna variable disponible
explica. Incorporar precipitación, temperatura y reportes de incidentes es la
extensión con mayor retorno esperado, y quedó fuera por exceder el alcance del
ejercicio.

**Optimización exhaustiva de hiperparámetros.** Se exploraron cuatro
configuraciones sobre tres ventanas temporales. La diferencia entre la mejor y la
peor fue de 0,0045 en RMSLE, lo que sugiere rendimientos decrecientes en esa
dirección; se priorizó una validación rigurosa sobre una búsqueda amplia, por ser
lo que evalúa el enunciado.

**Modelo diferenciado para trayectos de aeropuerto.** El análisis identificó un
régimen distinto, pero el efecto parcial resultó bajo y la importancia en el
modelo prácticamente nula, de modo que un modelo segmentado no parece
justificado. Se documenta la evaluación, no la implementación.

**Intervalos de predicción.** El análisis de residuos muestra que la ventana
adecuada es asimétrica y crece con la distancia. Una regresión cuantílica
entregaría esos intervalos directamente en lugar de derivarlos del error medio,
y es el siguiente paso natural para llevar la recomendación operativa a
producción.

**Series de tiempo más largas.** Con seis meses de datos no es posible separar
tendencia de estacionalidad anual ni evaluar la estabilidad del modelo entre
años. El diseño del pipeline ya contempla ese crecimiento.

**Comparación con el Parquet oficial de NYC TLC.** Habría permitido contrastar
volumen y rendimiento entre formatos. Se descartó por priorizar la profundidad
del análisis sobre la cobertura de retos opcionales.
