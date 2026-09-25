# Notebooks del TFM

Estos notebooks recogen todo el proceso experimental del TFM: preparación del
dataset, entrenamiento de los modelos, comparación de resoluciones y umbrales,
selección del modelo final, evaluación en test e interpretabilidad.

Todos están guardados con sus resultados (tablas, gráficas y mensajes), así que
**se pueden leer en GitHub o en Jupyter sin ejecutar nada**.

## Antes de ejecutarlos

Por su tamaño, el repositorio no incluye los resultados intermedios de cada
fase: los modelos entrenados de cada experimento (`artifacts/experiments/`),
las predicciones guardadas ni las carpetas de resultados de cada análisis. Solo
incluye:

- La partición y las correcciones del dataset preparado
  (`artifacts/datasets/dfire_seed42_val10_v1/`).
- El modelo final (`artifacts/14_final_model_freeze/final/`).
- Las tablas y figuras finales de test, interpretabilidad y Raspberry Pi (`results/`).

Por eso hay dos formas de usar los notebooks:

- **Leerlos**: basta con abrirlos. Es la forma de revisar cómo se obtuvo cada
  tabla y figura de la memoria.
- **Ejecutarlos de nuevo**: hay que seguir el orden de la tabla de abajo desde
  el principio, con el dataset D-Fire descargado y una GPU, porque cada notebook
  usa lo que generan los anteriores. Entrenar todos los modelos lleva muchas
  horas. Si se ejecuta un notebook de análisis sin haber generado antes sus
  datos de entrada, fallará al no encontrarlos en `artifacts/`.

### Qué se puede ejecutar con una copia limpia del repositorio

| Situación | Notebooks que funcionan de principio a fin |
|---|---|
| Sin el dataset | 07 (usa el modelo final incluido) |
| Con el dataset en `data/D-Fire` | 01 y 02 con sus valores por defecto (no entrenan), además del 07 |
| El resto (03, 05, 06, 08, 09, 10, 11 y 12) | Necesitan resultados intermedios que no están en el repositorio: fallan al cargarlos hasta que se generan con los notebooks anteriores |

Los identificadores de los experimentos que usa cada análisis están fijados en
`configs/*.yaml`. Al volver a entrenar, el notebook 02 crea identificadores
nuevos (con la fecha y la hora), así que hay que actualizar esos ficheros de
configuración antes de ejecutar los análisis posteriores. Los notebooks 11 y 12
son el cierre del proceso: el 11 no permite repetir la evaluación en test.

## Entorno

La forma más sencilla de ejecutarlos es el entorno Docker de notebooks, descrito
en la sección 4 del `README.md` principal. Desde la raíz del repositorio:

```bash
docker compose -f compose.notebooks.yaml -f compose.gpu.yaml up -d --build
```

y abrir <http://127.0.0.1:8888/lab>. El dataset debe estar en `data/D-Fire`
(sección 3 del `README.md` principal).

Los comandos de `tools/` que aparecen en esta guía se lanzan desde la raíz del
repositorio, en una terminal de Jupyter Lab o con:

```bash
docker compose -f compose.notebooks.yaml exec notebook python tools/<script>.py
```

## Cómo funcionan

- Cada notebook empieza con una **Guía de lectura y ejecución**: para qué
  sirve, qué necesita, qué genera y qué notebook va después.
- En los notebooks que entrenan o calculan algo, al principio hay unas
  variables que activan las partes costosas (entrenar, evaluar, calcular).
  **Todas están desactivadas por defecto**, así que ejecutar el notebook
  completo no entrena ni recalcula nada: solo lee los resultados guardados en
  `artifacts/` (que deben existir; ver la sección anterior). El 10 y el 11 no
  tienen estas variables: sus cálculos se lanzan con los scripts de `tools/`
  indicados en su sección.
- El conjunto de **test** no se usa hasta el notebook 11. Todas las decisiones
  (modelo, resolución y umbrales) se toman con la partición de validación.
- Los números siguen el orden en que se hizo el trabajo. No hay notebook 04: era
  un análisis de escenas parecidas que se descartó porque D-Fire no indica de
  qué cámara o vídeo sale cada imagen, y no influyó en el resultado final.
- Las carpetas de `artifacts/` tienen su propia numeración, que no coincide con
  la de los notebooks (por ejemplo, el 09 escribe en
  `artifacts/12_final_validation_selection/`), así que no hay que confundirlas
  con el notebook 04 que falta.

## Orden y contenido

| Notebook | Qué hace | Variables para ejecutar | Sección de la memoria |
|---|---|---|---|
| 01 | Prepara el dataset | `RUN_PREPARATION` | 3 |
| 02 | Entrena los modelos | `RUN_TRAINING` | 4.1 |
| 03 | Evalúa cada modelo en validación | `RUN_STANDARD_EVALUATION`, `RUN_ERROR_ANALYSIS` | 4.1 |
| 05 | Barrido de umbrales de confianza | `RUN_SWEEP` | — (análisis de apoyo) |
| 06 | Comparación de resoluciones | `RUN_COMPARISON` | 4.2 |
| 07 | (Opcional) Prueba manual con una imagen o vídeo | `INPUT_PATH` | — |
| 08 | Ajuste de hiperparámetros de YOLO26s | `RUN_TRAINING`, `RUN_EVALUATION` | 4.4 |
| 09 | Selección final en validación | `RUN_SELECTION` | 4.5 |
| 10 | Comprobación con todas las configuraciones | — (script) | 4.5 |
| 11 | Evaluación final en test | — (script) | 5 |
| 12 | Interpretabilidad | `RUN_ANALYSIS` | 6 |

### 01 · Preparación del dataset

Lee el dataset original de `data/D-Fire` sin modificarlo, comprueba que todas
las imágenes y anotaciones se pueden leer, corrige cajas que se salen de la
imagen o no tienen tamaño, busca imágenes repetidas entre particiones y aparta
el 10 % del entrenamiento como validación (semilla 42). Resultado: 15.500
imágenes de entrenamiento, 1.721 de validación y 4.306 de test.

Genera `artifacts/datasets/dfire_seed42_val10_v1/`, que ya está incluida en el
repositorio. Incluso con `RUN_PREPARATION=False` necesita el dataset original
en `data/D-Fire`, porque comprueba las imágenes y muestra ejemplos. Para volver a
crear la versión preparada hay que activar `RUN_PREPARATION=True` y, como la
carpeta ya existe, también `ALLOW_REBUILD=True` (o indicar otro
`DATASET_VERSION`).

### 02 · Entrenamiento

Entrena un modelo o una cola de modelos:

- `MODEL_KEY`: modelo a entrenar (`yolov8n`, `yolov8s`, `yolo26n` o `yolo26s`).
- `TRAINING_PROFILE`: configuración de entrenamiento (`controlled` a 640 px,
  `controlled_768` o `controlled_1024`). Viene preparado con `yolo26s` y
  `controlled_768`, que es la configuración del modelo final.
- `TRAINING_QUEUE`: lista de entrenamientos para lanzarlos uno detrás de otro.
- `RUN_TRAINING=True`: lanza el entrenamiento.

Todos los modelos comparten los mismos hiperparámetros: 100 épocas, batch 16,
AdamW con tasa de aprendizaje inicial 0,001 y parada temprana tras 20 épocas
sin mejora. Cada entrenamiento se guarda en `artifacts/experiments/<id>/`.
Los perfiles y modelos están definidos en `configs/model_registry.yaml`.

### 03 · Evaluación en validación

Calcula las métricas estándar (precisión, recall, mAP) de un experimento y
analiza sus errores con un umbral de confianza de 0,25. Se elige el experimento
con `MODEL_KEY` o `EXPERIMENT_ID` (por defecto, el YOLO26s de la comparación
inicial). Al final reúne todos los modelos evaluados y reproduce las Tablas 1 y
2 de la memoria. Para evaluar varios experimentos de una vez:
`tools/run_validation_evaluations.py`.

### 05 · Barrido de umbrales

Prueba 21 umbrales de confianza en tres modelos a 640 px (YOLOv8s, YOLO26s y
YOLO26n) y mide, para cada uno, el recall y la proporción de imágenes sin humo
ni fuego con alarma.
También revisa los errores más habituales. Se ejecuta con `RUN_SWEEP=True` o con:

```bash
python tools/run_threshold_sweep.py
python tools/verify_threshold_sweep.py
```

Configuración: `configs/threshold_sweep.yaml`. Resultados:
`artifacts/05_threshold_sweep/`.

### 06 · Comparación de resoluciones

Compara YOLOv8s entrenado a 640, 768 y 1024 px, evaluando cada uno a las tres
resoluciones (nueve combinaciones), con un máximo del 2 % de imágenes sin humo
ni fuego con alarma. Incluye el coste de cálculo de cada resolución.

```bash
python tools/run_resolution_comparison.py
python tools/verify_resolution_comparison.py
```

Configuración: `configs/resolution_comparison.yaml`. Resultados:
`artifacts/06_resolution_comparison/`.

### 07 · Prueba manual (opcional)

Aplica un modelo a una imagen o vídeo cualquiera y muestra el resultado
anotado. No interviene en ninguna decisión del TFM. Por defecto ya usa el
modelo final incluido en el repositorio (`WEIGHTS_PATH` e `IMGSZ = 768`); solo
hay que indicar el fichero en la celda de parámetros o subirlo con el botón del
notebook:

```python
INPUT_PATH = "ruta/a/mi_imagen.jpg"
```

Los resultados se guardan en `artifacts/07_manual_inference/`.

### 08 · Ajuste de hiperparámetros de YOLO26s

Prueba cuatro cambios de hiperparámetros entrenando 50 épocas cada uno: HP01
(tasa de aprendizaje menor), HP02 (weight decay mayor), HP03 (aumento de datos
más suave) y HP04 (HP01 y HP03 juntos). Los compara con las primeras 50 épocas
del entrenamiento de referencia (Tabla 6 de la memoria).

- `RUN_TRAINING=True`: lanza los cuatro entrenamientos (unas 9,4 horas en
  total; si se interrumpe, se puede reanudar).
- `RUN_EVALUATION=True`: evalúa los cuatro ensayos (o
  `tools/run_hyperparameter_evaluation.py`).

Ninguno mejoró de forma clara al modelo de referencia, que es el que se
mantuvo. Configuración: `configs/yolo26s_hyperparameter_search.yaml`.

### 09 · Selección final en validación

Compara los dos candidatos finales, YOLO26s (entrenado y evaluado a 768 px) y
YOLOv8s (entrenado a 768 px y evaluado a 640 px), con umbrales distintos para
humo y fuego y un máximo del 1 % de imágenes sin humo ni fuego con alarma.
Elige YOLO26s con umbrales 0,36 para humo y 0,16 para fuego. Usa también la
comparación de arquitecturas a 768 px y la revisión de falsos positivos por
clase:

```bash
python tools/run_architecture_768_comparison.py
python tools/run_class_threshold_and_fp_review.py
python tools/run_final_validation_selection.py
```

Cada script tiene su `verify_*.py` correspondiente en `tools/` y su
configuración en `configs/`. El primero genera la comparación a 768 px de la
sección 4.3 de la memoria (Tabla 5 y Figura 4); sus resultados no se incluyen
en el repositorio.

Tras esta selección, el modelo y sus umbrales se congelaron con
`tools/freeze_final_model.py`, que genera `artifacts/14_final_model_freeze/`
(incluida en el repositorio).

### 10 · Comprobación con todas las configuraciones

Aplica el mismo criterio del 1 % a las 14 combinaciones de modelo y resolución
entrenadas, y muestra aparte los cuatro ensayos de hiperparámetros. Sirve para
comprobar que la elección no dependía de haber comparado solo dos candidatos.

```bash
python tools/run_all_models_01pct_comparison.py
python tools/verify_all_models_01pct_comparison.py
```

### 11 · Evaluación final en test

Muestra la única evaluación del modelo congelado sobre las 4.306 imágenes de
test. Es de solo lectura: no permite cambiar el modelo ni buscar otros
umbrales. La evaluación se hizo una vez con:

```bash
python tools/run_final_test_evaluation.py
python tools/verify_final_test_evaluation.py
```

Configuración: `configs/final_test_evaluation.yaml`. Las tablas y figuras
finales están copiadas en `results/final_test/`.

### 12 · Interpretabilidad

Analiza seis casos de validación (un acierto, una falsa alarma y un fallo de
detección por clase) con Eigen-CAM, oclusión y una prueba de eliminación de
regiones. Se ejecuta con `RUN_ANALYSIS=True` o con
`tools/run_model_interpretability.py`. Las figuras y tablas están en
`results/interpretability/`.

## Mantener la documentación de los notebooks

La primera versión de los notebooks se generó con los scripts
`tools/build_*.py`, pero después su texto se revisó a mano. Por eso **no hay que
volver a ejecutar esos scripts**: sobrescribirían los notebooks entregados, con
sus explicaciones y sus resultados guardados. Se conservan solo como referencia.
Tampoco se usan en la memoria los scripts de evaluación con vídeos
(`tools/*operational_video*`).

La guía de lectura del principio de cada notebook se añade o actualiza con:

```bash
python tools/document_delivery_notebooks.py
```

Este comando solo modifica esas celdas de texto; no toca el código ni los
resultados.
