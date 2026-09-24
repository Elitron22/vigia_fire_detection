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
- Al principio de cada notebook hay unas variables que activan las partes
  costosas (entrenar, evaluar, calcular). **Todas están desactivadas por
  defecto**, así que ejecutar el notebook completo no entrena ni recalcula
  nada: solo muestra los resultados ya guardados en `artifacts/`.
- El conjunto de **test** no se usa hasta el notebook 11. Todas las decisiones
  (modelo, resolución y umbrales) se toman con la partición de validación.
- Los números siguen el orden en que se hizo el trabajo. No hay notebook 04: era
  un análisis de escenas parecidas que se descartó porque D-Fire no indica de
  qué cámara o vídeo sale cada imagen, y no influyó en el resultado final.

## Orden y contenido

| Notebook | Qué hace | Variables para ejecutar | Sección de la memoria |
|---|---|---|---|
| 01 | Prepara el dataset | `RUN_PREPARATION` | 3 |
| 02 | Entrena los modelos | `RUN_TRAINING` | 4.1 |
| 03 | Evalúa cada modelo en validación | `RUN_STANDARD_EVALUATION`, `RUN_ERROR_ANALYSIS` | 4.1 |
| 05 | Barrido de umbrales de confianza | `RUN_SWEEP` | 4.2 |
| 06 | Comparación de resoluciones | `RUN_COMPARISON` | 4.2 |
| 07 | (Opcional) Prueba manual con una imagen o vídeo | — | — |
| 08 | Ajuste de hiperparámetros de YOLO26s | `RUN_TRAINING`, `RUN_EVALUATION` | 4.4 |
| 09 | Selección final en validación | `RUN_SELECTION` | 4.3 y 4.5 |
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
repositorio. Solo hace falta activar `RUN_PREPARATION=True` para volver a
crearla.

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
con `MODEL_KEY` o `EXPERIMENT_ID`. Para evaluar varios experimentos de una vez:
`tools/run_validation_evaluations.py`.

### 05 · Barrido de umbrales

Prueba 21 umbrales de confianza en los modelos finalistas y mide, para cada
uno, el recall y la proporción de imágenes sin humo ni fuego con alarma.
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
anotado. No interviene en ninguna decisión del TFM. Para usarlo con el modelo
final incluido en el repositorio, cambiar en la celda de parámetros:

```python
WEIGHTS_PATH = "artifacts/14_final_model_freeze/final/weights/best.pt"
IMGSZ = 768
INPUT_PATH = "ruta/a/mi_imagen.jpg"
```

Los resultados se guardan en `artifacts/07_manual_inference/`.

### 08 · Ajuste de hiperparámetros de YOLO26s

Prueba cuatro cambios de hiperparámetros (tasa de aprendizaje, weight decay,
aumento de datos más suave y la combinación de ambos) entrenando 50 épocas cada
uno, y los compara con las primeras 50 épocas del entrenamiento de referencia.

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
configuración en `configs/`.

### 10 · Comprobación con todas las configuraciones

Aplica el mismo criterio del 1 % a las 14 combinaciones de modelo y resolución
entrenadas, y muestra aparte los cuatro ensayos de hiperparámetros. Sirve para
comprobar que la elección no dependía de haber comparado solo dos candidatos.

```bash
python tools/run_all_models_01pct_comparison.py
python tools/verify_all_models_01pct_comparison.py
```

Tras este paso, el modelo y sus umbrales se congelaron con
`tools/freeze_final_model.py`, que genera `artifacts/14_final_model_freeze/`
(incluida en el repositorio).

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

Los notebooks se generaron con los scripts `tools/build_*.py`. Reconstruir un
notebook con ellos borra sus resultados guardados, así que después hay que
ejecutarlo de nuevo. La guía de lectura del principio de cada notebook se añade
o actualiza con:

```bash
python tools/document_delivery_notebooks.py
```

Este comando solo modifica esas celdas de texto; no toca el código ni los
resultados.
