# Flujo reproducible de notebooks del TFM

Los notebooks de esta carpeta sustituyen la ejecución monolítica del baseline.
El material histórico y los análisis descartados no forman parte de la carpeta
de entrega; aquí se conservan únicamente los cuadernos necesarios para explicar
o demostrar el flujo final.

## Cómo leer esta carpeta

- Cada notebook comienza con una **Guía de lectura y ejecución** que explica su
  finalidad, requisitos, entradas, salidas y relación con el resto del flujo.
- Los números forman parte de la trazabilidad histórica; el salto del 03 al 05
  es intencionado. El antiguo notebook 04 de auditoría perceptual de escenas se
  retiró porque no intervino en el modelo ni en las conclusiones finales y el
  dataset no ofrece identificadores fiables de cámara, vídeo o evento.
- El flujo formal está formado por 01, 02, 03, 05, 06, 08, 09, 10, 11 y 12.
  El notebook 07 es una demostración opcional para imagen y vídeo.
- Los modos de entrenamiento o evaluación costosa están desactivados por
  defecto. Los cuadernos de resultados se pueden leer sin volver a entrenar.

## Orden de ejecución

1. `01_DFire_preparacion_dataset.ipynb`
   - Activa `RUN_PREPARATION=True` únicamente al crear una nueva versión.
   - Produce `artifacts/datasets/dfire_seed42_val10_v1`.
2. `02_DFire_entrenamiento_modelos.ipynb`
   - Para un solo modelo, selecciona `MODEL_KEY` y deja `TRAINING_QUEUE=[]`.
   - Para varios, añade un diccionario por experimento a `TRAINING_QUEUE`; se
     ejecutan secuencialmente al activar `RUN_TRAINING=True`.
   - Registra cada ejecución en `artifacts/experiments/<experiment_id>`.
   - Libera la memoria de GPU entre trabajos y conserva los fallos con estado
     `failed`. `CONTINUE_ON_ERROR` decide si la cola prosigue tras un fallo.
3. `03_DFire_evaluacion_modelos.ipynb`
   - Selecciona un experimento completo.
   - Trabaja por defecto en `val`: métricas estándar de Ultralytics y errores a
     confianza fija 0,25. Permite consultar resultados guardados.
   - El selector `test` se reserva para la configuración final congelada.
5. `05_DFire_barrido_umbrales.ipynb`
   - Continúa la comparación del notebook 03 con los tres finalistas.
   - Muestra 21 umbrales, escenarios de alarmas del 1 %, 2 % y 5 %, métricas por
     clase, tamaño de cajas y revisión dirigida de errores.
   - Arranca en consulta (`RUN_SWEEP=False`): muestra la última ejecución
     completa. Puede fijarse `RUN_ID` para presentar una ejecución concreta.
   - Los escenarios no fijan automáticamente el umbral de despliegue.
6. `06_DFire_comparacion_resolucion.ipynb`
   - Compara YOLOv8s entrenado a 640, 768 y 1024 mediante una cuadrícula 3×3 de
     resolución de entrenamiento e inferencia, siempre sobre validación.
   - Mide métricas estándar, punto operativo con ≤2 % de negativas con alarma,
     recall por tamaño, cambios emparejados, desglose de errores y coste de cómputo.
   - Arranca en consulta (`RUN_COMPARISON=False`) y carga artefactos verificados.
7. `07_DFire_pruebas_manual_inferencia.ipynb`
   - Aplica un checkpoint registrado o unos pesos indicados a una imagen o vídeo
     externo, mediante ruta o selector de subida en Jupyter.
   - Los vídeos anotados se guardan como WebM/VP8 y se incrustan en la salida
     para que el reproductor funcione también desde el frontend de Jupyter.
   - Guarda el resultado anotado y su configuración en
     `artifacts/07_manual_inference/`; no entrena ni utiliza `test`.
8. `08_DFire_optimizacion_hiperparametros_YOLO26s.ipynb`
   - Ejecuta un cribado multifidelidad de YOLO26s a 768: compara las primeras 50
     épocas del baseline con cuatro ensayos de 50 épocas sobre `lr0`,
     `weight_decay`, aumentos moderados y la combinación `lr0` + aumentos.
   - La cola nocturna se estima en unas 9,4 h, puede reanudar interrupciones y
     conserva la misma semilla durante el cribado.
   - Una segunda fase entrena desde cero la receta elegida durante 100 épocas.
     Ambas fases arrancan desactivadas y mantienen el test cerrado.
   - `RUN_EVALUATION=True` ejecuta el evaluador automático: selecciona dos
     candidatos, barre umbrales separados y genera tablas, figuras y resumen.
9. `09_DFire_seleccion_final_validacion.ipynb`
   - Cierra la comparación YOLO26s 768→768 frente a YOLOv8s 768→640 sobre las
     1.721 imágenes del split de validación congelado.
   - Presenta métricas globales, por clase y por tamaño, umbrales separados,
     falsos positivos/negativos y evidencia emparejada.
   - Exige como máximo un 1 % de imágenes negativas con alarma y selecciona un
     único checkpoint previo a test. No ejecuta inferencia sobre test.
10. `10_DFire_comparacion_todos_modelos_1pct.ipynb`
   - Repite el protocolo final del 1 % para las 14 configuraciones completas de
     arquitectura y resolución ya estudiadas.
   - Presenta por separado los cuatro ensayos de hiperparámetros de 50 épocas,
     que no compiten directamente con los entrenamientos completos.
   - Incluye rankings de máxima sensibilidad y máximo F1, métricas por clase y
     tamaño, y deja explícito el intercambio entre recall macro y recall de fuego.
   - Reutiliza cachés verificadas, mantiene test bloqueado y reproduce los dos
     puntos operativos del notebook 09.
11. `11_DFire_evaluacion_final_test.ipynb`
   - Presenta la única evaluación final del YOLO26s 768→768 congelado, con
     umbrales humo 0,36 y fuego 0,16.
   - Incluye métricas estándar y operativas, desglose por clase y tamaño,
     matriz de errores, alarmas negativas, ejemplos y comparación con validación.
   - Es de solo consulta: no contiene mecanismos para reevaluar, buscar umbrales
     ni seleccionar otro modelo usando test.
12. `12_DFire_interpretabilidad_modelo.ipynb`
   - Explica el YOLO26s final mediante Eigen-CAM multiescala y sensibilidad por
     oclusión sobre aciertos, falsas alarmas y omisiones de humo y fuego.
   - Incluye una prueba de eliminación frente a regiones aleatorias, mapas
     serializados, limitaciones y lectura responsable de seis casos dirigidos.
   - Usa exclusivamente validación; no consulta test ni modifica el modelo o los
     umbrales congelados.

Flujo principal: **01 → 02 → 03 → 05 → 06 → 09 → 10 → congelado → 11 (test final) → 12 (interpretabilidad sobre val)**.

El notebook 07 es una utilidad cualitativa para demostraciones y pruebas
manuales; no forma parte de la selección formal de modelo ni de umbral.

El notebook 08 es un experimento posterior y dirigido. No sustituye la
comparación controlada de arquitecturas: documenta una búsqueda manual acotada
y debe informar únicamente de las ejecuciones realmente realizadas.

El notebook 09 documenta la comparación final inicial entre dos candidatos. El
notebook 10 amplía el mismo criterio del 1 % al inventario completo antes de
congelar definitivamente el modelo. Ninguno consulta el conjunto de test.

## Mantener la documentación de los cuadernos

Si se reconstruye alguno mediante los generadores de `tools/`, se debe volver a
aplicar la guía homogénea de entrega y validar el inventario:

```bash
python tools/document_delivery_notebooks.py
python tests/verify_pipeline_artifacts.py
```

El primer comando solo añade o actualiza celdas Markdown; conserva el código,
los metadatos y las salidas existentes.

## Barrido inicial y revisión de errores

La configuración está en `configs/threshold_sweep.yaml`. Fija explícitamente los
identificadores de los tres experimentos para no elegir otro entrenamiento por
accidente. Se genera una inferencia a confianza 0,01 por modelo, con las mismas
1.721 imágenes de validación y el protocolo operativo del notebook 03.

```bash
python tools/run_threshold_sweep.py
python tools/run_threshold_sweep.py --offline
```

El primer comando reutiliza cachés válidas o genera las que faltan con GPU; el
segundo exige que todas existan. No se vuelve a entrenar. Cada nuevo barrido
genera una carpeta en `artifacts/05_threshold_sweep/validation/<run_id>`.
`latest.json` apunta únicamente a una ejecución completa. Los intentos
interrumpidos permanecen identificados como `incomplete` y no se presentan.

Las predicciones se guardan en `evaluation/val/threshold_cache` de cada
experimento, separadas de `error_analysis` para no sustituir el diagnóstico a
0,25. La caché verifica pesos, manifiesto, código de inferencia, parámetros y
versiones. El barrido debe reproducir el punto histórico 0,25 por imagen.

Los gráficos se exportan a PNG y SVG; las tablas completas y copias de código
permiten auditar las cifras. Las observaciones visuales son una revisión
dirigida, no un etiquetado exhaustivo ni una corrección de las anotaciones.

Para comprobar el cálculo y reconstruir únicamente el notebook 05:

```bash
python -m unittest discover -s tests -p test_tfm_thresholds.py
python tools/verify_threshold_sweep.py
python tools/build_threshold_notebook.py
```

El generador conserva las celdas y salidas del notebook 03 y añade un enlace al
05. Reconstruir el 05 elimina sus salidas; después debe ejecutarse de arriba a
abajo en modo consulta y guardarse para la presentación.

## Comparación controlada de resolución

La configuración está en `configs/resolution_comparison.yaml`. El análisis usa
las mismas 1.721 imágenes para las nueve combinaciones
`entrenamiento 640/768/1024 × evaluación 640/768/1024`. No consulta test.

```bash
python tools/run_resolution_comparison.py
python tools/verify_resolution_comparison.py
python tools/build_resolution_notebook.py
```

Las predicciones y validaciones estándar se guardan en las cachés verificadas de
cada experimento. Cada comparación crea una carpeta inmutable en
`artifacts/06_resolution_comparison/validation/<run_id>` y exporta tablas, código,
gráficos PNG/SVG y una galería de cambios.

## Modelos

El registro está en `configs/model_registry.yaml`. Inicialmente contiene
`yolov8n`, `yolov8s`, `yolo26n` y `yolo26s`. Para añadir otro modelo compatible con
Ultralytics se crea otra entrada en ese archivo; no hay que duplicar notebooks.

El perfil `controlled` fija el mismo protocolo base para todos los modelos. Los
ajustes propios de una arquitectura deben guardarse como experimentos distintos.

## Seguridad y reproducibilidad

- Los notebooks arrancan en modo no destructivo: no entrenan ni evalúan por
  defecto.
- El dataset original no se modifica.
- El test no se importa en el notebook de entrenamiento.
- Las rutas se reconstruyen para Docker, Windows o Colab.
- Al comenzar el primer entrenamiento, la vista corregida se copia una sola vez
  al volumen Docker rápido `/workspace/.cache/tfm-datasets`. Los experimentos
  posteriores reutilizan esa copia y evitan la lectura lenta del volumen Windows.
- El baseline YOLOv8s ya entrenado está registrado como
  `legacy_yolov8s_baseline`, sin copiar sus pesos.
