# Evaluación final congelada en test

## tl;dr

Se evaluó una única configuración: **YOLO26s 768→768**, checkpoint congelado y
umbrales humo **0,36** / fuego **0,16**. A estos umbrales obtiene en test
**precisión micro 67.88%**, **recall micro 77.32%**,
**F1 micro 72.29%** y **recall macro 77.53%**.
Activa 21 de 2005 imágenes
negativas (1.05%); por tanto, **no cumple**
el límite previo del 1 %. Este resultado queda cerrado: no se han buscado umbrales
ni comparado modelos en test y no se modificará la configuración a partir de él.

## Contrato previo a test

- Modelo: YOLO26s, entrenamiento e inferencia a 768.
- Experimento: `yolo26s_dfire_seed42_20260912T165304Z`.
- Umbrales: humo 0,36; fuego 0,16.
- IoU de acierto: 0,50; NMS solicitado: 0,70.
- Límite operativo decidido en validación: ≤1 % de negativas con alarma.
- Población test: 4306 imágenes, 2005 negativas,
  2311 cajas de humo y 2878 de fuego.
- Exposición histórica conocida de test: Sí: un baseline YOLOv8s fue consultado en agosto de 2026. El YOLO26s congelado no
  se había evaluado previamente en test, pero no debe describirse el conjunto como totalmente virgen.

## Métricas estándar

La precisión y el recall globales son la media entre clases en el punto de operación
interno de Ultralytics; F1 es su media armónica. El mAP usa IoU 0,50 o la media
0,50:0,95 y es independiente de los umbrales por clase elegidos para despliegue.

| scope | precision | recall | f1 | mAP50 | mAP50_95 | preprocess_ms | inference_ms | loss_ms | postprocess_ms |
|---|---|---|---|---|---|---|---|---|---|
| Global (macro clases) | 78.67% | 72.77% | 75.61% | 79.27% | 46.38% | 0.241371 | 2.270044 | 0.000274 | 0.149381 |
| Humo | 84.16% | 80.48% | 82.28% | 85.67% | 53.65% | 0.241371 | 2.270044 | 0.000274 | 0.149381 |
| Fuego | 73.19% | 65.06% | 68.88% | 72.86% | 39.10% | 0.241371 | 2.270044 | 0.000274 | 0.149381 |

## Punto operativo fijo por clase

| scope | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|
| Humo | 1836 | 339 | 475 | 84.41% | 79.45% | 81.85% |
| Fuego | 2176 | 1559 | 702 | 58.26% | 75.61% | 65.81% |

Global micro: TP=4012, FP=1898, FN=1177.

## Recall por tamaño

Bandas por fracción de área de la imagen: pequeño <1 %, mediano 1–10 % y grande ≥10 %.

| class_name | size_band | gt_boxes | tp | fn | recall |
|---|---|---|---|---|---|
| Fuego | Pequeño | 1860 | 1369 | 491 | 73.60% |
| Fuego | Mediano | 848 | 672 | 176 | 79.25% |
| Fuego | Grande | 170 | 135 | 35 | 79.41% |
| Humo | Pequeño | 495 | 327 | 168 | 66.06% |
| Humo | Mediano | 541 | 375 | 166 | 69.32% |
| Humo | Grande | 1275 | 1134 | 141 | 88.94% |

## Matriz de errores

Las confusiones cruzadas emparejan una predicción de clase incorrecta con una GT
no detectada de la otra clase cuando IoU≥0,50. El fondo recoge FP restantes y FN
restantes. Es una matriz operacional a 0,36/0,16, no la matriz estándar de mAP.

| Real | Pred. humo | Pred. fuego | Pred. fondo |
|---|---|---|---|
| Humo real | 1836 | 11 | 464 |
| Fuego real | 12 | 2176 | 690 |
| Fondo real | 327 | 1548 | 0 |

## Imágenes negativas

| scope | negative_images | negative_images_with_alarm | negative_image_false_alarm_rate |
|---|---|---|---|
| any | 2005 | 21 | 1.05% |
| smoke | 2005 | 14 | 0.70% |
| fire | 2005 | 7 | 0.35% |
| both | 2005 | 0 | 0.00% |

La unidad del límite es la **imagen completamente negativa con al menos una caja**,
no el número de cajas FP ni las alarmas por hora de vídeo.

## Validación frente a test

| metric | validation | test | delta_test_minus_validation |
|---|---|---|---|
| Precisión micro | 68.82% | 67.88% | -0.94% |
| Recall micro | 78.91% | 77.32% | -1.60% |
| F1 micro | 73.52% | 72.29% | -1.23% |
| Recall macro | 78.95% | 77.53% | -1.42% |
| Recall clase peor | 78.53% | 75.61% | -2.92% |
| Alarma negativa | 0.89% | 1.05% | 0.15% |

### Estándar por clase

| scope | precision_val | recall_val | mAP50_val | mAP50_95_val | precision_test | recall_test | f1 | mAP50_test | mAP50_95_test | preprocess_ms | inference_ms | loss_ms | postprocess_ms | precision_delta_test_minus_val | recall_delta_test_minus_val | mAP50_delta_test_minus_val | mAP50_95_delta_test_minus_val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all_macro | 80.43% | 74.77% | 80.14% | 46.68% | 78.67% | 72.77% | 75.61% | 79.27% | 46.38% | 0.241371 | 2.270044 | 0.000274 | 0.149381 | -1.76% | -2.00% | -0.87% | -0.31% |
| smoke | 85.72% | 81.25% | 85.64% | 53.72% | 84.16% | 80.48% | 82.28% | 85.67% | 53.65% | 0.241371 | 2.270044 | 0.000274 | 0.149381 | -1.56% | -0.77% | 0.03% | -0.07% |
| fire | 75.14% | 68.30% | 74.64% | 39.64% | 73.19% | 65.06% | 68.88% | 72.86% | 39.10% | 0.241371 | 2.270044 | 0.000274 | 0.149381 | -1.95% | -3.24% | -1.78% | -0.54% |

### Punto operativo por clase

| class_name | threshold | precision_val | recall_val | f1_val | tp_val | fp_val | fn_val | precision_test | recall_test | f1_test | tp_test | fp_test | fn_test | precision_delta_test_minus_val | recall_delta_test_minus_val | f1_delta_test_minus_val |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke | 0.360000 | 85.81% | 79.38% | 82.47% | 762 | 126 | 198 | 84.41% | 79.45% | 81.85% | 1836 | 339 | 475 | -1.40% | 0.07% | -0.61% |
| fire | 0.160000 | 59.01% | 78.53% | 67.38% | 907 | 630 | 248 | 58.26% | 75.61% | 65.81% | 2176 | 1559 | 702 | -0.75% | -2.92% | -1.58% |

### Recall por tamaño

| class_name | size_band | gt_boxes_val | recall_val | gt_boxes_test | recall_test | recall_delta_test_minus_val |
|---|---|---|---|---|---|---|
| smoke | small | 188 | 68.62% | 495 | 66.06% | -2.56% |
| smoke | medium | 222 | 67.57% | 541 | 69.32% | 1.75% |
| smoke | large | 550 | 87.82% | 1275 | 88.94% | 1.12% |
| fire | small | 760 | 75.92% | 1860 | 73.60% | -2.32% |
| fire | medium | 324 | 81.79% | 848 | 79.25% | -2.54% |
| fire | large | 71 | 91.55% | 170 | 79.41% | -12.14% |

Las diferencias test−validación describen generalización observada; no se usan
como criterio para retocar pesos, resolución o umbrales.

## Ejemplos representativos

Se exportaron 12 ejemplos dirigidos de aciertos y errores en
`representative_examples/`. Son ilustraciones post hoc, no una muestra aleatoria
ni evidencia para recalibrar el sistema. Azul=TP, naranja=FP, verde=GT detectada,
amarillo=GT omitida.

## Cierre metodológico

La evaluación queda registrada como la única evaluación final del YOLO26s congelado.
El siguiente trabajo del proyecto debe centrarse en documentación, exportación y
benchmark de despliegue con datos externos o de validación; no en volver a consultar
test ni en seleccionar otro modelo a partir de estas cifras.
