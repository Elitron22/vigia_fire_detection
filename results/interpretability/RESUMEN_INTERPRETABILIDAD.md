# Interpretabilidad del YOLO26s final

## tl;dr

Se explicaron seis casos dirigidos del conjunto de **validación**: acierto,
falsa alarma negativa y omisión limítrofe para humo y fuego. Al ocultar el 25 %
de las regiones con mayor sensibilidad, la caída mediana de confianza fue
96.2%, frente a 24.4% al ocultar regiones aleatorias.
La estrategia dirigida redujo más la confianza en 6/6 casos. El máximo
del mapa de oclusión cayó dentro de la caja objetivo en 3/6 casos y el
de Eigen-CAM en 1/6.

## Método

- Modelo congelado: **YOLO26s 768→768**, hash `bfb54c4726519fb473bb7c4825577e51c05af2b0db5fd46bf1931a8a29285f67`.
- Punto operativo: humo 0,36 y fuego 0,16; no se modifican pesos ni umbrales.
- Eigen-CAM multiescala: primera componente de las activaciones de las capas
  [16, 19, 22], normalizadas, deshecho el letterbox y promediadas.
- Oclusión: rejilla 6×6;
  cada región se sustituye por su versión desenfocada y se mide la caída relativa
  de la misma detección (clase y solapamiento IoU≥0,30).
- Fidelidad: se ocultan progresivamente las regiones ordenadas por importancia y
  se comparan con 5 selecciones aleatorias.

## Diseño de casos

La selección es dirigida y determinista. Cubre tipos de comportamiento, pero no
es una muestra aleatoria ni permite estimar la frecuencia de las explicaciones.
Las omisiones escogidas son limítrofes: existía una predicción coincidente a
confianza baja, pero quedaba por debajo del umbral operativo.

| order | case | class_name | filename | cached_target_confidence | operating_threshold |
|---|---|---|---|---|---|
| 1 | true_positive | smoke | WEB06295.jpg | 0.8933 | 0.3600 |
| 2 | true_positive | fire | WEB03832.jpg | 0.8412 | 0.1600 |
| 3 | negative_false_alarm | smoke | WEB00156.jpg | 0.6455 | 0.3600 |
| 4 | negative_false_alarm | fire | WEB02251.jpg | 0.5172 | 0.1600 |
| 5 | borderline_false_negative | smoke | WEB05832.jpg | 0.3575 | 0.3600 |
| 6 | borderline_false_negative | fire | WEB05014.jpg | 0.1596 | 0.1600 |

## Fidelidad por caso

| order | case | class_name | filename | fresh_target_confidence | max_single_occlusion_drop | top25_score_retained | random25_score_retained | occlusion_mass_in_target | occlusion_pointing_game | eigencam_mass_in_target | eigencam_pointing_game |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | true_positive | smoke | WEB06295.jpg | 0.8934 | 0.9989 | 0.0033 | 0.7192 | 0.1531 | False | 0.1711 | False |
| 2 | true_positive | fire | WEB03832.jpg | 0.8412 | 0.9346 | 0.0719 | 0.7934 | 0.9841 | True | 0.0555 | False |
| 3 | negative_false_alarm | smoke | WEB00156.jpg | 0.6458 | 0.8555 | 0.0000 | 0.4342 | 0.4377 | False | 0.0351 | False |
| 4 | negative_false_alarm | fire | WEB02251.jpg | 0.5179 | 1.0000 | 0.0000 | 0.3384 | 0.7364 | True | 0.0721 | False |
| 5 | borderline_false_negative | smoke | WEB05832.jpg | 0.3575 | 0.9841 | 0.1271 | 1.4600 | 0.9866 | True | 0.8191 | True |
| 6 | borderline_false_negative | fire | WEB05014.jpg | 0.1587 | 0.5469 | 0.3434 | 2.1488 | 0.1675 | False | 0.0127 | False |

## Interpretación responsable

Los mapas muestran asociaciones locales del modelo, no una explicación causal del
incendio ni una garantía de robustez. Eigen-CAM es **no específico de clase**:
indica actividad multiescala, no prueba que una zona cause la etiqueta humo o
fuego. La oclusión sí es específica para una detección, pero depende del tamaño
de rejilla, del desenfoque elegido y puede introducir imágenes fuera de distribución.
La prueba de eliminación es local y solo usa seis casos.

Las falsas alarmas ayudan a formular hipótesis visuales sobre texturas o colores,
pero no demuestran su causa sin un estudio adicional. Las omisiones limítrofes
explican evidencia débil del modelo y no representan los FN persistentes sin
ninguna detección asociable.

## Alcance

No se ha consultado `test`, no se han buscado nuevos umbrales y estos resultados
no cambian la selección final. El análisis satisface el requisito de incluir mapas
de activación y pruebas de oclusión sobre humo, fuego, aciertos, falsas alarmas y
omisiones, con una comprobación cuantitativa de fidelidad.
