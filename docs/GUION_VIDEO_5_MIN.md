# Guion del vídeo de presentación (máximo 5 minutos)

## 0:00-0:35 — Problema y objetivo

- Explicar por qué una cámara puede complementar sensores tradicionales.
- Presentar el objetivo: detectar humo y fuego, confirmar persistencia y avisar.
- Aclarar que es un prototipo de apoyo, no un sistema certificado.

## 0:35-1:15 — Datos y metodología

- D-Fire: 17.221 imágenes de train y 4.306 de test.
- Validación estratificada con semilla 42.
- Selección de modelo y umbrales sin consultar test.
- Criterio: maximizar recall bajo un presupuesto del 1 % de negativas con alarma.

## 1:15-2:00 — Modelo final

- YOLO26s entrenado e inferido a 768 px.
- Umbrales: humo 0,36 y fuego 0,16.
- Mostrar brevemente la arquitectura y el flujo de alerta.

## 2:00-3:25 — Demostración

1. Abrir la aplicación.
2. Analizar una imagen y señalar cajas, clase y confianza.
3. Procesar un vídeo y explicar que la notificación no detiene las detecciones.
4. Mostrar la pestaña de cámara y la persistencia temporal.
5. Enseñar un mensaje de Telegram con captura, ocultando token y chat ID.

No esperar a que un vídeo largo termine durante la grabación; utilizar un clip breve preparado previamente.

## 3:25-4:10 — Resultados

- Recall: 79,45 % en humo y 75,61 % en fuego.
- F1 micro: 72,29 %; mAP50-95: 46,38 %.
- 21 de 2.005 negativas con alarma: 1,047 %.
- Mostrar uno o dos errores representativos y explicar su impacto.

## 4:10-4:40 — Raspberry Pi 5

- PyTorch: 0,381 FPS.
- NCNN a 640 px: 5,973 FPS y 167,41 ms.
- Explicar el compromiso entre precisión y velocidad.

## 4:40-5:00 — Conclusión

- Resumir la aportación de extremo a extremo.
- Destacar reproducibilidad, alertas no duplicadas y despliegue real.
- Cerrar con el siguiente paso: piloto multi-cámara en el entorno objetivo.

## Ajustes de exportación sugeridos

- Formato: MP4 con H.264 y audio AAC.
- Resolución: 1920x1080 o 1280x720.
- Frecuencia: 25 o 30 FPS.
- Para quedar por debajo de 50 MB en 5 minutos, utilizar un bitrate total aproximado de 1,2 Mbit/s; comprobar el tamaño final y reducir a 720p si fuese necesario.
- La voz en off es obligatoria según la guía; no es necesario aparecer en cámara.
