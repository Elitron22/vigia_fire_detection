# Auditoría final de la aplicación

Fecha: 2026-09-16. Alcance: pantalla móvil de configuración y selección de
fuente. Objetivo: comprobar que el flujo principal sigue siendo comprensible y
operable después de alinear la inferencia con el modelo final y preparar la Pi.

## Pasos capturados

1. **Inicio e imagen — saludable.** El modelo final, 768 px, CPU, umbral base y
   umbrales por clase quedan visibles. La carga de imagen tiene nombre accesible
   y es operable con teclado. Evidencia: `01-inicio-movil.png`.
2. **Vídeo — saludable.** El tab comunica que se conservará un resultado
   anotado sin prometer un códec concreto. La zona de selección es operable con
   teclado. Evidencia: `02-video-movil.png`.
3. **Cámara — saludable con límite explícito.** Se distingue la cámara del
   navegador del servidor que ejecuta inferencia. Evidencia:
   `03-camara-movil.png`.

## Cambios derivados

- Contraste del color de acento: `#e85e36` (3,03:1 sobre crema) sustituido por
  `#b94727` (4,61:1).
- Tabs con `aria-selected` y asociación con sus paneles.
- Dropzones con nombre accesible, foco visible y activación por Enter/Espacio.
- Umbral base representable exactamente como 0,16 (`step=0.01`).
- Copia de vídeo independiente del códec y copia de cámara cliente/servidor.
- Desplazamiento al resultado tras imagen/vídeo, respetando
  `prefers-reduced-motion`.
- Perfil Raspberry Pi con modelo, resolución y parámetros de inferencia
  bloqueados para impedir que la interfaz se aparte accidentalmente del punto
  operativo congelado del TFM.

## Límites

Las capturas no prueban por sí solas compatibilidad completa con lectores de
pantalla ni conformidad WCAG. No se concedió permiso real de cámara durante la
auditoría visual. La inferencia de imagen, vídeo y WebSocket se verificó por una
prueba funcional separada.
