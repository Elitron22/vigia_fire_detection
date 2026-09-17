# Vigía: detección de humo y fuego con visión artificial

Trabajo Fin de Máster de Elías Ruiz Fernández. El proyecto desarrolla y valida un sistema de detección temprana de humo y fuego a partir de imágenes, vídeos y cámara en directo. Incluye una aplicación web, alertas persistentes por Telegram y una variante optimizada para Raspberry Pi 5.

## Demostración en un comando

### Requisitos

- Docker Desktop 4.x o Docker Engine con Docker Compose v2.
- Arquitectura x86-64 para la demostración local.
- Al menos 8 GB de RAM y 6 GB libres en disco durante la primera construcción.

Desde esta carpeta:

```powershell
docker compose up --build
```

Abrir <http://127.0.0.1:8000>. La aplicación arranca en CPU con el modelo PyTorch final a 768 px. El modelo NCNN a 640 px también queda disponible en el selector.

Para detenerla:

```powershell
docker compose down
```

No es necesario descargar dependencias manualmente ni disponer de GPU para probar la aplicación. La primera construcción descarga las dependencias y puede tardar varios minutos. El dataset completo no se incluye en esta entrega para mantener el repositorio manejable; sí se conservan los manifiestos, auditorías y artefactos finales necesarios para auditar el flujo.

## Verificación automatizada

Con la imagen ya construida:

```powershell
docker compose run --rm --no-deps -e TFM_APP_MODEL_PATH= app python -m unittest discover -s tests -p "test_fire_app.py" -q
```

La prueba funcional de la entrega se ha validado además con una inferencia real de imagen y ambos modelos visibles en la API.

## Alertas de Telegram (opcional)

La aplicación funciona sin Telegram en modo de simulación. Para activar envíos reales:

1. Copiar `.env.example` como `.env`.
2. Completar el token y el identificador del chat.
3. Establecer `TFM_TELEGRAM_MODE=live`.
4. Reiniciar `docker compose up --build`.

El archivo `.env` está excluido del repositorio. Nunca se deben publicar credenciales.

## Qué puede comprobar el evaluador

- Analizar una imagen y descargar el resultado anotado.
- Procesar un vídeo sin recibir avisos duplicados para el mismo archivo.
- Usar la cámara del navegador con la misma regla temporal que el vídeo.
- Comparar el modelo PyTorch final y la variante NCNN optimizada.
- Consultar el estado del servicio en `/api/health` y la documentación en `/docs`.

## Resultados finales resumidos

El modelo final es YOLO26s entrenado e inferido a 768 px. Los umbrales se fijaron en validación antes de una única evaluación final en test: 0,36 para humo y 0,16 para fuego.

| Métrica en test | Resultado |
|---|---:|
| Precisión micro | 67,88 % |
| Recall micro | 77,32 % |
| F1 micro | 72,29 % |
| Recall de humo | 79,45 % |
| Recall de fuego | 75,61 % |
| mAP50-95 | 46,38 % |
| Imágenes negativas con alarma | 21/2.005 (1,047 %) |

En Raspberry Pi 5, la variante NCNN a 640 px obtuvo 167,41 ms de latencia media y 5,973 FPS en el protocolo de referencia. El modelo PyTorch permanece como referencia académica y opción predeterminada; NCNN es la opción operativa para el dispositivo de borde.

## Organización del repositorio

```text
├── fire_app/                 API, interfaz web y alertas
├── artifacts/14_.../         modelo PyTorch final y manifiesto SHA-256
├── artifacts/datasets/       manifiestos, auditorías y YAML del dataset preparado
├── deployment/               exportaciones y despliegue en Raspberry Pi 5
├── notebooks/                11 cuadernos documentados del flujo experimental
├── configs/                  contratos de experimentos y aplicación
├── tools/                    ejecutores y verificadores reproducibles
├── tests/                    pruebas unitarias y funcionales
├── results/                  tablas, figuras y resúmenes seleccionados
├── Dockerfile                imagen ligera de demostración
└── compose.yaml              arranque local de la aplicación
```

El orden y propósito de los cuadernos se explica en `notebooks/README.md`. Los resultados completos de cada fase no se duplican en Git: se incluyen únicamente las evidencias finales necesarias para auditar las conclusiones, junto con los manifiestos de preparación del dataset.

## Dataset y reproducción experimental

El dataset D-Fire original no se redistribuye dentro del repositorio. La fuente oficial es <https://github.com/gaia-solutions-on-demand/DFireDataset>. La versión empleada contiene 17.221 imágenes de entrenamiento y 4.306 de test. Para reproducir la preparación completa, el dataset debe colocarse localmente en `data/D-Fire`.

El notebook `notebooks/01_DFire_preparacion_dataset.ipynb` genera la versión preparada `dfire_seed42_val10_v1`. En `artifacts/datasets/dfire_seed42_val10_v1` se conservan el manifiesto único, la partición train/val/test, los ficheros de auditoría y los YAML de datos. La validación se obtiene reservando el 10 % del entrenamiento original mediante estratificación por tipo de imagen y semilla 42.

La reconstrucción completa de entrenamientos requiere una GPU NVIDIA compatible y el dataset original disponible localmente. La demostración de la aplicación no necesita GPU.

## Privacidad, alcance y limitaciones

- El sistema es un prototipo de apoyo y no sustituye sistemas certificados de detección de incendios.
- La alerta depende de la calidad, encuadre y dominio de las imágenes.
- El test final superó por una imagen el objetivo de un máximo del 1 % de negativas con alarma; el umbral no se reajustó después de observar test.
- La integración con Telegram es opcional y sus credenciales permanecen fuera del código.

## Integridad del modelo

El manifiesto `artifacts/14_final_model_freeze/final/freeze_manifest.json` registra la procedencia y el SHA-256 del checkpoint final. La aplicación verifica ese hash antes de cargarlo.

## Autor

Elías Ruiz Fernández, 2026.
