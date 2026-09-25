# Vigía: aplicación de detección de humo y fuego

Aplicación web del TFM. Permite analizar una imagen, procesar un vídeo o usar
la cámara del navegador con el modelo final, confirmar las detecciones con una
regla temporal antes de dar una alerta y avisar por Telegram.

Está hecha con FastAPI (servidor y API), Ultralytics YOLO y OpenCV
(inferencia), y una interfaz sencilla en HTML, CSS y JavaScript sin frameworks.

## Estructura

| Fichero | Contenido |
|---|---|
| `main.py` | API y rutas web |
| `inference.py` | Carga de modelos e inferencia en imagen, vídeo y fotogramas de cámara |
| `alerting.py` | Regla temporal de alertas y registro de eventos |
| `telegram.py` | Envío de avisos por Telegram |
| `model_registry.py` | Descubrimiento de modelos disponibles y comprobación de su SHA-256 |
| `config.py` | Lectura y validación de la configuración |
| `static/` | Interfaz web |

## Puesta en marcha

### Con Docker (recomendado)

Desde la raíz del repositorio:

```bash
docker compose up --build
```

Abrir <http://127.0.0.1:8000>. La imagen (`Dockerfile` y `compose.yaml`) ya
incluye el modelo PyTorch final a 768 px y la versión NCNN a 640 px, así que no
hace falta entrenar ni descargar el dataset. Funciona solo en CPU. La sección 1
del `README.md` principal explica los requisitos y el uso de la interfaz.

Con Docker, la aplicación usa el perfil `configs/app.release.yaml` y guarda los
resultados en la carpeta `runtime/` del repositorio:

- `runtime/results/images/<id>/` y `runtime/results/videos/<id>/`: resultados
  anotados y un `metadata.json` con los parámetros y detecciones de cada análisis.
- `runtime/events.jsonl`: registro de todas las alertas.

### Sin Docker (opcional)

Con Python 3.12, desde la raíz del repositorio y preferiblemente dentro de un
entorno virtual (`python -m venv .venv`), instalar PyTorch y torchvision
(<https://pytorch.org/get-started/locally/>) y después:

```bash
pip install -r requirements-app.txt
python tools/run_detection_app.py --host 127.0.0.1
```

Sin indicar nada, usa el perfil `configs/app.yaml`: carga el modelo PyTorch
final y guarda los resultados en `artifacts/08_detection_app/`. Este perfil
escucha en `0.0.0.0` (todas las interfaces de red); con `--host 127.0.0.1` la
aplicación solo es accesible desde el propio ordenador. El puerto se cambia con
`--port`. Por defecto se ejecuta en CPU; si PyTorch tiene soporte CUDA, se puede
usar la GPU definiendo `TFM_APP_DEVICE=0`.

## Perfiles de configuración

El perfil se elige con la variable de entorno `TFM_APP_CONFIG`.

| Perfil | Dónde se usa | Modelos disponibles |
|---|---|---|
| `configs/app.release.yaml` | Docker (`compose.yaml`) | PyTorch 768 px y NCNN 640 px |
| `configs/app.yaml` | Ejecución local sin Docker | Modelo PyTorch final (parámetros ajustables por API) |
| `configs/app.rpi5.ncnn640.yaml` | Raspberry Pi, versión recomendada | NCNN 640 px (por defecto) y PyTorch 768 px |
| `configs/app.rpi5.yaml` | Raspberry Pi, versión PyTorch | PyTorch 768 px |
| `configs/app.rpi5.ncnn.yaml` | Versión NCNN 768 px descartada | NCNN 768 px (su modelo no está en el repositorio) |

En la interfaz se pueden mover los umbrales de humo y fuego y el tiempo mínimo
para alertar. La resolución, el IoU y la confianza mínima de inferencia no
aparecen en la interfaz: con `configs/app.yaml` se pueden cambiar por API (con
los parámetros `imgsz`, `nms_iou` y `confidence`), y en los perfiles de Docker
y de Raspberry Pi quedan fijados a los del TFM.

## Modelos y umbrales

El modelo PyTorch final se carga desde
`artifacts/14_final_model_freeze/final/`. Antes de cargarlo, la aplicación
comprueba que su SHA-256 coincide con el de `freeze_manifest.json`.

Se puede añadir un segundo modelo, como la versión NCNN, con estas variables de
entorno (Docker y los servicios de la Raspberry Pi ya las definen):

| Variable | Contenido | Valor por defecto |
|---|---|---|
| `TFM_APP_MODEL_PATH` | Ruta al modelo (fichero `.pt` o carpeta NCNN) | — |
| `TFM_APP_MODEL_ID` | Identificador del modelo; debe coincidir con el del perfil (por ejemplo `yolo26s_640_ncnn_rpi5` para el NCNN 640) | `yolo26s_final_rpi5` |
| `TFM_APP_TRAINED_IMGSZ` | Resolución con la que se entrenó | `768` |
| `TFM_APP_MODEL_KEY` | Arquitectura | `yolo26s` |
| `TFM_APP_MODEL_SHA256` | Huella SHA-256 que debe tener el fichero (opcional) | — |

Por ejemplo, para usar el NCNN 640 con el perfil de Docker hay que definir las
mismas variables que `compose.yaml`.

Cada modelo tiene sus propios umbrales de confianza por clase: una detección
solo se muestra y cuenta para las alertas si alcanza o supera el umbral de su
clase.

| Modelo | Humo | Fuego | Origen |
|---|---:|---:|---|
| YOLO26s PyTorch 768 px | 0,36 | 0,16 | Elegidos en validación con un máximo del 1 % de imágenes sin humo ni fuego con alarma |
| YOLO26s NCNN 640 px | 0,365 | 0,165 | Recalculados en validación, porque la conversión a NCNN cambia ligeramente las confianzas |

La interfaz carga los umbrales del modelo seleccionado. Se pueden mover con los
controles deslizantes de «Ajustes avanzados» para hacer pruebas (el botón
«Restaurar valores recomendados» los devuelve a su valor), pero con otros
valores los resultados ya no corresponden a las métricas del TFM.

## Alertas

Para que una alerta no salte por un único fotograma, la regla se aplica por
separado para humo y para fuego:

1. La confianza de la detección debe alcanzar o superar el umbral de su clase.
2. La detección debe mantenerse durante `hold_seconds` (3 s por defecto, ajustable en la interfaz).
3. Si desaparece más de `maximum_gap_seconds` (0,75 s), la cuenta vuelve a empezar.
4. Tras `clear_seconds` (5 s) sin detecciones, la alerta se cierra.
5. Después hay un `cooldown_seconds` (60 s) antes de poder abrir otra alerta de la misma clase.

En una imagen suelta, cualquier detección por encima del umbral genera la
alerta directamente. En vídeo se usa el tiempo del propio vídeo y en la cámara
el tiempo real.

Cada imagen, vídeo o sesión de cámara envía como máximo un aviso a Telegram.
Las alertas siguientes de la misma fuente se guardan en el registro marcadas
como `suppressed`, sin repetir el mensaje.

Al terminar un vídeo, la interfaz muestra el número de detecciones y la
confianza máxima de cada clase, el tiempo medio de procesamiento por fotograma
y el número de eventos de alerta.

## Telegram

Por defecto la aplicación funciona en modo simulación (`dry_run`): registra las
alertas pero no envía nada. Para enviar avisos reales hacen falta tres
variables:

```text
TFM_TELEGRAM_MODE=live
TELEGRAM_BOT_TOKEN=<token del bot>
TELEGRAM_CHAT_ID=<identificador del chat>
```

- **Con Docker:** en el fichero `.env` de la raíz del repositorio (copiando
  `.env.example`). La sección 2 del `README.md` principal explica cómo obtener
  el token y el identificador.
- **En la Raspberry Pi:** en el fichero `.env.telegram` de la carpeta del
  proyecto, que los servicios de systemd leen al arrancar (ver
  `deployment/README.md`).

Los dos ficheros están excluidos de Git. El botón «Enviar mensaje de prueba»
(en «Diagnóstico del sistema») se activa cuando el modo es `live` y hay
credenciales, y envía un mensaje de prueba.

Cada aviso incluye la clase (humo o fuego), la confianza máxima, el instante,
el origen y una captura anotada. Si hay humo y fuego a la vez se envía un solo
aviso, dando prioridad al fuego. Si Telegram falla, el error se guarda en el
registro y el análisis continúa.

## API

La documentación interactiva está en <http://127.0.0.1:8000/docs>.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/health` | Estado del servicio, modelo por defecto y dispositivo |
| `GET /api/models` | Modelos disponibles, su resolución y sus umbrales |
| `GET /api/config` | Configuración pública y estado de Telegram |
| `GET /api/events` | Últimas alertas registradas |
| `POST /api/telegram/test` | Envía un mensaje de prueba a Telegram |
| `POST /api/analyze/image` | Analiza una imagen; devuelve las detecciones en JSON y la imagen anotada |
| `POST /api/analyze/video` | Analiza un vídeo; devuelve el vídeo anotado (MP4, o AVI si no hay `ffmpeg`) |
| `WS /api/live` | Recibe fotogramas de la cámara y responde con detecciones y estado de las alertas |

Las rutas de análisis reciben el fichero directamente como cuerpo de la
petición y su nombre en la cabecera `X-Filename`. Admiten los parámetros de
consulta `experiment_id`, `smoke_threshold` y `fire_threshold` (y, en vídeo,
`hold_seconds`), además de `confidence`, `nms_iou` e `imgsz` cuando el perfil lo
permite. El tamaño máximo de subida lo fija `max_upload_mib` en el perfil. La respuesta de vídeo incluye
además las cabeceras `X-TFM-Run-Id`, `X-TFM-Event-Count`,
`X-TFM-Detection-Summary` (detecciones y confianza máxima por clase) y
`X-TFM-Processing-Ms-Per-Frame`.

## Pruebas

Con la imagen de Docker ya construida:

```bash
docker compose run --rm --no-deps -e TFM_APP_MODEL_PATH= app python -m unittest discover -s tests -p "test_fire_app.py" -q
```

Las pruebas no cargan ningún modelo real. Comprueban la regla de alertas, los
umbrales por clase, el descubrimiento de modelos, el registro de eventos, los
avisos de Telegram y las rutas de la API.

`tests/smoke_fire_app_end_to_end.py` es una prueba adicional que sí carga el
modelo final y analiza una imagen real, un vídeo generado a partir de ella y
el WebSocket de la cámara. Se ejecuta sin Docker, con el entorno de la sección
"Sin Docker" e indicando una imagen en la que se vea humo o fuego (por ejemplo,
una del dataset D-Fire), porque la prueba comprueba que salta una alerta:

```bash
python tests/smoke_fire_app_end_to_end.py --image ruta/a/una/imagen.jpg
```

## Raspberry Pi 5

Las imágenes de Docker del repositorio son para ordenadores x86-64 y no sirven
en la Raspberry Pi. En la Pi se pueden instalar las dos versiones del modelo
(NCNN 640 px, recomendada, y PyTorch 768 px); la guía completa está en
`deployment/README.md`.
