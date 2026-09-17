# Aplicación de detección de humo y fuego

Primera aplicación operativa del TFM. Permite seleccionar un experimento
completo, analizar una imagen, procesar un vídeo y usar la cámara del navegador.
La API conserva resultados y metadatos reproducibles en
`artifacts/08_detection_app/`.

## Puesta en marcha

Las dependencias y el puerto 8000 ya forman parte de la imagen del proyecto. Al
terminar el entrenamiento que esté en curso, reconstruir el contenedor una vez:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up -d --build
docker compose exec notebook python tools/run_detection_app.py
```

Abrir <http://127.0.0.1:8000>. Durante un entrenamiento también puede arrancarse
la aplicación en CPU dentro del contenedor actual, pero el mapeo nuevo del puerto
8000 solo se aplica al recrearlo. No se debe recrear el contenedor mientras haya
un entrenamiento en marcha.

La configuración está en `configs/app.yaml`. El dispositivo queda fijado a CPU
en `compose.yaml` para no competir por la GPU con los entrenamientos. Cuando se
quiera medir la ejecución final con GPU se puede anular mediante
`TFM_APP_DEVICE=0`.

## Selección del modelo

La primera opción procede del manifiesto congelado en
`artifacts/14_final_model_freeze/final/`: YOLO26s entrenado e inferido a 768.
Después se muestran los experimentos completos que sigan disponibles. En un
dispositivo externo puede fijarse un modelo exportado con
`TFM_APP_MODEL_PATH`; esto permite cargar el directorio NCNN de Raspberry Pi sin
copiar todos los experimentos.

El perfil final `configs/app.rpi5.ncnn640.yaml` limita expresamente el selector
a dos variantes: YOLO26s NCNN con inferencia a 640 para Raspberry Pi y el
YOLO26s PyTorch congelado, evaluado en test con inferencia a 768. La resolución
e IoU se aplican automáticamente y no se muestran como controles en la
interfaz final.

La interfaz separa los umbrales operativos de humo y fuego. Sus valores
iniciales son los calibrados en validación (`0,365` y `0,165`); pueden ajustarse
para pruebas exploratorias, pero cualquier cambio deja de corresponder al punto
operativo y a las métricas oficiales del TFM.

La inferencia usa 0,16 como suelo para no descartar fuego antes del filtrado por
clase. Las detecciones que se muestran y alimentan la regla temporal respetan el
punto operativo final: 0,36 para humo y 0,16 para fuego, seleccionado en
validación con el límite previo del 1 % de imágenes negativas con alarma. La
evaluación posterior en test no se utiliza para modificar estos valores.

## Alertas temporales

La regla se evalúa por separado para `smoke` y `fire`:

1. La confianza debe superar el umbral de su clase.
2. La presencia debe mantenerse durante `hold_seconds`.
3. Una interrupción superior a `maximum_gap_seconds` reinicia la confirmación.
4. Tras desaparecer durante `clear_seconds`, el incidente se cierra.
5. `cooldown_seconds` impide alertas repetidas inmediatamente después.

Los eventos quedan en `artifacts/08_detection_app/events.jsonl`. En imágenes
aisladas, una detección que supere el umbral genera una alerta inmediata. En
vídeo se usa el tiempo del propio archivo; en cámara se usa tiempo monotónico y
se exige presencia durante `hold_seconds` en ambos casos. Cada imagen, vídeo o
sesión de cámara envía como máximo un aviso a Telegram; los disparos posteriores
de esa misma fuente se registran como `suppressed` para conservar trazabilidad
sin repetir mensajes.

## Telegram

El modo inicial es `dry_run`: registra qué habría enviado sin realizar ninguna
petición externa. Para enviar mensajes reales, crear el bot con `@BotFather`,
enviarle primero `/start` y guardar las credenciales en `.env.telegram`, archivo
excluido de Git:

```powershell
.\deployment\local\configure_telegram.ps1
.\deployment\local\run_ncnn640.ps1
```

El botón «Probar Telegram» se habilita cuando la aplicación está en modo `live`
y ambas credenciales están configuradas. En Raspberry Pi se utiliza el mismo
archivo `.env.telegram` en la raíz de la instalación y systemd lo carga sin
incorporar los secretos al fichero de servicio.

Al activarse un incidente se envían la clase, la confianza máxima, el instante,
el origen y una captura anotada. Si una imagen contiene humo y fuego se envía un
solo aviso, priorizando fuego. Los fallos de Telegram se escriben en el evento y
no interrumpen el análisis.

## API

- `GET /api/health`: estado del servicio, modelos y dispositivo.
- `GET /api/models`: checkpoints registrados disponibles.
- `GET /api/config`: parámetros públicos y estado de Telegram.
- `GET /api/events`: últimos eventos temporales.
- `POST /api/analyze/image`: bytes de una imagen; devuelve JSON y JPG anotado.
- `POST /api/analyze/video`: bytes de un vídeo; devuelve MP4/H.264 cuando
  `ffmpeg` está disponible y AVI/MJPEG como respaldo portable.
- `WS /api/live`: fotogramas JPEG y respuestas con detecciones y estado.

Los endpoints HTTP reciben el archivo como cuerpo binario y el nombre original
en la cabecera `X-Filename`; así no se añade una dependencia de formularios. La
documentación OpenAPI está disponible en `/docs`.

## Verificación

```powershell
docker compose exec notebook python -m unittest discover -s tests
docker compose exec notebook python -m py_compile fire_app/*.py tools/run_detection_app.py
docker compose exec notebook python tests/smoke_fire_app_end_to_end.py
```

Las pruebas cubren la continuidad temporal, los cortes, el cierre, el tiempo de
espera, los umbrales por clase, el registro de eventos y los endpoints de
configuración. La comprobación funcional de inferencia se realiza en CPU para
no interferir con el entrenamiento.

## Raspberry Pi 5

La imagen Docker de entrenamiento es CUDA/x86-64 y no es válida para la Pi.
El perfil ARM64 despliega inicialmente el checkpoint PyTorch congelado; la
exportación NCNN probada no conservó el límite del 1 % de alarmas negativas.
El instalador, el servicio y el benchmark están documentados en
`deployment/rpi5/README.md`.
