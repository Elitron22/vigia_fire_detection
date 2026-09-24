# Vigía en Raspberry Pi 5

Esta guía explica cómo instalar Vigía en una Raspberry Pi 5. Hay dos versiones
del modelo, y se pueden instalar las dos a la vez porque usan puertos y
entornos distintos:

| Versión | Tiempo por imagen en la Pi | Umbrales (humo / fuego) | Puerto | Uso |
|---|---:|---|---:|---|
| **NCNN 640 px** | 167 ms (≈6 imágenes/s) | 0,365 / 0,165 | 8003 | **Recomendada** para la Pi |
| PyTorch 768 px | 2.621 ms (≈0,4 imágenes/s) | 0,36 / 0,16 | 8000 | Modelo final del TFM, como referencia |

La versión PyTorch es exactamente el modelo final evaluado en test, pero en la
Pi es demasiado lenta para vídeo o cámara. La versión NCNN es el mismo modelo
convertido a NCNN, un formato optimizado para procesadores ARM. Como la
conversión cambia ligeramente las confianzas, sus umbrales se recalcularon
usando solo la partición de validación, para mantener las falsas alarmas por
debajo del 1 %. El conjunto de test no se usó en ese proceso.

Los dos modelos ya están en el repositorio:

- NCNN 640 px: `deployment/rpi5/model/yolo26s_640_ncnn_model/`
- PyTorch 768 px: `artifacts/14_final_model_freeze/final/weights/best.pt`

## Requisitos

- Raspberry Pi 5 con Raspberry Pi OS de 64 bits.
- Conexión a Internet en la Pi para instalar dependencias.
- Acceso por SSH desde el PC.

## 1. Descargar el proyecto en la Pi

Conectarse a la Pi por SSH y ejecutar:

```bash
git clone https://github.com/Elitron22/vigia_fire_detection.git
cd vigia_fire_detection

sudo apt update
sudo apt install -y python3-venv python3-pip python3-torch python3-torchvision \
  ffmpeg libgl1 libglib2.0-0
```

PyTorch se instala desde los paquetes de Raspberry Pi OS y no desde `pip`,
porque la versión de `pip` para ARM intentaría descargar dependencias de CUDA
que la Pi no puede usar.

Todos los comandos siguientes se ejecutan desde la carpeta `vigia_fire_detection`.

## 2. Instalar la versión NCNN 640 px (recomendada)

```bash
# Colocar el modelo donde lo espera el servicio
mkdir -p model
cp -r deployment/rpi5/model/yolo26s_640_ncnn_model model/

# Entorno de Python con las dependencias de esta versión
python3 -m venv --system-site-packages .venv-rpi5-ncnn
.venv-rpi5-ncnn/bin/python -m pip install --upgrade pip
.venv-rpi5-ncnn/bin/python -m pip install --no-cache-dir -r deployment/rpi5_ncnn/requirements.txt
```

**Arrancarla como servicio** (se inicia sola al encender la Pi):

```bash
sed -e "s|__TFM_USER__|$USER|g" -e "s|__TFM_ROOT__|$PWD|g" \
  deployment/rpi5_ncnn/tfm-fire-ncnn640.service.template \
  | sudo tee /etc/systemd/system/tfm-fire-ncnn640.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now tfm-fire-ncnn640.service
```

Para ver su estado: `sudo systemctl status tfm-fire-ncnn640`.
Para pararlo: `sudo systemctl stop tfm-fire-ncnn640`.

**O arrancarla a mano** para una prueba rápida:

```bash
TFM_PROJECT_ROOT=$PWD \
TFM_APP_CONFIG=$PWD/configs/app.rpi5.ncnn640.yaml \
TFM_APP_MODEL_PATH=$PWD/model/yolo26s_640_ncnn_model \
TFM_APP_MODEL_ID=yolo26s_640_ncnn_rpi5 \
TFM_APP_MODEL_KEY=yolo26s \
TFM_APP_TRAINED_IMGSZ=768 \
.venv-rpi5-ncnn/bin/python tools/run_detection_app.py --host 127.0.0.1 --port 8003
```

La aplicación queda en el puerto **8003** de la Pi. En su selector aparecen el
modelo NCNN y el PyTorch; en la Pi conviene usar el NCNN.

## 3. Instalar la versión PyTorch 768 px (opcional)

```bash
# Colocar el modelo final donde lo espera el instalador
mkdir -p model
cp artifacts/14_final_model_freeze/final/weights/best.pt model/yolo26s_768_final.pt

# Instalar (añadir --service para que arranque sola al encender la Pi)
bash deployment/rpi5/install.sh --service
```

El instalador crea su propio entorno de Python (`.venv-rpi5`), comprueba que el
modelo es el correcto y, con `--service`, instala el servicio `tfm-fire`. Sin
`--service`, al terminar muestra el comando para arrancarla a mano. La
aplicación queda en el puerto **8000** de la Pi.

## 4. Abrir la interfaz desde el PC

Por seguridad, la aplicación solo acepta conexiones desde la propia Pi. Para
abrirla desde el PC se crea un túnel SSH (sustituir `usuario` y
`raspberrypi.local` por los de tu Pi):

```bash
# Versión NCNN
ssh -L 8003:127.0.0.1:8003 usuario@raspberrypi.local

# Versión PyTorch
ssh -L 8000:127.0.0.1:8000 usuario@raspberrypi.local
```

Con la sesión abierta, entrar desde el navegador del PC en
<http://127.0.0.1:8003> (NCNN) o <http://127.0.0.1:8000> (PyTorch). El túnel
también permite usar la cámara del PC desde la interfaz. No se recomienda
abrir estos puertos directamente a Internet.

## 5. Avisos por Telegram (opcional)

Los dos servicios leen las credenciales del fichero `.env.telegram`, en la
carpeta `vigia_fire_detection` de la Pi. Para activarlos, crear ese fichero con:

```text
TFM_TELEGRAM_MODE=live
TELEGRAM_BOT_TOKEN=<token del bot>
TELEGRAM_CHAT_ID=<identificador del chat>
```

y reiniciar el servicio correspondiente (`sudo systemctl restart tfm-fire-ncnn640`
o `sudo systemctl restart tfm-fire`). La sección 2 del `README.md` principal
explica cómo obtener el token y el identificador del chat.

## 6. Medir la velocidad en la Pi

Copiar a la Pi una carpeta con algunas imágenes de prueba y ejecutar:

```bash
# Versión NCNN 640 px
.venv-rpi5-ncnn/bin/python tools/benchmark_rpi5.py \
  --model model/yolo26s_640_ncnn_model --imgsz 640 \
  --smoke-threshold 0.365 --fire-threshold 0.165 \
  --images ~/imagenes_benchmark --runs 20

# Versión PyTorch 768 px
.venv-rpi5/bin/python tools/benchmark_rpi5.py \
  --model model/yolo26s_768_final.pt \
  --images ~/imagenes_benchmark --runs 20
```

El resultado (tiempo medio, mediana, percentil 95, imágenes por segundo,
temperatura y memoria) se guarda en `artifacts/rpi5_benchmark.json`. Usar
`--output` para guardar cada medición en un fichero distinto.

## Resultados obtenidos

Medidos en una Raspberry Pi 5 de 16 GB con Raspberry Pi OS de 64 bits, seis
imágenes y 20 repeticiones:

| Versión | Resolución | Tiempo por imagen | Imágenes por segundo |
|---|---:|---:|---:|
| PyTorch | 768 px | 2.621,36 ms | 0,381 |
| NCNN | 768 px | 252,04 ms | 3,968 |
| NCNN | 640 px | 167,41 ms | 5,973 |

En todas las pruebas la temperatura se mantuvo entre 45,75 y 50,15 °C, el uso
de memoria por debajo de 600 MB y la Pi no redujo su velocidad por
calentamiento. Los datos completos están en `results/rpi5/`.

La versión NCNN a 768 px (umbrales 0,415 / 0,165) se probó pero se descartó en
favor de la de 640 px, que tarda un 33 % menos por imagen con métricas muy parecidas.
Su modelo no se incluye en el repositorio.

## Cámara CSI opcional

La interfaz web usa la cámara del equipo que abre el navegador. Para usar una
cámara conectada directamente a la Pi, Raspberry Pi OS utiliza `rpicam` y
Picamera2:

```bash
sudo apt install -y python3-picamera2 --no-install-recommends
rpicam-hello --list-cameras
```

La aplicación no lee todavía esta cámara directamente; habría que añadirla
como fuente nueva.

## Ficheros de esta carpeta y de `deployment/rpi5_ncnn/`

| Fichero | Para qué sirve |
|---|---|
| `deployment/rpi5/install.sh` | Instalador de la versión PyTorch. |
| `deployment/rpi5/tfm-fire.service.template` | Servicio de la versión PyTorch (puerto 8000). |
| `deployment/rpi5/requirements.txt` | Dependencias de la versión PyTorch. |
| `deployment/rpi5/model/yolo26s_640_ncnn_model/` | Modelo NCNN 640 px y sus informes de calibración. |
| `deployment/rpi5_ncnn/tfm-fire-ncnn640.service.template` | Servicio de la versión NCNN 640 px (puerto 8003). |
| `deployment/rpi5_ncnn/requirements.txt` | Dependencias de la versión NCNN. |
| `deployment/rpi5_ncnn/install.sh`, `tfm-fire-ncnn.service.template` | Instalación de la versión NCNN 768 px descartada; no se usan en esta guía. |
