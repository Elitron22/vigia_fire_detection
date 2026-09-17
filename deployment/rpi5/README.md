# Despliegue en Raspberry Pi 5

Este perfil ejecuta el checkpoint PyTorch YOLO26s final a 768 px con los
umbrales congelados del TFM: humo `0.36` y fuego `0.16`. Está preparado para
Raspberry Pi OS de 64 bits y no usa la imagen CUDA/x86-64 del entrenamiento.
El instalador usa los paquetes CPU `python3-torch` y `python3-torchvision` de
Raspberry Pi OS/Debian. Esto evita que el wheel ARM genérico de PyPI descargue
dependencias CUDA destinadas a otras plataformas ARM.

Se evaluó también una exportación NCNN sobre las 1.721 imágenes de validación.
Con los umbrales PyTorch elevaba las negativas con alarma a 10/783 (1,277 %),
por lo que no es un sustituto transparente. Después se calibró como una variante
de despliegue separada, solo con validación, en humo `0.415` y fuego `0.165`:
7/783 alarmas negativas (0,894 %). Su paquete, servicio y resultados están en
`deployment/rpi5_ncnn/`; no modifica este perfil ni consulta test.

## 1. Preparar el artefacto en el PC

Desde la raíz del proyecto:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml exec -T notebook `
  python tools/build_rpi5_bundle.py
```

El resultado es `dist/tfm-fire-rpi5.tar.gz`.

## 2. Copiar e instalar

Sustituir `usuario` y `raspberrypi.local` por los valores reales:

```powershell
scp dist/tfm-fire-rpi5.tar.gz usuario@raspberrypi.local:/home/usuario/
ssh usuario@raspberrypi.local
tar -xzf ~/tfm-fire-rpi5.tar.gz -C ~
cd ~/tfm-fire-rpi5
bash deployment/rpi5/install.sh
```

Para instalar también el servicio de arranque automático:

```bash
bash deployment/rpi5/install.sh --service
```

El servicio escucha solo en `127.0.0.1:8000`. Desde el PC se abre un túnel:

```powershell
ssh -L 8000:127.0.0.1:8000 usuario@raspberrypi.local
```

Mientras esa sesión esté abierta, visitar `http://127.0.0.1:8000`. Este método
también permite que el navegador autorice la cámara al tratar la página como un
origen local. No se recomienda publicar el puerto 8000 directamente en Internet.

## 3. Benchmark en la Pi

Copiar a la Pi un pequeño directorio de imágenes de validación o imágenes
manuales que no se vaya a usar para recalibrar el sistema y ejecutar:

```bash
cd ~/tfm-fire-rpi5
source .venv-rpi5/bin/activate
export TFM_APP_MODEL_PATH="$PWD/model/yolo26s_768_final.pt"
python tools/benchmark_rpi5.py --images ~/imagenes_benchmark --runs 30
```

El informe guarda latencia media/mediana/p95, FPS, temperatura, memoria máxima y
el estado de throttling. El benchmark no cambia el modelo ni los umbrales.

### Medición real inicial

En una Raspberry Pi 5 de 16 GB con Raspberry Pi OS de 64 bits, PyTorch 2.6 CPU,
seis imágenes y 20 repeticiones a 768 px se obtuvo una latencia media de
`2621.36 ms` (`0.381 FPS`), p95 de `2626.29 ms`, temperatura de `49.05` a
`50.15 °C` y `throttled=0x0`. Por tanto, el límite observado es el backend CPU,
no la memoria ni la refrigeración. El resultado completo queda en
`artifacts/rpi5_benchmark.json` dentro de la Pi.

## Cámara CSI opcional

La interfaz web usa la cámara del equipo que abre el navegador. Para una cámara
CSI conectada físicamente a la Pi, Raspberry Pi OS actual utiliza `rpicam` y
Picamera2. Se puede instalar, si no viene ya incluido, con:

```bash
sudo apt install -y python3-picamera2 --no-install-recommends
rpicam-hello --list-cameras
```

La captura CSI continua no está activada por defecto en esta aplicación: debe
añadirse como fuente explícita después de verificar el rendimiento térmico y la
cadencia alcanzable con el modelo final.
