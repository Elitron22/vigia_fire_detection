# Variante NCNN para Raspberry Pi 5

Esta variante de despliegue es independiente del modelo PyTorch final del TFM.
Se exportó desde el mismo checkpoint YOLO26s 768 y sus umbrales (`smoke=0.415`,
`fire=0.165`) se calibraron exclusivamente con validación. No se ejecutó
inferencia sobre test para seleccionarla o calibrarla.

Desde la raíz del proyecto se construye e instala con:

```powershell
docker compose -f compose.yaml exec -T notebook python tools/build_rpi5_ncnn_bundle.py
scp dist/tfm-fire-rpi5-ncnn.tar.gz fireai@fireai.local:/home/fireai/
ssh fireai@fireai.local
```

```bash
tar -xzf ~/tfm-fire-rpi5-ncnn.tar.gz -C ~
cd ~/tfm-fire-rpi5-ncnn
bash deployment/rpi5_ncnn/install.sh --service
```

El instalador verifica primero el SHA-256 de cada fichero incluido en el
manifiesto y que la calibración aprobada procede de `val`, no de `test`.

El servicio `tfm-fire-ncnn` escucha en `127.0.0.1:8002`, por lo que puede
convivir con `tfm-fire` (PyTorch, puerto 8000). Desde el PC:

```powershell
ssh -L 8002:127.0.0.1:8002 fireai@fireai.local
```

Después se abre `http://127.0.0.1:8002`. Para medir rendimiento en la Pi:

```bash
cd ~/tfm-fire-rpi5-ncnn
source .venv-rpi5-ncnn/bin/activate
python tools/benchmark_rpi5.py \
  --model model/yolo26s_768_ncnn_model \
  --images ~/tfm-fire-rpi5/benchmark_images \
  --smoke-threshold 0.415 --fire-threshold 0.165 --runs 20
```

## Prueba alternativa a 640 px

Se exportó y calibró también `yolo26s_640_ncnn_model`, siempre sobre validación.
Usa humo `0.365` y fuego `0.165`, y se sirve de manera independiente mediante
`tfm-fire-ncnn640` en `127.0.0.1:8003`. En la misma Pi y protocolo alcanzó
`167.41 ms` (`5.97 FPS`), frente a `252.04 ms` (`3.97 FPS`) a 768 px. El
informe comparativo está en
`artifacts/17_rpi5_deployment_validation/validation/ncnn_resolution_comparison/`.
