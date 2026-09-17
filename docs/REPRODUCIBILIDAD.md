# Reproducibilidad y ejecución

## Nivel 1: demostración funcional

Objetivo: probar imágenes, vídeos, cámara, reglas temporales y la interfaz sin repetir el entrenamiento.

```powershell
docker compose up --build
```

La imagen contiene el modelo PyTorch congelado y la exportación NCNN a 640 px. La aplicación se publica únicamente en `127.0.0.1:8000`.

Comprobaciones rápidas:

```powershell
docker compose ps
docker compose run --rm --no-deps -e TFM_APP_MODEL_PATH= app python -m unittest discover -s tests -p "test_fire_app.py" -q
```

## Nivel 2: auditoría de resultados

Los 11 notebooks de la entrega contienen el flujo completo:

1. preparación y particionado;
2. entrenamiento;
3. evaluación inicial;
4. selección de umbral;
5. comparación de resolución;
6. inferencia cualitativa opcional;
7. búsqueda dirigida de hiperparámetros;
8. selección final en validación;
9. comparación final de modelos;
10. evaluación única en test;
11. interpretabilidad.

La entrega no incluye la antigua auditoría perceptual de escenas: D-Fire no
proporciona identificadores fiables de cámara, vídeo o evento y aquel análisis
no intervino en la selección del modelo ni en los resultados finales. Se mantiene
la comprobación bloqueante de duplicados exactos durante la preparación.

Las tablas y figuras esenciales ya están en `results/`, por lo que se pueden revisar sin repetir cómputo.

## Nivel 3: reproducción experimental completa

Requisitos adicionales:

- GPU NVIDIA compatible con CUDA 13.0;
- Docker con acceso a GPU;
- dataset oficial D-Fire en `data/D-Fire`;
- espacio suficiente para checkpoints y cachés.

El entorno exacto está fijado en `docker/requirements.txt`. La imagen de entrenamiento utilizada fue `pytorch/pytorch:2.12.1-cuda13.0-cudnn9-runtime` fijada por digest en el proyecto de trabajo. Para repetir los experimentos se recomienda reconstruir una imagen de entrenamiento con esas dependencias y ejecutar los cuadernos en orden.

La reproducción de entrenamiento no es necesaria para demostrar la aplicación ni para verificar el hash del modelo final.

## Particiones y prevención de fuga

- D-Fire oficial: 17.221 imágenes de entrenamiento y 4.306 de test.
- Validación: 10 % estratificado del train oficial, semilla 42.
- Selección de arquitectura, hiperparámetros y umbrales: exclusivamente en train/validación.
- Evaluación final: una ejecución sobre test después de congelar modelo y umbrales.
- El test no se utilizó para reajustar el umbral, aunque el límite del 1 % se excedió por una imagen.

## Integridad

El checkpoint final se acompaña de `freeze_manifest.json`. El SHA-256 esperado es:

```text
bfb54c4726519fb473bb7c4825577e51c05af2b0db5fd46bf1931a8a29285f67
```

En PowerShell:

```powershell
Get-FileHash artifacts/14_final_model_freeze/final/weights/best.pt -Algorithm SHA256
```

## Telegram

Telegram es una integración opcional. El repositorio solo contiene `.env.example`; las credenciales reales no forman parte de la entrega. En modo `dry_run` se registran los eventos sin realizar peticiones externas.
