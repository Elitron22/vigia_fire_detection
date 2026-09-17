# Mapa de evidencias

| Afirmación | Evidencia incluida |
|---|---|
| Modelo y umbrales congelados antes de test | `artifacts/14_final_model_freeze/final/freeze_manifest.json` |
| Métricas finales en 4.306 imágenes | `results/final_test/RESUMEN_EVALUACION_FINAL_TEST.md` y CSV asociados |
| Comparación validación frente a test | `results/final_test/figures/01_operating_validation_vs_test.png` |
| Casos representativos y errores | `results/final_test/figures/05_representative_examples.png` |
| Interpretabilidad por CAM y oclusión | `results/interpretability/RESUMEN_INTERPRETABILIDAD.md` |
| Validación cuantitativa de explicaciones | `results/interpretability/07_deletion_curve.png` |
| Latencia real en Raspberry Pi 5 | `results/rpi5/*benchmark.json` |
| Selección de NCNN a 640 px | `results/rpi5/comparison_report.json` |
| Auditoría móvil de la aplicación | `results/app_audit/README.md` y capturas |
| Pruebas automatizadas | `tests/` |

Los resultados exhaustivos y cachés intermedias no se duplican en esta copia limpia. Las evidencias seleccionadas son suficientes para reconstruir las tablas de la memoria y comprobar cada conclusión principal.
