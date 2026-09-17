"""Construye el notebook de consulta de la evaluación final; nunca ejecuta inferencia."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "11_DFire_evaluacion_final_test.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nb["cells"] = [
    markdown("""
# D-Fire · Evaluación final congelada en test

## tl;dr

Se evaluó **una sola configuración**: YOLO26s entrenado e inferido a 768,
checkpoint congelado, humo 0,36 y fuego 0,16. En test obtiene **67,88 % de
precisión micro, 77,32 % de recall micro, 72,29 % de F1 micro, 79,27 % de
mAP50 y 46,38 % de mAP50-95**. Activa 21/2.005 negativas (1,047 %): supera el
límite del 1 % por una sola imagen. La configuración no se reajusta.

Este cuaderno es deliberadamente de **solo consulta**. No contiene una bandera
para volver a evaluar, buscar umbrales o comparar modelos.
"""),
    markdown("""
## Contexto y método

### Supuestos y bloqueo

- Selección previa: `yolo26s_dfire_seed42_20260912T165304Z`.
- Entrenamiento/inferencia: 768→768.
- Umbrales operativos: humo 0,36; fuego 0,16; IoU de acierto 0,50.
- Límite previo: ≤1 % de imágenes completamente negativas con alguna alarma.
- Las métricas estándar de Ultralytics aportan mAP; TP/FP/FN usan los umbrales
  congelados por clase.
- Los ejemplos son diagnósticos post hoc y no se usan para modificar el sistema.
- Consta una evaluación histórica del antiguo baseline YOLOv8s en test; el
  YOLO26s congelado no se había evaluado antes. Por ello no se describe test
  como completamente virgen.
"""),
    code("""
from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown

def find_root(start=Path.cwd()):
    for candidate in [start, *start.parents, Path('/workspace/TFM')]:
        if (candidate / 'artifacts/15_final_test_evaluation/test/evaluation_state.json').is_file():
            return candidate.resolve()
    raise FileNotFoundError('No se encuentra el artefacto final de test')

ROOT = find_root()
PARENT = ROOT / 'artifacts/15_final_test_evaluation/test'
state = json.loads((PARENT / 'evaluation_state.json').read_text(encoding='utf-8'))
assert state['status'] == 'complete'
assert state['single_final_evaluation'] is True
assert state['inference_attempts'] == 1
RUN = ROOT / state['run_rel']
summary = json.loads((RUN / 'run_summary.json').read_text(encoding='utf-8'))
verification = json.loads((RUN / 'verification_report.json').read_text(encoding='utf-8'))
assert verification['status'] == 'passed'
assert summary['threshold_search_executed'] is False
assert summary['model_comparison_executed'] is False
print(f"Artefacto cerrado: {RUN.relative_to(ROOT)}")
print(f"Checkpoint SHA-256: {summary['weights_sha256']}")
print(f"Verificación: {verification['status']} · intentos de inferencia: {state['inference_attempts']}")
"""),
    markdown("""
## Datos

La población se reconcilia con el manifiesto congelado antes de mostrar cifras.
"""),
    code("""
population = summary['observed_test_population']
pd.DataFrame([
    {'Unidad': 'Imágenes', 'Test': population['images']},
    {'Unidad': 'Imágenes negativas', 'Test': population['negative_images']},
    {'Unidad': 'Cajas de humo', 'Test': population['smoke_boxes']},
    {'Unidad': 'Cajas de fuego', 'Test': population['fire_boxes']},
]).style.format({'Test': '{:,.0f}'})
"""),
    markdown("""
## Resultados

### 1. Métricas estándar

El global es macro entre clases. El F1 mostrado es la media armónica de la
precisión y recall estándar; mAP50-95 promedia IoU 0,50:0,95.
"""),
    code("""
standard = pd.read_csv(RUN / 'test_standard_metrics.csv')
standard_display = standard[['scope','precision','recall','f1','mAP50','mAP50_95']].copy()
standard_display['scope'] = standard_display.scope.map({'all_macro':'Global','smoke':'Humo','fire':'Fuego'})
display(standard_display.style.format({c:'{:.2%}' for c in ['precision','recall','f1','mAP50','mAP50_95']}))
"""),
    markdown("""
### 2. Punto operativo congelado

Estas son las cifras que corresponden a humo 0,36 y fuego 0,16.
"""),
    code("""
operating = pd.read_csv(RUN / 'test_operating_metrics.csv').iloc[0]
classes = pd.read_csv(RUN / 'test_class_metrics.csv')
class_display = classes[classes.scope.isin(['smoke','fire'])][['scope','tp','fp','fn','precision','recall','f1']].copy()
class_display['scope'] = class_display.scope.map({'smoke':'Humo','fire':'Fuego'})
display(class_display.style.format({c:'{:.2%}' for c in ['precision','recall','f1']}))
pd.DataFrame([{
    'Precisión micro': operating.precision,
    'Recall micro': operating.recall,
    'F1 micro': operating.f1,
    'Recall macro': operating.macro_recall,
    'Recall de la clase peor': operating.minimum_class_recall,
}]).style.format('{:.2%}')
"""),
    code("""
display(Image(filename=str(RUN / 'figures/01_operating_validation_vs_test.png'), width=900))
"""),
    markdown("""
Respecto a validación, test pierde 0,94 puntos de precisión micro, 1,60 de
recall micro, 1,23 de F1 micro y 1,42 de recall macro. Es una degradación
moderada; la mayor caída por clase aparece en el recall operativo de fuego
(−2,92 puntos).
"""),
    markdown("""
### 3. Recall por clase y tamaño
"""),
    code("""
sizes = pd.read_csv(RUN / 'test_size_metrics.csv')
size_display = sizes[['class_name','size_band','gt_boxes','tp','fn','recall']].copy()
size_display['class_name'] = size_display.class_name.map({'smoke':'Humo','fire':'Fuego'})
size_display['size_band'] = size_display.size_band.map({'small':'Pequeño (<1 %)','medium':'Mediano (1–10 %)','large':'Grande (≥10 %)'})
display(size_display.style.format({'recall':'{:.2%}'}))
display(Image(filename=str(RUN / 'figures/03_size_validation_vs_test.png'), width=1000))
"""),
    markdown("""
La principal dificultad continúa en los objetos pequeños: 66,06 % de recall
para humo y 73,60 % para fuego. La caída de fuego grande (−12,14 puntos) debe
leerse con su denominador: 170 cajas en test frente a 71 en validación.
"""),
    markdown("""
### 4. Errores y alarmas sobre negativas
"""),
    code("""
confusion = pd.read_csv(RUN / 'test_confusion_matrix.csv', index_col=0)
alarms = pd.read_csv(RUN / 'test_negative_image_alarms.csv')
display(confusion.rename(index={'smoke':'Humo real','fire':'Fuego real','background':'Fondo real'},
                         columns={'smoke':'Pred. humo','fire':'Pred. fuego','background':'Pred. fondo'}))
display(alarms.style.format({'negative_image_false_alarm_rate':'{:.3%}'}))
display(Image(filename=str(RUN / 'figures/04_errors_and_negative_alarms.png'), width=1000))
"""),
    markdown("""
El sistema activa 21/2.005 negativas (1,047 %). El máximo compatible con ≤1 %
era 20/2.005; el exceso es exactamente una imagen. Se conserva este incumplimiento
tal cual porque elevar un umbral después de observar test equivaldría a ajustar
con el conjunto final.
"""),
    markdown("""
### 5. Ejemplos representativos

Selección dirigida: alarmas negativas, falsos negativos, FP en positivas,
confusiones de clase y aciertos. No estima la frecuencia de causas visuales.
"""),
    code("""
examples = pd.read_csv(RUN / 'representative_examples.csv')
display(examples[['order','selection_reason','filename','gt_boxes','tp','fp','fn','max_confidence']])
display(Image(filename=str(RUN / 'figures/05_representative_examples.png'), width=1100))
"""),
    markdown("""
## Takeaways

1. El YOLO26s 768→768 mantiene una generalización razonable: mAP50-95 pasa de
   46,68 % a 46,38 % y F1 micro operativo de 73,52 % a 72,29 %.
2. Humo es más preciso (84,41 %) y fuego conserva mayor carga de FP, con 58,26 %
   de precisión a cambio de 75,61 % de recall.
3. El objetivo de alarmas negativas queda marginalmente fuera: 1,047 % frente a
   1 %. Debe informarse, no corregirse con test.
4. La evaluación final queda cerrada. Los pasos siguientes son documentar estos
   resultados y evaluar exportación/latencia en el dispositivo, usando datos
   externos o validación para cualquier desarrollo adicional.
"""),
]

TARGET.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, TARGET)
print(TARGET)

