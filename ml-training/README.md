# ml-training

Notebooks y datos para el entrenamiento del modelo de clasificación postural de SitRight.

## Flujo

| Notebook | Qué hace |
|---|---|
| `01-exploration.ipynb` | EDA del dataset crudo — distribuciones, separabilidad, calidad |
| `02-preprocessing.ipynb` | Filtra 3 clases, selecciona features del sensor dorsal, guarda CSV procesado |
| `03-training.ipynb` | Entrena Random Forest, evalúa métricas, verifica criterios de aceptación |
| `04-export.ipynb` | Entrena modelo final sobre dataset completo, guarda `src/models/rf_v1.pkl` |

Ejecutar en orden. Solo correr `04-export.ipynb` si `03-training.ipynb` pasó todos los criterios.

## Dataset

`pkmandke/Human-Posture-Dataset` — descargado en `data/raw/posture_dataset.csv` (no versionado en git).

Para re-descargar:

```powershell
Invoke-WebRequest -Uri "https://drive.google.com/uc?export=download&id=1GtNRi9CgHOlGpsa-Dlp6uzAqXUVdEh54" -OutFile "data/raw/posture_dataset.csv"
```

## Dependencias

```bash
pip install -r requirements.txt
jupyter lab
```

## Criterios de aceptación del modelo (ADR-004)

| Métrica | Mínimo |
|---|---|
| Accuracy en test set | ≥ 80% |
| Tiempo de inferencia | < 100 ms |
| Tamaño del `.pkl` | < 50 MB |
