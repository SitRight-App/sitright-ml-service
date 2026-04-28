#!/usr/bin/env bash
# Build script para Render — instala dependencias, descarga dataset y entrena el modelo.
set -e

echo "=== 1. Instalando dependencias ==="
pip install -r requirements.txt
pip install pandas  # solo necesario para el entrenamiento, no para inferencia

echo "=== 2. Descargando dataset ==="
mkdir -p ml-training/data/raw src/models
python -c "
import urllib.request, sys
url = 'https://drive.google.com/uc?export=download&id=1GtNRi9CgHOlGpsa-Dlp6uzAqXUVdEh54'
print('Descargando pkmandke/Human-Posture-Dataset...')
urllib.request.urlretrieve(url, 'ml-training/data/raw/posture_dataset.csv')
import os
size_mb = os.path.getsize('ml-training/data/raw/posture_dataset.csv') / 1024 / 1024
print(f'Dataset descargado: {size_mb:.2f} MB')
"

echo "=== 3. Entrenando modelo RF v1 ==="
python ml-training/train_rf_v1.py

echo "=== Build completado ==="
ls -lh src/models/
