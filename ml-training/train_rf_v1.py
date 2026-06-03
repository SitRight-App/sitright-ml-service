"""
Script de entrenamiento RF v1 — equivalente a ejecutar los notebooks 02, 03 y 04 en secuencia.
Uso: python train_rf_v1.py
Genera: ../src/models/rf_v1.pkl
"""
import time
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sk_shuffle

BASE = Path(__file__).parent
RAW_PATH = BASE / "data/raw/posture_dataset.csv"
PROCESSED_DIR = BASE / "data/processed"
OUTPUT_PATH = BASE / "../src/models/rf_v1.pkl"

CLASS_NAMES = {0: "adequate", 1: "forward_slouch", 2: "excessive_recline"}
# dataset → sitright. Mapeo de labels del Human-Posture-Dataset a las 3 clases
# del ADR-004: sitting(0)→adequate, forward bending(2)→forward_slouch,
# backward bending(3)→excessive_recline. Se descartan standing(1), sleeping(4),
# running(5) por estar fuera del alcance postural sedentario.
LABEL_MAP = {0.0: 0, 2.0: 1, 3.0: 2}

# ── 1. Carga ──────────────────────────────────────────────────────────────────
print("1. Cargando dataset...")
df = pd.read_csv(RAW_PATH, index_col=0)
print(f"   Shape original: {df.shape}")

# ── 2. Filtrado a 3 clases ────────────────────────────────────────────────────
print("2. Filtrando a 3 clases...")
df_f = df[df["label"].isin(LABEL_MAP.keys())].copy()
df_f["label"] = df_f["label"].map(LABEL_MAP)
for cls_id, cls_name in CLASS_NAMES.items():
    n = (df_f["label"] == cls_id).sum()
    print(f"   {cls_name}: {n} muestras")

# ── 3. Features y shuffle ─────────────────────────────────────────────────────
print("3. Preparando features...")
X = df_f[["Ax1", "Ay1", "Az1"]].values
y = df_f["label"].values
X, y = sk_shuffle(X, y, random_state=42)

# ── 4. Evaluación (train/test split 80/20) ────────────────────────────────────
print("4. Evaluando modelo (split 80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
clf_eval = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
t0 = time.perf_counter()
clf_eval.fit(X_train, y_train)
t_train = time.perf_counter() - t0

y_pred = clf_eval.predict(X_test)
acc = accuracy_score(y_test, y_pred)
target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]

print(f"\n   Entrenamiento: {t_train:.2f}s")
print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
print(classification_report(y_test, y_pred, target_names=target_names))

# ── 5. Inferencia y tamaño ────────────────────────────────────────────────────
times = []
for _ in range(500):
    t0 = time.perf_counter()
    clf_eval.predict_proba([[0.1, 9.7, 0.3]])
    times.append(time.perf_counter() - t0)
median_ms = np.median(times) * 1000

with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
    tmp_name = tmp.name
joblib.dump(clf_eval, tmp_name)
size_mb = Path(tmp_name).stat().st_size / (1024 * 1024)
Path(tmp_name).unlink()

print(f"   Inferencia mediana: {median_ms:.3f} ms")
print(f"   Tamaño estimado: {size_mb:.2f} MB")

# ── 6. Criterios de aceptación ────────────────────────────────────────────────
print("\n" + "=" * 45)
print("  Criterios de aceptación — ADR-004")
print("=" * 45)
ok, fail = "OK", "FAIL"
print(f"  Accuracy >= 80%   : {ok if acc >= 0.80 else fail}  ({acc*100:.2f}%)")
print(f"  Inferencia <100ms : {ok if median_ms < 100 else fail}  ({median_ms:.2f}ms)")
print(f"  Modelo <50MB      : {ok if size_mb < 50 else fail}  ({size_mb:.2f}MB)")
print("=" * 45)

if not (acc >= 0.80 and median_ms < 100 and size_mb < 50):
    print("\nALGUN CRITERIO FALLO. Revisar antes de exportar.")
    raise SystemExit(1)

# ── 7. Modelo final sobre dataset completo ────────────────────────────────────
print("\n5. Entrenando modelo final sobre dataset completo...")
clf_final = RandomForestClassifier(
    n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
)
clf_final.fit(X, y)

bundle = {
    "model": clf_final,
    "label_map": CLASS_NAMES,
    "features": ["Ax1", "Ay1", "Az1"],
    "model_version": "rf_v1",
    "feature_description": "dorsal sensor accelerometer (ax, ay, az)",
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, OUTPUT_PATH)
real_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
print(f"   Guardado: {OUTPUT_PATH}")
print(f"   Tamaño real: {real_size_mb:.2f} MB")

# ── 8. Smoke test ─────────────────────────────────────────────────────────────
print("\n6. Smoke test...")
loaded = joblib.load(OUTPUT_PATH)
# Valores en g (el dataset y el firmware reportan en g, no en m/s²).
# Referencias: medias de cada clase en el dataset filtrado.
test_cases = [
    ([0.06, -0.12, 0.95], "adequate"),
    ([-0.92, 0.00, 0.05], "forward_slouch"),
    ([-0.85, 0.25, -0.10], "excessive_recline"),
]
for features, expected in test_cases:
    proba = loaded["model"].predict_proba([features])[0]
    idx = int(proba.argmax())
    cls = loaded["label_map"][idx]
    conf = proba[idx]
    status = "OK" if cls == expected else f"WARN (expected {expected})"
    print(f"   {features} -> {cls} ({conf:.3f}) {status}")

print("\nModelo rf_v1.pkl generado y verificado.")
