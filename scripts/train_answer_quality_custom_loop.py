"""Train TensorFlow answer-quality model using a full custom tf.GradientTape loop.

This script is for the AI Side Quest:
- custom training + evaluation loop with tf.GradientTape
- TensorBoard logging
- model export to .keras

Run:
    python scripts/train_answer_quality_custom_loop.py

View TensorBoard:
    tensorboard --logdir models/logs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds_assets import answer_quality_train_path, answer_quality_val_path, answer_quality_test_path
from model_builder import (
    ANSWER_QUALITY_META_PATH,
    evaluate_split_datasets,
    train_answer_quality_model_custom_loop,
)

train_path = str(answer_quality_train_path())
val_path = str(answer_quality_val_path())
test_path = str(answer_quality_test_path())

print("=== Road2Work Custom GradientTape Training ===")
print("Train:", train_path)
print("Val  :", val_path)
print("Test :", test_path)

model = train_answer_quality_model_custom_loop(
    train_csv_path=train_path,
    val_csv_path=val_path,
    test_csv_path=test_path,
    epochs=25,
    batch_size=32,
    learning_rate=1e-3,
    min_accuracy=0.85,
    max_mae=0.02,
)

if model is None:
    raise SystemExit("Training gagal. Cek dataset dan dependency TensorFlow.")

print("\n=== Evaluation on DS split datasets ===")
report = evaluate_split_datasets(include_predictions=False)
for split, result in report.items():
    print(f"\n[{split.upper()}]")
    if "error" in result:
        print(result)
    else:
        print("samples:", result.get("n_samples"))
        print("accuracy:", round(result.get("answer_quality_accuracy", 0), 4))
        print("mae:", round(result.get("readiness_score_mae", 0), 4))
        print("accuracy pass:", result.get("target_accuracy_pass"))
        print("mae pass:", result.get("target_mae_pass"))

try:
    meta = json.loads(Path(ANSWER_QUALITY_META_PATH).read_text(encoding="utf-8"))
    print("\nTensorBoard log dir:", meta.get("tensorboard_log_dir"))
    print("Run: tensorboard --logdir models/logs")
except Exception:
    pass
