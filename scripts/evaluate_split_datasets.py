"""Evaluate saved TensorFlow answer-quality model on train/val/test split datasets."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_builder import evaluate_split_datasets

report = evaluate_split_datasets(include_predictions=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
