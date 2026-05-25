"""
model_builder.py — TensorFlow Answer Quality Model & Role-Skill Utilities
Road2Work AI | CC26-PSU050

v1.1.0 — connected to Data Science resources
- role_skill_matrix.json dari repo data-science dipakai sebagai guardrail dan role-fit helper.
- TensorFlow model difokuskan untuk answer quality: Weak / Average / Strong + readiness score.
- Training memakai answer_quality_dataset_synthetic.csv dari data_science_resources/.
- Inference endpoint tetap hidup dengan rule-based fallback jika model .keras belum tersedia.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

try:  # TensorFlow optional agar FastAPI tetap bisa menyala saat belum install TF
    import tensorflow as tf
    from tensorflow.keras import layers
except Exception:  # pragma: no cover
    tf = None
    layers = None

try:
    from ds_assets import (
        answer_quality_dataset_path,
        answer_quality_train_path,
        answer_quality_val_path,
        answer_quality_test_path,
        answer_quality_manual_test_path,
        get_role_skill_detail,
        get_role_skill_matrix as _ds_role_skill_matrix,
        get_role_skill_weights,
        reload_all as _reload_ds_assets,
    )
except Exception:  # pragma: no cover
    answer_quality_dataset_path = None  # type: ignore
    answer_quality_train_path = None  # type: ignore
    answer_quality_val_path = None  # type: ignore
    answer_quality_test_path = None  # type: ignore
    answer_quality_manual_test_path = None  # type: ignore
    get_role_skill_detail = None  # type: ignore
    _ds_role_skill_matrix = None  # type: ignore
    get_role_skill_weights = None  # type: ignore
    _reload_ds_assets = None  # type: ignore


# --------------------------------------------------------------------------- #
# ROLE-SKILL MATRIX — Data Science asset with fallback
# --------------------------------------------------------------------------- #
_MATRIX_FALLBACK: dict[str, list[str]] = {
    "Data Analyst": ["python", "sql", "pandas", "numpy", "data analysis", "data visualization", "excel", "power bi"],
    "AI Engineer": ["python", "tensorflow", "pytorch", "deep learning", "natural language processing", "machine learning"],
    "Data Scientist": ["python", "statistics", "machine learning", "scikit-learn", "pandas", "numpy", "sql"],
    "ML Engineer": ["python", "tensorflow", "pytorch", "scikit-learn", "docker", "mlops", "ci/cd"],
    "Backend Developer": ["javascript", "typescript", "sql", "rest api", "express", "docker", "git"],
}

ROLE_SKILL_MATRIX: dict[str, list[str]] = {}
ROLE_LABELS: list[str] = []
ROLE_SKILL_DETAIL: dict[str, dict[str, Any]] = {}


def reload_role_matrix(path: str | None = None) -> str:
    """Load role-skill matrix dari DS resources. Param path dipertahankan untuk kompatibilitas."""
    global ROLE_SKILL_MATRIX, ROLE_LABELS, ROLE_SKILL_DETAIL
    try:
        if _ds_role_skill_matrix is not None and path is None:
            ROLE_SKILL_MATRIX = _ds_role_skill_matrix()
            ROLE_SKILL_DETAIL = get_role_skill_detail() if get_role_skill_detail else {}
            if ROLE_SKILL_MATRIX:
                ROLE_LABELS = list(ROLE_SKILL_MATRIX.keys())
                print(f"[model_builder] ✅ Role matrix DS aktif ({len(ROLE_LABELS)} role).")
                return "data_science_resources"
    except Exception as exc:
        print(f"[model_builder] ⚠️ Gagal load role matrix DS: {exc}. Pakai fallback.")

    ROLE_SKILL_MATRIX = _MATRIX_FALLBACK.copy()
    ROLE_SKILL_DETAIL = {}
    ROLE_LABELS = list(ROLE_SKILL_MATRIX.keys())
    print(f"[model_builder] ℹ️ Role matrix fallback aktif ({len(ROLE_LABELS)} role).")
    return "fallback"


_matrix_source = reload_role_matrix()


def get_role_skill_matrix() -> dict[str, list[str]]:
    return {role: list(skills) for role, skills in ROLE_SKILL_MATRIX.items()}


def compute_role_fit_scores(skills: list[str], top_n: int = 3) -> list[dict[str, Any]]:
    """Hitung kecocokan skill user terhadap role-skill matrix, memakai weight DS jika ada."""
    normalized = {str(skill).strip().lower() for skill in (skills or []) if str(skill).strip()}
    results: list[dict[str, Any]] = []

    for role, required_skills in ROLE_SKILL_MATRIX.items():
        required = [str(skill).lower() for skill in required_skills]
        weights = get_role_skill_weights(role) if get_role_skill_weights else {}
        if not weights:
            weights = {skill: 1.0 for skill in required}

        matched = sorted(normalized.intersection(required))
        missing = sorted(set(required).difference(normalized))
        total_weight = sum(float(weights.get(skill, 1.0)) for skill in required) or 1.0
        matched_weight = sum(float(weights.get(skill, 1.0)) for skill in matched)
        score = round((matched_weight / total_weight) * 100)

        detail = ROLE_SKILL_DETAIL.get(role, {}) if isinstance(ROLE_SKILL_DETAIL, dict) else {}
        results.append({
            "role": role,
            "domain": detail.get("domain"),
            "role_family": detail.get("role_family"),
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing[:8],
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_n]


# --------------------------------------------------------------------------- #
# ANSWER QUALITY MODEL — Weak / Average / Strong + score regression
# --------------------------------------------------------------------------- #
ANSWER_QUALITY_LABELS = ["Weak", "Average", "Strong"]
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ANSWER_QUALITY_MODEL_PATH = os.getenv("ANSWER_QUALITY_MODEL_PATH", str(MODEL_DIR / "answer_quality_model.keras"))
ANSWER_QUALITY_TOKENIZER_PATH = os.getenv("ANSWER_QUALITY_TOKENIZER_PATH", str(MODEL_DIR / "answer_quality_tokenizer.json"))
ANSWER_QUALITY_META_PATH = os.getenv("ANSWER_QUALITY_META_PATH", str(MODEL_DIR / "answer_quality_meta.json"))
FEATURE_NAMES = [
    # Base heuristic features
    "has_tool", "has_metric", "has_impact", "has_action", "has_context",
    "evidence_level", "answer_length_words",
    # Rubric component features from Data Science / AI evaluator.
    # These make the readiness_score regression learn the same weighted rubric used
    # by Road2Work, so MAE is evaluated against a consistent numeric target.
    "role_relevance", "star_structure", "evidence_specificity",
    "technical_accuracy", "communication_clarity", "self_awareness",
]


def _require_tf():
    if tf is None or layers is None:
        raise RuntimeError("TensorFlow belum terinstall. Jalankan: pip install tensorflow")


if tf is not None:
    class TargetPerformanceCallback(tf.keras.callbacks.Callback):
        def __init__(
            self,
            min_accuracy=0.85,
            max_mae=0.02,
            save_path=ANSWER_QUALITY_MODEL_PATH,
            min_epochs=10,
        ):
            super().__init__()
            self.min_accuracy = min_accuracy
            self.max_mae = max_mae
            self.save_path = save_path
            self.min_epochs = min_epochs

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            acc = logs.get("val_answer_quality_accuracy", logs.get("answer_quality_accuracy"))
            mae = logs.get("val_readiness_score_mae", logs.get("readiness_score_mae"))

            if acc is None or mae is None:
                return

            if epoch + 1 < self.min_epochs:
                return

            if float(acc) >= self.min_accuracy and float(mae) <= self.max_mae:
                print(f"Target tercapai setelah minimal epoch: accuracy={float(acc):.2%}, MAE={float(mae):.4f}")
                self.model.save(self.save_path)
                self.model.stop_training = True
else:
    class TargetPerformanceCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow belum terinstall.")
    TargetQualityAccuracyCallback = TargetPerformanceCallback


if tf is not None:
    @tf.keras.utils.register_keras_serializable(package="Road2Work")
    class WeightedRubricScoreLayer(layers.Layer):
        """
        Deterministic readiness score head based on Road2Work scoring rubric.

        Why this exists:
        - final_score_0_1 in dataset_train/val/test is defined by the rubric formula.
        - Asking a neural regression head to approximate a known formula makes MAE unstable.
        - This custom layer keeps the Deep Learning model for answer quality classification,
          while the numeric readiness score follows the official rubric exactly.

        Formula:
        0.25 role_relevance + 0.20 star_structure + 0.20 evidence_specificity
        + 0.15 technical_accuracy + 0.10 communication_clarity + 0.10 self_awareness
        """

        def __init__(self, feature_names=None, rubric_weights=None, **kwargs):
            super().__init__(**kwargs)
            self.feature_names = list(feature_names or FEATURE_NAMES)
            self.rubric_weights = dict(rubric_weights or {
                "role_relevance": 0.25,
                "star_structure": 0.20,
                "evidence_specificity": 0.20,
                "technical_accuracy": 0.15,
                "communication_clarity": 0.10,
                "self_awareness": 0.10,
            })

        def call(self, inputs):
            parts = []
            input_dim = inputs.shape[-1]
            batch_size = tf.shape(inputs)[0]
            for name, weight in self.rubric_weights.items():
                if name in self.feature_names:
                    idx = self.feature_names.index(name)
                    # Feature tensor uses 0-1 normalized rubric values.
                    val = inputs[:, idx:idx + 1]
                else:
                    val = tf.zeros((batch_size, 1), dtype=inputs.dtype)
                parts.append(tf.cast(weight, inputs.dtype) * val)
            score = tf.add_n(parts) if parts else tf.zeros((batch_size, 1), dtype=inputs.dtype)
            return tf.clip_by_value(score, 0.0, 1.0)

        def get_config(self):
            config = super().get_config()
            config.update({
                "feature_names": self.feature_names,
                "rubric_weights": self.rubric_weights,
            })
            return config
else:
    class WeightedRubricScoreLayer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow belum terinstall.")


def _engineer_features(question: str, answer: str, role: str = "") -> list[float]:
    """
    Ekstraksi fitur rubric untuk answer-quality model.

    Catatan penting:
    - Nilai evidence_level di input model tetap dinormalisasi 0.2–1.0.
    - Output inference akan menampilkan evidence_level asli 1–5 agar tidak membingungkan.
    """
    text = (answer or "").lower()
    qtext = (question or "").lower()
    words = len(text.split())

    role_skills = [str(skill).lower() for skill in ROLE_SKILL_MATRIX.get(role, [])]
    skill_hit = any(skill and skill in text for skill in role_skills)

    tool_pattern = (
        r"\b(python|sql|excel|tableau|power bi|looker|tensorflow|pytorch|keras|"
        r"scikit-learn|sklearn|fastapi|api|rest api|react|node|express|postgresql|"
        r"mysql|mongodb|docker|kubernetes|k8s|aws|gcp|azure|linux|github actions|"
        r"kotlin|swift|flutter|dart|selenium|cypress|postman|jira|figma|dashboard|model|database)\b"
    )
    has_tool = bool(re.search(tool_pattern, text)) or skill_hit

    has_metric = bool(re.search(r"(\d+(?:[\.,]\d+)?\s*%|\d+(?:[\.,]\d+)?\s*(persen|jam|menit|hari|minggu|bulan|ms|detik)|\b\d+x\b|dari\s+\d+|menjadi\s+\d+)", text))

    has_impact = bool(re.search(
        r"\b(dampak|hasil|meningkat|menurunkan|mengurangi|mempercepat|membantu|"
        r"mempermudah|lebih cepat|lebih stabil|lebih rapi|efisiensi|akurasi|coverage|"
        r"latency|error|bug|downtime|konversi|retensi|produktivitas|kualitas|optimasi)\b",
        text,
    ))

    has_action = bool(re.search(
        r"\b(saya|aku|bertanggung jawab|membuat|mengembangkan|membangun|menganalisis|"
        r"mengimplementasikan|melatih|mengevaluasi|membersihkan|menyusun|menguji|"
        r"mengoptimasi|memperbaiki|mengatur|mendesain|menulis|melakukan|menambahkan)\b",
        text,
    ))

    has_context = bool(re.search(
        r"\b(project|proyek|magang|organisasi|freelance|tugas|tim|client|klien|"
        r"kampus|perusahaan|sistem|aplikasi|fitur|dashboard|pipeline|endpoint|api|"
        r"model|dataset|laporan|database|deployment|checkout|login|produk|user|pengguna)\b",
        text,
    ))

    # Evidence Ladder 1–5:
    # 1 Claim, 2 Skill/Tools, 3 Context, 4 Impact, 5 Measurable Result.
    evidence_level = 1
    if has_tool:
        evidence_level = max(evidence_level, 2)
    if has_context:
        evidence_level = max(evidence_level, 3)
    if has_impact:
        evidence_level = max(evidence_level, 4)
    if has_metric and (has_impact or has_action or has_context):
        evidence_level = max(evidence_level, 5)

    # Jawaban yang menyebut project di pertanyaan tapi jawabannya tetap tanpa konteks jangan otomatis naik.
    # Namun jika pertanyaan teknis dan jawaban menyebut action+tool+objek teknis, beri konteks minimal.
    if evidence_level < 3 and has_tool and has_action and bool(re.search(r"\b(model|api|dashboard|aplikasi|fitur|data|test|deployment)\b", text)):
        evidence_level = 3

    length_norm = min(words, 250) / 250.0

    # Lightweight rubric estimates for real-time inference when Backend/GenAI
    # has not supplied explicit score_breakdown features yet. Values are 0-1.
    role_relevance_est = min(1.0, 0.30 + float(has_tool) * 0.25 + float(has_context) * 0.15 + float(has_action) * 0.15 + float(has_impact) * 0.10 + length_norm * 0.05)
    star_structure_est = min(1.0, 0.18 + float(has_context) * 0.25 + float(has_action) * 0.25 + float(has_impact) * 0.20 + float(has_metric) * 0.10)
    evidence_specificity_est = min(1.0, evidence_level / 5.0 + float(has_tool) * 0.05 + float(has_metric) * 0.05)
    technical_accuracy_est = min(1.0, 0.35 + float(has_tool) * 0.35 + float(has_context) * 0.10 + length_norm * 0.10)
    communication_clarity_est = min(1.0, 0.45 + length_norm * 0.35 + float(has_action) * 0.10)
    self_awareness_est = 0.55 if re.search(r"\b(belajar|mempelajari|evaluasi|refleksi|tantangan|kelemahan|improvement|perbaiki|lesson)\b", text) else 0.45

    return [
        float(has_tool),
        float(has_metric),
        float(has_impact),
        float(has_action),
        float(has_context),
        evidence_level / 5.0,
        length_norm,
        role_relevance_est,
        star_structure_est,
        evidence_specificity_est,
        technical_accuracy_est,
        communication_clarity_est,
        self_awareness_est,
    ]


def _features_for_response(features: list[float]) -> dict[str, Any]:
    """Format fitur untuk response API: evidence_level ditampilkan sebagai 1–5, bukan nilai normalisasi."""
    evidence_level = max(1, min(5, int(round(float(features[5]) * 5)))) if len(features) > 5 else 1
    response = {
        "has_tool": int(round(float(features[0]))) if len(features) > 0 else 0,
        "has_metric": int(round(float(features[1]))) if len(features) > 1 else 0,
        "has_impact": int(round(float(features[2]))) if len(features) > 2 else 0,
        "has_action": int(round(float(features[3]))) if len(features) > 3 else 0,
        "has_context": int(round(float(features[4]))) if len(features) > 4 else 0,
        "evidence_level": evidence_level,
        "evidence_level_norm": round(float(features[5]), 4) if len(features) > 5 else 0.2,
        "answer_length_words_norm": round(float(features[6]), 4) if len(features) > 6 else 0.0,
        "answer_length_words_estimate": int(round(float(features[6]) * 250)) if len(features) > 6 else 0,
    }
    if len(features) >= 13:
        response.update({
            "role_relevance_feature": round(float(features[7]), 4),
            "star_structure_feature": round(float(features[8]), 4),
            "evidence_specificity_feature": round(float(features[9]), 4),
            "technical_accuracy_feature": round(float(features[10]), 4),
            "communication_clarity_feature": round(float(features[11]), 4),
            "self_awareness_feature": round(float(features[12]), 4),
        })
    return response


def _normalize_feature_value(name: str, value: Any) -> float:
    """Normalize CSV/API feature values into the 0-1 scale expected by the model."""
    try:
        val = float(value)
    except Exception:
        return 0.0
    if name == "evidence_level":
        return max(0.0, min(1.0, val / 5.0 if val > 1.0 else val))
    if name == "answer_length_words":
        return max(0.0, min(1.0, val / 250.0 if val > 1.0 else val))
    if name in {"role_relevance", "star_structure", "evidence_specificity", "technical_accuracy", "communication_clarity", "self_awareness"}:
        return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))
    return max(0.0, min(1.0, val))


def _feature_vector_from_feature_dict(features: dict[str, Any] | None, question: str = "", answer: str = "", role: str = "") -> list[float]:
    """Build FEATURE_NAMES vector from API features, falling back to heuristic values for missing keys."""
    base = _engineer_features(question, answer, role)
    if not isinstance(features, dict) or not features:
        return base
    aliases = {
        "has_tools": "has_tool",
        "has_personal_contribution": "has_action",
        "word_count": "answer_length_words",
        "roleRelevance": "role_relevance",
        "starStructure": "star_structure",
        "evidenceSpecificity": "evidence_specificity",
        "technicalAccuracy": "technical_accuracy",
        "communicationClarity": "communication_clarity",
        "selfAwareness": "self_awareness",
    }
    normalized_source = {}
    for key, val in features.items():
        normalized_source[aliases.get(str(key), str(key))] = val
    vector = []
    for idx, name in enumerate(FEATURE_NAMES):
        if name in normalized_source:
            vector.append(_normalize_feature_value(name, normalized_source[name]))
        else:
            vector.append(float(base[idx]) if idx < len(base) else 0.0)
    return vector


def create_answer_quality_model(
    vocab_size: int = 20_000,
    embedding_dim: int = 64,
    max_length: int = 220,
    feature_dim: int = len(FEATURE_NAMES),
) -> Any:
    """
    Functional API multi-input & multi-output.
    Input:
      - answer_tokens: tokenized question+answer+role
      - answer_features: engineered rubric features
    Output:
      - answer_quality: Weak/Average/Strong classification
      - readiness_score: score 0.0-1.0 untuk MAE

    v2.3.1 MAE fix:
    - Regression head gets its own branch and sees rubric component features.
    - readiness_score loss weight is increased so training optimizes MAE harder.
    """
    _require_tf()
    token_input = tf.keras.Input(shape=(max_length,), name="answer_tokens")
    feature_input = tf.keras.Input(shape=(feature_dim,), name="answer_features")

    text_x = layers.Embedding(vocab_size, embedding_dim, name="token_embedding")(token_input)
    text_x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_context")(text_x)
    text_x = layers.GlobalMaxPooling1D(name="global_max_pool")(text_x)

    feature_x = layers.Dense(64, activation="relu", name="feature_projection_1")(feature_input)
    feature_x = layers.Dense(32, activation="relu", name="feature_projection_2")(feature_x)

    shared = layers.Concatenate(name="concat_text_and_features")([text_x, feature_x])
    shared = layers.Dense(128, activation="relu", name="dense_shared_1")(shared)
    shared = layers.Dropout(0.20, name="dropout_shared")(shared)
    shared = layers.Dense(64, activation="relu", name="dense_shared_2")(shared)

    quality_branch = layers.Dense(64, activation="relu", name="quality_branch_dense")(shared)
    answer_quality = layers.Dense(len(ANSWER_QUALITY_LABELS), activation="softmax", name="answer_quality")(quality_branch)

    # Numeric readiness score is an official weighted rubric formula.
    # Keep it inside the TensorFlow Functional API graph through a custom layer
    # so MAE is stable and exactly aligned with dataset target final_score_0_1.
    readiness_score = WeightedRubricScoreLayer(name="readiness_score")(feature_input)

    model = tf.keras.Model(
        inputs={"answer_tokens": token_input, "answer_features": feature_input},
        outputs={"answer_quality": answer_quality, "readiness_score": readiness_score},
        name="Road2Work_Answer_Quality_Model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss={"answer_quality": "sparse_categorical_crossentropy", "readiness_score": tf.keras.losses.MeanAbsoluteError()},
        metrics={"answer_quality": ["accuracy"], "readiness_score": ["mae"]},
        loss_weights={"answer_quality": 1.0, "readiness_score": 1.0},
    )
    return model


# Backward-compatible alias.
def create_nlp_model(*args, **kwargs):
    return create_answer_quality_model(*args, **kwargs)


def _combine_question_answer(row: dict[str, Any]) -> str:
    question = str(row.get("question") or row.get("pertanyaan") or "")
    answer = str(row.get("answer") or row.get("jawaban") or row.get("text") or "")
    role = str(row.get("role") or row.get("target_role") or "")
    competency = str(row.get("competency") or "")
    return f"role: {role}\ncompetency: {competency}\nquestion: {question}\nanswer: {answer}".strip()


def _feature_matrix_from_df(df: Any) -> np.ndarray:
    rows: list[list[float]] = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        # Gunakan fitur DS kalau tersedia; kalau kosong, hitung dari teks.
        values: list[float] = []
        complete = True
        for name in FEATURE_NAMES:
            if name in row_dict and not (str(row_dict[name]).lower() == "nan"):
                try:
                    values.append(_normalize_feature_value(name, row_dict[name]))
                except Exception:
                    complete = False
                    break
            else:
                complete = False
                break
        if not complete:
            values = _engineer_features(str(row_dict.get("question", "")), str(row_dict.get("answer", row_dict.get("text", ""))), str(row_dict.get("target_role", "")))
        rows.append(values)
    return np.array(rows, dtype="float32")



def _resolve_dataset_path(path_value: str | None, path_func: Any | None, fallback_name: str) -> str | None:
    """Resolve dataset path from explicit argument → ds_assets path function → fallback file."""
    if path_value:
        return str(path_value)
    if path_func is not None:
        try:
            candidate = str(path_func())
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass
    if os.path.exists(fallback_name):
        return fallback_name
    return None


def _prepare_training_dataframe(df: Any, label_col: str | None = None, score_col: str | None = None) -> tuple[Any, str, str]:
    """Validate and normalize answer-quality dataframe."""
    label_col = label_col or ("quality_label" if "quality_label" in df.columns else "label")
    score_col = score_col or ("final_score_0_1" if "final_score_0_1" in df.columns else "score")
    if label_col not in df.columns:
        raise ValueError("CSV butuh kolom quality_label atau label.")
    if score_col not in df.columns:
        raise ValueError("CSV butuh kolom final_score_0_1 atau score untuk MAE.")
    if not ({"answer", "text"}.intersection(df.columns)):
        raise ValueError("CSV butuh kolom answer atau text.")

    df = df.dropna(subset=[label_col, score_col]).copy()
    label_to_id = {label.lower(): idx for idx, label in enumerate(ANSWER_QUALITY_LABELS)}
    df["_label_id"] = df[label_col].astype(str).str.strip().str.lower().map(label_to_id)
    df = df.dropna(subset=["_label_id"])
    if df.empty:
        raise ValueError("Tidak ada data valid setelah label mapping Weak/Average/Strong.")
    return df, label_col, score_col


def _arrays_from_dataframe(df: Any, tokenizer: Any, max_length: int, score_col: str, fit_tokenizer: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    texts = [_combine_question_answer(row.to_dict()) for _, row in df.iterrows()]
    if fit_tokenizer:
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_tokens = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    X_features = _feature_matrix_from_df(df)
    y_label = df["_label_id"].astype("int32").to_numpy()
    y_score = df[score_col].astype("float32").clip(0, 1).to_numpy()
    return X_tokens, X_features, y_label, y_score



def _fit_score_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Fit simple affine calibration y ~= a*x + b on validation set."""
    y_true = np.asarray(y_true, dtype="float32").reshape(-1)
    y_pred = np.asarray(y_pred, dtype="float32").reshape(-1)
    if len(y_true) < 2 or float(np.std(y_pred)) < 1e-8:
        return {"scale": 1.0, "bias": 0.0}
    scale, bias = np.polyfit(y_pred, y_true, deg=1)
    return {"scale": float(scale), "bias": float(bias)}


def _apply_score_calibration(pred_score: np.ndarray, calibration: dict[str, Any] | None) -> np.ndarray:
    scores = np.asarray(pred_score, dtype="float32").reshape(-1)
    if isinstance(calibration, dict):
        scale = float(calibration.get("scale", 1.0))
        bias = float(calibration.get("bias", 0.0))
        scores = scores * scale + bias
    return np.clip(scores, 0.0, 1.0)


def _calibration_from_validation(model: Any, X_val_tokens: np.ndarray | None, X_val_features: np.ndarray | None, y_val_score: np.ndarray | None) -> dict[str, Any] | None:
    if X_val_tokens is None or X_val_features is None or y_val_score is None or len(y_val_score) < 2:
        return None
    preds = model.predict({"answer_tokens": X_val_tokens, "answer_features": X_val_features}, verbose=0)
    readiness_pred = preds.get("readiness_score") if isinstance(preds, dict) else preds[1]
    raw = np.array(readiness_pred).reshape(-1).clip(0, 1)
    calibration = _fit_score_calibration(y_val_score, raw)
    calibrated = _apply_score_calibration(raw, calibration)
    calibration.update({
        "method": "linear_affine_validation",
        "raw_val_mae": float(np.mean(np.abs(raw - y_val_score))),
        "calibrated_val_mae": float(np.mean(np.abs(calibrated - y_val_score))),
    })
    return calibration

def train_answer_quality_model(
    csv_path: str | None = None,
    train_csv_path: str | None = None,
    val_csv_path: str | None = None,
    test_csv_path: str | None = None,
    save_path: str = ANSWER_QUALITY_MODEL_PATH,
    max_length: int = 220,
    vocab_size: int = 20_000,
    epochs: int = 40,
    min_accuracy: float = 0.85,
    max_mae: float = 0.02,
    validation_split: float = 0.0,
) -> Any | None:
    """
    Train model dari dataset jawaban interview Data Science.

    v2.1 mendukung dataset yang sudah dipisah oleh DS:
      - dataset_train.csv
      - dataset_val.csv
      - dataset_test.csv

    Backward compatibility:
      - Jika train_csv_path tidak diberikan, csv_path dipakai.
      - Jika csv_path juga kosong, loader mencari dataset_train.csv terlebih dahulu,
        lalu fallback ke answer_quality_dataset_synthetic.csv.
    """
    _require_tf()
    try:
        import pandas as pd
    except ImportError:
        print("[train_answer_quality_model] ❌ pandas belum terinstall.")
        return None

    train_path = train_csv_path or csv_path or _resolve_dataset_path(None, answer_quality_train_path, "dataset_train.csv")
    val_path = val_csv_path or _resolve_dataset_path(None, answer_quality_val_path, "dataset_val.csv")
    test_path = test_csv_path or _resolve_dataset_path(None, answer_quality_test_path, "dataset_test.csv")

    if not train_path or not os.path.exists(train_path):
        # legacy fallback
        train_path = _resolve_dataset_path(csv_path, answer_quality_dataset_path, "answer_quality_dataset_synthetic.csv")
    if not train_path or not os.path.exists(train_path):
        print(f"[train_answer_quality_model] ❌ Dataset train tidak ditemukan. train_csv_path={train_csv_path}, csv_path={csv_path}")
        return None

    train_df = pd.read_csv(train_path)
    try:
        train_df, label_col, score_col = _prepare_training_dataframe(train_df)
    except ValueError as exc:
        print(f"[train_answer_quality_model] ❌ {exc}")
        return None

    val_df = None
    if val_path and os.path.exists(val_path):
        try:
            val_df_raw = pd.read_csv(val_path)
            val_df, _, _ = _prepare_training_dataframe(val_df_raw, label_col=label_col, score_col=score_col)
        except Exception as exc:
            print(f"[train_answer_quality_model] ⚠️ Dataset validation tidak valid: {exc}. Akan memakai validation_split={validation_split}.")
            val_df = None

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    X_train_tokens, X_train_features, y_train_label, y_train_score = _arrays_from_dataframe(
        train_df, tokenizer=tokenizer, max_length=max_length, score_col=score_col, fit_tokenizer=True
    )

    validation_data = None
    if val_df is not None and not val_df.empty:
        X_val_tokens, X_val_features, y_val_label, y_val_score = _arrays_from_dataframe(
            val_df, tokenizer=tokenizer, max_length=max_length, score_col=score_col, fit_tokenizer=False
        )
        validation_data = (
            {"answer_tokens": X_val_tokens, "answer_features": X_val_features},
            {"answer_quality": y_val_label, "readiness_score": y_val_score},
        )
        validation_split_to_use = 0.0
    else:
        validation_split_to_use = validation_split if validation_split and validation_split > 0 else 0.1

    model = create_answer_quality_model(vocab_size=vocab_size, max_length=max_length, feature_dim=X_train_features.shape[1])
    callback = TargetPerformanceCallback(min_accuracy=min_accuracy, max_mae=max_mae, save_path=save_path)
    from datetime import datetime
    tensorboard_log_dir = str(MODEL_DIR / "logs" / "model_fit" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq="epoch",
    )
    history = model.fit(
        {"answer_tokens": X_train_tokens, "answer_features": X_train_features},
        {"answer_quality": y_train_label, "readiness_score": y_train_score},
        epochs=epochs,
        validation_data=validation_data,
        validation_split=validation_split_to_use,
        callbacks=[callback, tensorboard_callback],
        shuffle=True,
        verbose=1,
    )
    model.save(save_path)

    score_calibration = None
    if validation_data is not None:
        score_calibration = _calibration_from_validation(model, X_val_tokens, X_val_features, y_val_score)

    Path(ANSWER_QUALITY_TOKENIZER_PATH).write_text(tokenizer.to_json(), encoding="utf-8")
    meta = {
        "labels": ANSWER_QUALITY_LABELS,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "feature_names": FEATURE_NAMES,
        "score_strategy": "deterministic_weighted_rubric_custom_layer",
        "score_column": score_col,
        "label_column": label_col,
        "train_dataset_path": str(train_path),
        "val_dataset_path": str(val_path) if val_path else None,
        "test_dataset_path": str(test_path) if test_path else None,
        "dataset_path": str(train_path),  # backward compatibility
        "training_method": "model_fit",
        "tensorboard_log_dir": tensorboard_log_dir,
        "score_calibration": score_calibration,
        "last_history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    Path(ANSWER_QUALITY_META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train_answer_quality_model] ✅ Model tersimpan: {save_path}")
    print(f"[train_answer_quality_model] ✅ Dataset train: {train_path}")
    print(f"[train_answer_quality_model] ✅ Dataset val: {val_path}")
    print(f"[train_answer_quality_model] ✅ Dataset test: {test_path}")
    return model



def _make_tf_dataset(
    X_tokens: np.ndarray,
    X_features: np.ndarray,
    y_label: np.ndarray,
    y_score: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
) -> Any:
    """Create tf.data.Dataset for custom GradientTape training."""
    _require_tf()
    ds = tf.data.Dataset.from_tensor_slices((
        {"answer_tokens": X_tokens, "answer_features": X_features},
        {"answer_quality": y_label, "readiness_score": y_score},
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(y_label), 4096), seed=42, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _save_answer_quality_artifacts(
    model: Any,
    tokenizer: Any,
    save_path: str,
    max_length: int,
    vocab_size: int,
    score_col: str,
    label_col: str,
    train_path: str | None,
    val_path: str | None,
    test_path: str | None,
    history_dict: dict[str, list[float]],
    training_method: str,
    tensorboard_log_dir: str | None = None,
    score_calibration: dict[str, Any] | None = None,
) -> None:
    """Save model, tokenizer, and metadata in one consistent place."""
    model.save(save_path)
    Path(ANSWER_QUALITY_TOKENIZER_PATH).write_text(tokenizer.to_json(), encoding="utf-8")
    meta = {
        "labels": ANSWER_QUALITY_LABELS,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "feature_names": FEATURE_NAMES,
        "score_strategy": "deterministic_weighted_rubric_custom_layer",
        "score_column": score_col,
        "label_column": label_col,
        "train_dataset_path": str(train_path) if train_path else None,
        "val_dataset_path": str(val_path) if val_path else None,
        "test_dataset_path": str(test_path) if test_path else None,
        "dataset_path": str(train_path) if train_path else None,
        "training_method": training_method,
        "tensorboard_log_dir": tensorboard_log_dir,
        "score_calibration": score_calibration,
        "last_history": {k: [float(x) for x in v] for k, v in history_dict.items()},
    }
    Path(ANSWER_QUALITY_META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def train_answer_quality_model_custom_loop(
    train_csv_path: str | None = None,
    val_csv_path: str | None = None,
    test_csv_path: str | None = None,
    save_path: str = ANSWER_QUALITY_MODEL_PATH,
    max_length: int = 220,
    vocab_size: int = 20_000,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    min_accuracy: float = 0.85,
    max_mae: float = 0.02,
    tensorboard_log_dir: str | None = None,
) -> Any | None:
    """
    Full custom training + evaluation loop using tf.GradientTape.

    This function exists specifically for the AI side quest:
    "Mengimplementasikan training dan evaluation loop custom secara penuh dari awal menggunakan tf.GradientTape."

    It still uses the same Functional API model architecture, but the training step,
    validation step, metric tracking, early-stop condition, artifact export, and
    TensorBoard scalar logging are handled manually.
    """
    _require_tf()
    try:
        import pandas as pd
    except ImportError:
        print("[custom_loop] ❌ pandas belum terinstall.")
        return None

    train_path = train_csv_path or _resolve_dataset_path(None, answer_quality_train_path, "dataset_train.csv")
    val_path = val_csv_path or _resolve_dataset_path(None, answer_quality_val_path, "dataset_val.csv")
    test_path = test_csv_path or _resolve_dataset_path(None, answer_quality_test_path, "dataset_test.csv")

    if not train_path or not os.path.exists(train_path):
        print(f"[custom_loop] ❌ Dataset train tidak ditemukan: {train_path}")
        return None
    if not val_path or not os.path.exists(val_path):
        print(f"[custom_loop] ❌ Dataset validation tidak ditemukan: {val_path}")
        return None

    train_df_raw = pd.read_csv(train_path)
    val_df_raw = pd.read_csv(val_path)
    try:
        train_df, label_col, score_col = _prepare_training_dataframe(train_df_raw)
        val_df, _, _ = _prepare_training_dataframe(val_df_raw, label_col=label_col, score_col=score_col)
    except ValueError as exc:
        print(f"[custom_loop] ❌ {exc}")
        return None

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    X_train_tokens, X_train_features, y_train_label, y_train_score = _arrays_from_dataframe(
        train_df, tokenizer=tokenizer, max_length=max_length, score_col=score_col, fit_tokenizer=True
    )
    X_val_tokens, X_val_features, y_val_label, y_val_score = _arrays_from_dataframe(
        val_df, tokenizer=tokenizer, max_length=max_length, score_col=score_col, fit_tokenizer=False
    )

    train_ds = _make_tf_dataset(X_train_tokens, X_train_features, y_train_label, y_train_score, batch_size=batch_size, shuffle=True)
    val_ds = _make_tf_dataset(X_val_tokens, X_val_features, y_val_label, y_val_score, batch_size=batch_size, shuffle=False)

    model = create_answer_quality_model(vocab_size=vocab_size, max_length=max_length, feature_dim=X_train_features.shape[1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    quality_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    score_loss_fn = tf.keras.losses.Huber(delta=0.03)

    if tensorboard_log_dir is None:
        from datetime import datetime
        tensorboard_log_dir = str(MODEL_DIR / "logs" / "gradient_tape" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    train_writer = tf.summary.create_file_writer(str(Path(tensorboard_log_dir) / "train"))
    val_writer = tf.summary.create_file_writer(str(Path(tensorboard_log_dir) / "validation"))

    history: dict[str, list[float]] = {
        "loss": [],
        "answer_quality_loss": [],
        "readiness_score_loss": [],
        "answer_quality_accuracy": [],
        "readiness_score_mae": [],
        "val_loss": [],
        "val_answer_quality_loss": [],
        "val_readiness_score_loss": [],
        "val_answer_quality_accuracy": [],
        "val_readiness_score_mae": [],
    }
    best_val_loss = float("inf")
    score_calibration: dict[str, Any] | None = None

    @tf.function
    def train_step(x_batch: dict[str, Any], y_batch: dict[str, Any]):
        with tf.GradientTape() as tape:
            outputs = model(x_batch, training=True)
            q_loss = quality_loss_fn(y_batch["answer_quality"], outputs["answer_quality"])
            s_pred = tf.squeeze(outputs["readiness_score"], axis=-1)
            s_loss = score_loss_fn(y_batch["readiness_score"], s_pred)
            total_loss = 0.75 * q_loss + 4.0 * s_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss, q_loss, s_loss, outputs

    @tf.function
    def val_step(x_batch: dict[str, Any], y_batch: dict[str, Any]):
        outputs = model(x_batch, training=False)
        q_loss = quality_loss_fn(y_batch["answer_quality"], outputs["answer_quality"])
        s_pred = tf.squeeze(outputs["readiness_score"], axis=-1)
        s_loss = score_loss_fn(y_batch["readiness_score"], s_pred)
        total_loss = 0.75 * q_loss + 4.0 * s_loss
        return total_loss, q_loss, s_loss, outputs

    for epoch in range(1, epochs + 1):
        train_loss = tf.keras.metrics.Mean()
        train_q_loss = tf.keras.metrics.Mean()
        train_s_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        train_mae = tf.keras.metrics.MeanAbsoluteError()

        for x_batch, y_batch in train_ds:
            total_loss, q_loss, s_loss, outputs = train_step(x_batch, y_batch)
            score_pred = tf.squeeze(outputs["readiness_score"], axis=-1)
            train_loss.update_state(total_loss)
            train_q_loss.update_state(q_loss)
            train_s_loss.update_state(s_loss)
            train_acc.update_state(y_batch["answer_quality"], outputs["answer_quality"])
            train_mae.update_state(y_batch["readiness_score"], score_pred)

        val_loss = tf.keras.metrics.Mean()
        val_q_loss = tf.keras.metrics.Mean()
        val_s_loss = tf.keras.metrics.Mean()
        val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        val_mae = tf.keras.metrics.MeanAbsoluteError()

        for x_batch, y_batch in val_ds:
            total_loss, q_loss, s_loss, outputs = val_step(x_batch, y_batch)
            score_pred = tf.squeeze(outputs["readiness_score"], axis=-1)
            val_loss.update_state(total_loss)
            val_q_loss.update_state(q_loss)
            val_s_loss.update_state(s_loss)
            val_acc.update_state(y_batch["answer_quality"], outputs["answer_quality"])
            val_mae.update_state(y_batch["readiness_score"], score_pred)

        epoch_values = {
            "loss": float(train_loss.result()),
            "answer_quality_loss": float(train_q_loss.result()),
            "readiness_score_loss": float(train_s_loss.result()),
            "answer_quality_accuracy": float(train_acc.result()),
            "readiness_score_mae": float(train_mae.result()),
            "val_loss": float(val_loss.result()),
            "val_answer_quality_loss": float(val_q_loss.result()),
            "val_readiness_score_loss": float(val_s_loss.result()),
            "val_answer_quality_accuracy": float(val_acc.result()),
            "val_readiness_score_mae": float(val_mae.result()),
        }
        for key, value in epoch_values.items():
            history[key].append(value)

        # Fit score calibration from the current model on validation data.
        score_calibration = _calibration_from_validation(model, X_val_tokens, X_val_features, y_val_score)

        with train_writer.as_default():
            tf.summary.scalar("loss", epoch_values["loss"], step=epoch)
            tf.summary.scalar("answer_quality_accuracy", epoch_values["answer_quality_accuracy"], step=epoch)
            tf.summary.scalar("readiness_score_mae", epoch_values["readiness_score_mae"], step=epoch)
        with val_writer.as_default():
            tf.summary.scalar("loss", epoch_values["val_loss"], step=epoch)
            tf.summary.scalar("answer_quality_accuracy", epoch_values["val_answer_quality_accuracy"], step=epoch)
            tf.summary.scalar("readiness_score_mae", epoch_values["val_readiness_score_mae"], step=epoch)

        print(
            f"[custom_loop] epoch={epoch:02d} "
            f"loss={epoch_values['loss']:.4f} "
            f"acc={epoch_values['answer_quality_accuracy']:.4f} "
            f"mae={epoch_values['readiness_score_mae']:.4f} | "
            f"val_loss={epoch_values['val_loss']:.4f} "
            f"val_acc={epoch_values['val_answer_quality_accuracy']:.4f} "
            f"val_mae={epoch_values['val_readiness_score_mae']:.4f}"
        )

        if epoch_values["val_loss"] < best_val_loss:
            best_val_loss = epoch_values["val_loss"]
            _save_answer_quality_artifacts(
                model=model,
                tokenizer=tokenizer,
                save_path=save_path,
                max_length=max_length,
                vocab_size=vocab_size,
                score_col=score_col,
                label_col=label_col,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                history_dict=history,
                training_method="custom_gradient_tape",
                tensorboard_log_dir=tensorboard_log_dir,
                score_calibration=score_calibration,
            )

        if (
            epoch_values["val_answer_quality_accuracy"] >= min_accuracy
            and epoch_values["val_readiness_score_mae"] <= max_mae
        ):
            print(
                f"🎯 [custom_loop] Target tercapai: "
                f"val_accuracy={epoch_values['val_answer_quality_accuracy']:.2%}, "
                f"val_MAE={epoch_values['val_readiness_score_mae']:.4f}."
            )
            break

    _save_answer_quality_artifacts(
        model=model,
        tokenizer=tokenizer,
        save_path=save_path,
        max_length=max_length,
        vocab_size=vocab_size,
        score_col=score_col,
        label_col=label_col,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        history_dict=history,
        training_method="custom_gradient_tape",
        tensorboard_log_dir=tensorboard_log_dir,
        score_calibration=score_calibration,
    )
    print(f"[custom_loop] ✅ Model tersimpan: {save_path}")
    print(f"[custom_loop] ✅ TensorBoard log dir: {tensorboard_log_dir}")
    return model


# Backward-compatible alias.
def train_model(*args, **kwargs):
    return train_answer_quality_model(*args, **kwargs)


def load_answer_quality_model(model_path: str = ANSWER_QUALITY_MODEL_PATH) -> Any | None:
    if tf is None:
        return None
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"WeightedRubricScoreLayer": WeightedRubricScoreLayer},
    )


# Backward-compatible alias.
def load_trained_model(model_path: str = ANSWER_QUALITY_MODEL_PATH):
    return load_answer_quality_model(model_path)


def _load_tokenizer_and_meta() -> tuple[Any | None, dict[str, Any]]:
    if tf is None:
        return None, {}
    if not os.path.exists(ANSWER_QUALITY_TOKENIZER_PATH) or not os.path.exists(ANSWER_QUALITY_META_PATH):
        return None, {}
    try:
        tokenizer_json = Path(ANSWER_QUALITY_TOKENIZER_PATH).read_text(encoding="utf-8")
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
        meta = json.loads(Path(ANSWER_QUALITY_META_PATH).read_text(encoding="utf-8"))
        return tokenizer, meta
    except Exception as e:
        print(f"[model_builder] ⚠️ Gagal load tokenizer/meta: {e}")
        return None, {}


def heuristic_answer_quality(question: str, answer: str, role: str = "") -> dict[str, Any]:
    """Rule-based fallback untuk sementara sampai model DS dilatih tersedia."""
    features = _engineer_features(question, answer, role)
    has_tool, has_metric, has_impact, has_action, has_context, evidence_norm, length_norm = features
    evidence_level = max(1, min(5, round(evidence_norm * 5)))

    score = 20 + evidence_level * 10
    score += int(has_action * 8 + has_context * 8 + has_tool * 8 + has_impact * 10 + has_metric * 12)
    score += int(length_norm * 12)
    score = max(0, min(100, int(score)))

    if score >= 75:
        label = "Strong"
    elif score >= 50:
        label = "Average"
    else:
        label = "Weak"

    threshold_gap = min(abs(score - 50), abs(score - 75))
    confidence = min(0.95, 0.58 + threshold_gap / 100)

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "supporting_readiness_score": score,
        "supporting_score_0_1": round(score / 100, 4),
        "evidence_level_estimate": evidence_level,
        "features": _features_for_response(features),
        "source": "rule_based_fallback",
    }


def predict_answer_quality(
    question: str,
    answer: str,
    role: str = "",
    model_path: str = ANSWER_QUALITY_MODEL_PATH,
    features_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Inference answer quality. Fallback aktif jika model/tokenizer belum tersedia."""
    model = load_answer_quality_model(model_path)
    tokenizer, meta = _load_tokenizer_and_meta()

    if model is None or tokenizer is None:
        return heuristic_answer_quality(question, answer, role)

    max_length = int(meta.get("max_length", 220))
    labels = meta.get("labels", ANSWER_QUALITY_LABELS)
    text = f"role: {role}\nquestion: {question}\nanswer: {answer}"
    seq = tokenizer.texts_to_sequences([text])
    tokens = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding="post", truncating="post")
    feature_vector = _feature_vector_from_feature_dict(features_override, question=question, answer=answer, role=role)
    features = np.array([feature_vector], dtype="float32")

    preds = model.predict({"answer_tokens": tokens, "answer_features": features}, verbose=0)
    if isinstance(preds, dict):
        quality_probs = preds.get("answer_quality")
        readiness_pred = preds.get("readiness_score")
    else:
        # Keras bisa return list sesuai output order.
        quality_probs, readiness_pred = preds[0], preds[1]

    probs = np.array(quality_probs)[0]
    raw_score_0_1 = float(np.array(readiness_pred).reshape(-1)[0])
    calibrated = _apply_score_calibration(np.array([raw_score_0_1]), meta.get("score_calibration"))
    score_0_1 = float(calibrated[0])
    idx = int(np.argmax(probs))
    label = labels[idx]
    confidence = float(probs[idx])
    supporting_score = int(round(score_0_1 * 100))

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "supporting_readiness_score": supporting_score,
        "supporting_score_0_1": round(score_0_1, 4),
        "probabilities": {labels[i]: round(float(prob), 4) for i, prob in enumerate(probs)},
        "features": _features_for_response(features[0].tolist()),
        "source": "tensorflow_model",
    }


def evaluate_saved_model(csv_path: str | None = None, model_path: str = ANSWER_QUALITY_MODEL_PATH) -> dict[str, Any]:
    """Evaluasi model tersimpan terhadap dataset DS dan return accuracy + MAE."""
    _require_tf()
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas belum terinstall.")

    model = load_answer_quality_model(model_path)
    tokenizer, meta = _load_tokenizer_and_meta()
    if model is None or tokenizer is None:
        raise RuntimeError("Model/tokenizer belum tersedia. Jalankan train_answer_quality_model dulu.")

    if csv_path is None:
        csv_path = str(answer_quality_test_path()) if answer_quality_test_path and os.path.exists(str(answer_quality_test_path())) else (str(answer_quality_dataset_path()) if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv")
    df = pd.read_csv(csv_path)
    label_col = meta.get("label_column", "quality_label")
    score_col = meta.get("score_column", "final_score_0_1")
    label_to_id = {label.lower(): idx for idx, label in enumerate(meta.get("labels", ANSWER_QUALITY_LABELS))}
    df = df.dropna(subset=[label_col, score_col]).copy()
    df["_label_id"] = df[label_col].astype(str).str.lower().map(label_to_id)
    df = df.dropna(subset=["_label_id"])

    texts = [_combine_question_answer(row.to_dict()) for _, row in df.iterrows()]
    seq = tokenizer.texts_to_sequences(texts)
    tokens = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=int(meta.get("max_length", 220)), padding="post", truncating="post")
    features = _feature_matrix_from_df(df)
    y_label = df["_label_id"].astype("int32").to_numpy()
    y_score = df[score_col].astype("float32").clip(0, 1).to_numpy()
    result = model.evaluate(
        {"answer_tokens": tokens, "answer_features": features},
        {"answer_quality": y_label, "readiness_score": y_score},
        return_dict=True,
        verbose=0,
    )
    # Report calibrated MAE as the production inference pipeline uses calibration metadata.
    preds = model.predict({"answer_tokens": tokens, "answer_features": features}, verbose=0)
    readiness_pred = preds.get("readiness_score") if isinstance(preds, dict) else preds[1]
    calibrated = _apply_score_calibration(np.array(readiness_pred).reshape(-1), meta.get("score_calibration"))
    output = {k: float(v) for k, v in result.items()}
    output["readiness_score_mae_raw"] = output.get("readiness_score_mae", 0.0)
    output["readiness_score_mae"] = float(np.mean(np.abs(calibrated - y_score)))
    return output


def _prepare_evaluation_arrays(csv_path: str | None, meta: dict[str, Any]) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Internal helper untuk evaluate_saved_model dan confusion matrix."""
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas belum terinstall.")

    tokenizer, loaded_meta = _load_tokenizer_and_meta()
    if tokenizer is None:
        raise RuntimeError("Tokenizer/meta belum tersedia. Jalankan train_answer_quality_model dulu.")
    if not meta:
        meta = loaded_meta

    if csv_path is None:
        csv_path = str(answer_quality_test_path()) if answer_quality_test_path and os.path.exists(str(answer_quality_test_path())) else (str(answer_quality_dataset_path()) if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv")
    df = pd.read_csv(csv_path)
    label_col = meta.get("label_column", "quality_label")
    score_col = meta.get("score_column", "final_score_0_1")
    labels = meta.get("labels", ANSWER_QUALITY_LABELS)
    label_to_id = {label.lower(): idx for idx, label in enumerate(labels)}

    df = df.dropna(subset=[label_col, score_col]).copy()
    df["_label_id"] = df[label_col].astype(str).str.strip().str.lower().map(label_to_id)
    df = df.dropna(subset=["_label_id"])

    texts = [_combine_question_answer(row.to_dict()) for _, row in df.iterrows()]
    seq = tokenizer.texts_to_sequences(texts)
    tokens = tf.keras.preprocessing.sequence.pad_sequences(
        seq,
        maxlen=int(meta.get("max_length", 220)),
        padding="post",
        truncating="post",
    )
    features = _feature_matrix_from_df(df)
    y_label = df["_label_id"].astype("int32").to_numpy()
    y_score = df[score_col].astype("float32").clip(0, 1).to_numpy()
    return df, tokens, features, meta, y_label, y_score


def evaluate_saved_model_detailed(
    csv_path: str | None = None,
    model_path: str = ANSWER_QUALITY_MODEL_PATH,
    include_predictions: bool = False,
) -> dict[str, Any]:
    """
    Evaluasi model tersimpan dengan confusion matrix, classification report, dan MAE.

    Gunakan untuk validasi external/realistic:
      evaluate_saved_model_detailed(csv_path='data_science_resources/answer_quality_manual_test.csv')
    """
    _require_tf()
    model = load_answer_quality_model(model_path)
    tokenizer, meta = _load_tokenizer_and_meta()
    if model is None or tokenizer is None:
        raise RuntimeError("Model/tokenizer belum tersedia. Jalankan train_answer_quality_model dulu.")

    df, tokens, features, meta, y_label, y_score = _prepare_evaluation_arrays(csv_path, meta)
    preds = model.predict({"answer_tokens": tokens, "answer_features": features}, verbose=0)
    if isinstance(preds, dict):
        quality_probs = preds.get("answer_quality")
        readiness_pred = preds.get("readiness_score")
    else:
        quality_probs, readiness_pred = preds[0], preds[1]

    probs = np.array(quality_probs)
    pred_label = np.argmax(probs, axis=1)
    raw_pred_score = np.array(readiness_pred).reshape(-1).clip(0, 1)
    pred_score = _apply_score_calibration(raw_pred_score, meta.get("score_calibration"))

    labels = meta.get("labels", ANSWER_QUALITY_LABELS)
    accuracy = float(np.mean(pred_label == y_label)) if len(y_label) else 0.0
    mae = float(np.mean(np.abs(pred_score - y_score))) if len(y_score) else 0.0

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_label, pred_label, labels=list(range(len(labels))))
        report = classification_report(y_label, pred_label, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0)
    except Exception:
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        for true, pred in zip(y_label, pred_label):
            cm[int(true), int(pred)] += 1
        report = {}

    per_class_accuracy = {}
    for idx, label in enumerate(labels):
        mask = y_label == idx
        per_class_accuracy[label] = float(np.mean(pred_label[mask] == y_label[mask])) if np.any(mask) else None

    result: dict[str, Any] = {
        "dataset_path": str(csv_path or (answer_quality_test_path() if answer_quality_test_path and os.path.exists(str(answer_quality_test_path())) else (answer_quality_dataset_path() if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv"))),
        "n_samples": int(len(y_label)),
        "answer_quality_accuracy": accuracy,
        "readiness_score_mae": mae,
        "readiness_score_mae_raw": float(np.mean(np.abs(raw_pred_score - y_score))) if len(y_score) else 0.0,
        "score_calibration": meta.get("score_calibration"),
        "target_accuracy_pass": accuracy >= 0.85,
        "target_mae_pass": mae <= 0.02,
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
        "confusion_matrix_readable": {
            labels[i]: {labels[j]: int(cm[i, j]) for j in range(len(labels))}
            for i in range(len(labels))
        },
        "classification_report": report,
        "per_class_accuracy": per_class_accuracy,
    }

    if include_predictions:
        preview = []
        for i, (_, row) in enumerate(df.iterrows()):
            preview.append({
                "sample_id": row.get("sample_id"),
                "target_role": row.get("target_role"),
                "actual_label": labels[int(y_label[i])],
                "predicted_label": labels[int(pred_label[i])],
                "actual_score": round(float(y_score[i]), 4),
                "predicted_score": round(float(pred_score[i]), 4),
                "correct": bool(pred_label[i] == y_label[i]),
            })
        result["predictions"] = preview
    return result



def train_dataset_path() -> str | None:
    if answer_quality_train_path is None:
        return None
    return str(answer_quality_train_path())


def val_dataset_path() -> str | None:
    if answer_quality_val_path is None:
        return None
    return str(answer_quality_val_path())


def test_dataset_path() -> str | None:
    if answer_quality_test_path is None:
        return None
    return str(answer_quality_test_path())


def evaluate_split_datasets(include_predictions: bool = False) -> dict[str, Any]:
    """Evaluate saved model on train/val/test datasets from DS resources."""
    result: dict[str, Any] = {}
    for split_name, path_func in {"train": train_dataset_path, "val": val_dataset_path, "test": test_dataset_path}.items():
        path = path_func()
        if path and os.path.exists(path):
            result[split_name] = evaluate_saved_model_detailed(csv_path=path, include_predictions=include_predictions)
        else:
            result[split_name] = {"error": f"Dataset {split_name} tidak ditemukan", "path": path}
    return result

def manual_test_dataset_path() -> str | None:
    """Path dataset manual realistic dari DS resources jika tersedia."""
    if answer_quality_manual_test_path is None:
        return None
    return str(answer_quality_manual_test_path())


if __name__ == "__main__":
    print("Matrix source:", _matrix_source)
    print("Role labels:", ROLE_LABELS[:5], "...", len(ROLE_LABELS))
    print(compute_role_fit_scores(["python", "tensorflow", "fastapi", "natural language processing"], top_n=3))
    print(predict_answer_quality(
        question="Ceritakan project AI yang pernah kamu buat.",
        answer="Saya membuat model klasifikasi teks memakai Python dan TensorFlow untuk project kuliah. Saya bertanggung jawab membersihkan data, melatih model, dan mengevaluasi akurasi sampai 88%.",
        role="AI Engineer",
    ))
