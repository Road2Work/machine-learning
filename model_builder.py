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
        answer_quality_manual_test_path,
        get_role_skill_detail,
        get_role_skill_matrix as _ds_role_skill_matrix,
        get_role_skill_weights,
        reload_all as _reload_ds_assets,
    )
except Exception:  # pragma: no cover
    answer_quality_dataset_path = None  # type: ignore
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
FEATURE_NAMES = ["has_tool", "has_metric", "has_impact", "has_action", "has_context", "evidence_level", "answer_length_words"]


def _require_tf():
    if tf is None or layers is None:
        raise RuntimeError("TensorFlow belum terinstall. Jalankan: pip install tensorflow")


if tf is not None:
    class TargetPerformanceCallback(tf.keras.callbacks.Callback):
        """Stop training saat accuracy dan MAE sudah memenuhi target side quest."""

        def __init__(self, min_accuracy: float = 0.85, max_mae: float = 0.02, save_path: str = ANSWER_QUALITY_MODEL_PATH):
            super().__init__()
            self.min_accuracy = min_accuracy
            self.max_mae = max_mae
            self.save_path = save_path

        def on_epoch_end(self, epoch: int, logs: dict | None = None):
            logs = logs or {}
            acc = logs.get("val_answer_quality_accuracy", logs.get("answer_quality_accuracy"))
            mae = logs.get("val_readiness_score_mae", logs.get("readiness_score_mae"))
            if acc is None or mae is None:
                return
            print(f"[callback] epoch={epoch + 1} accuracy={float(acc):.4f} mae={float(mae):.4f}")
            if float(acc) >= self.min_accuracy and float(mae) <= self.max_mae:
                print(f"🎯 Target tercapai: accuracy={float(acc):.2%}, MAE={float(mae):.4f}. Saving {self.save_path}")
                self.model.save(self.save_path)
                self.model.stop_training = True

    # Backward-compatible name dari kode lama.
    TargetQualityAccuracyCallback = TargetPerformanceCallback
else:
    class TargetPerformanceCallback:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TensorFlow belum terinstall.")
    TargetQualityAccuracyCallback = TargetPerformanceCallback


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
    return [
        float(has_tool),
        float(has_metric),
        float(has_impact),
        float(has_action),
        float(has_context),
        evidence_level / 5.0,
        length_norm,
    ]


def _features_for_response(features: list[float]) -> dict[str, Any]:
    """Format fitur untuk response API: evidence_level ditampilkan sebagai 1–5, bukan nilai normalisasi."""
    evidence_level = max(1, min(5, int(round(float(features[5]) * 5))))
    return {
        "has_tool": int(round(float(features[0]))),
        "has_metric": int(round(float(features[1]))),
        "has_impact": int(round(float(features[2]))),
        "has_action": int(round(float(features[3]))),
        "has_context": int(round(float(features[4]))),
        "evidence_level": evidence_level,
        "evidence_level_norm": round(float(features[5]), 4),
        "answer_length_words_norm": round(float(features[6]), 4),
        "answer_length_words_estimate": int(round(float(features[6]) * 250)),
    }


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
    """
    _require_tf()
    token_input = tf.keras.Input(shape=(max_length,), name="answer_tokens")
    feature_input = tf.keras.Input(shape=(feature_dim,), name="answer_features")

    x = layers.Embedding(vocab_size, embedding_dim, name="token_embedding")(token_input)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name="bilstm_context")(x)
    x = layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = layers.Concatenate(name="concat_text_and_features")([x, feature_input])
    x = layers.Dense(128, activation="relu", name="dense_readiness_features")(x)
    x = layers.Dropout(0.30, name="dropout_regularization")(x)
    x = layers.Dense(64, activation="relu", name="dense_quality_features")(x)

    answer_quality = layers.Dense(len(ANSWER_QUALITY_LABELS), activation="softmax", name="answer_quality")(x)
    readiness_score = layers.Dense(1, activation="sigmoid", name="readiness_score")(x)

    model = tf.keras.Model(
        inputs={"answer_tokens": token_input, "answer_features": feature_input},
        outputs={"answer_quality": answer_quality, "readiness_score": readiness_score},
        name="Road2Work_Answer_Quality_Model",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"answer_quality": "sparse_categorical_crossentropy", "readiness_score": "mae"},
        metrics={"answer_quality": ["accuracy"], "readiness_score": ["mae"]},
        loss_weights={"answer_quality": 1.0, "readiness_score": 0.5},
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
                    val = float(row_dict[name])
                    if name == "evidence_level":
                        val = val / 5.0
                    elif name == "answer_length_words":
                        val = min(val, 250.0) / 250.0
                    values.append(val)
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


def train_answer_quality_model(
    csv_path: str | None = None,
    save_path: str = ANSWER_QUALITY_MODEL_PATH,
    max_length: int = 220,
    vocab_size: int = 20_000,
    epochs: int = 25,
    min_accuracy: float = 0.85,
    max_mae: float = 0.02,
    validation_split: float = 0.2,
) -> Any | None:
    """
    Train model dari dataset jawaban interview Data Science.

    CSV minimum:
    - answer atau text
    - quality_label atau label: Weak / Average / Strong
    - final_score_0_1 atau score: nilai 0.0-1.0 untuk MAE
    Opsional: question, target_role, competency, engineered feature columns.
    """
    _require_tf()
    try:
        import pandas as pd
    except ImportError:
        print("[train_answer_quality_model] ❌ pandas belum terinstall.")
        return None

    if csv_path is None:
        csv_path = str(answer_quality_dataset_path()) if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv"
    if not os.path.exists(csv_path):
        print(f"[train_answer_quality_model] ❌ File {csv_path} belum ada.")
        return None

    df = pd.read_csv(csv_path)
    label_col = "quality_label" if "quality_label" in df.columns else "label"
    score_col = "final_score_0_1" if "final_score_0_1" in df.columns else "score"
    if label_col not in df.columns:
        print("[train_answer_quality_model] ❌ CSV butuh kolom quality_label atau label.")
        return None
    if score_col not in df.columns:
        print("[train_answer_quality_model] ❌ CSV butuh kolom final_score_0_1 atau score untuk MAE.")
        return None
    if not ({"answer", "text"}.intersection(df.columns)):
        print("[train_answer_quality_model] ❌ CSV butuh kolom answer atau text.")
        return None

    df = df.dropna(subset=[label_col, score_col]).copy()
    label_to_id = {label.lower(): idx for idx, label in enumerate(ANSWER_QUALITY_LABELS)}
    df["_label_id"] = df[label_col].astype(str).str.strip().str.lower().map(label_to_id)
    df = df.dropna(subset=["_label_id"])
    if df.empty:
        print("[train_answer_quality_model] ❌ Tidak ada data valid setelah label mapping.")
        return None

    texts = [_combine_question_answer(row.to_dict()) for _, row in df.iterrows()]
    y_label = df["_label_id"].astype("int32").to_numpy()
    y_score = df[score_col].astype("float32").clip(0, 1).to_numpy()
    X_features = _feature_matrix_from_df(df)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X_tokens = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

    model = create_answer_quality_model(vocab_size=vocab_size, max_length=max_length, feature_dim=X_features.shape[1])
    callback = TargetPerformanceCallback(min_accuracy=min_accuracy, max_mae=max_mae, save_path=save_path)
    history = model.fit(
        {"answer_tokens": X_tokens, "answer_features": X_features},
        {"answer_quality": y_label, "readiness_score": y_score},
        epochs=epochs,
        validation_split=validation_split,
        callbacks=[callback],
        verbose=1,
    )
    model.save(save_path)

    Path(ANSWER_QUALITY_TOKENIZER_PATH).write_text(tokenizer.to_json(), encoding="utf-8")
    meta = {
        "labels": ANSWER_QUALITY_LABELS,
        "max_length": max_length,
        "vocab_size": vocab_size,
        "feature_names": FEATURE_NAMES,
        "score_column": score_col,
        "label_column": label_col,
        "dataset_path": str(csv_path),
        "last_history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    Path(ANSWER_QUALITY_META_PATH).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train_answer_quality_model] ✅ Model tersimpan: {save_path}")
    print(f"[train_answer_quality_model] ✅ Tokenizer/meta tersimpan di {ANSWER_QUALITY_TOKENIZER_PATH} dan {ANSWER_QUALITY_META_PATH}")
    return model


# Backward-compatible alias.
def train_model(*args, **kwargs):
    return train_answer_quality_model(*args, **kwargs)


def load_answer_quality_model(model_path: str = ANSWER_QUALITY_MODEL_PATH) -> Any | None:
    if tf is None:
        return None
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


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
    features = np.array([_engineer_features(question, answer, role)], dtype="float32")

    preds = model.predict({"answer_tokens": tokens, "answer_features": features}, verbose=0)
    if isinstance(preds, dict):
        quality_probs = preds.get("answer_quality")
        readiness_pred = preds.get("readiness_score")
    else:
        # Keras bisa return list sesuai output order.
        quality_probs, readiness_pred = preds[0], preds[1]

    probs = np.array(quality_probs)[0]
    score_0_1 = float(np.array(readiness_pred).reshape(-1)[0])
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
        csv_path = str(answer_quality_dataset_path()) if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv"
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
    return {k: float(v) for k, v in result.items()}


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
        csv_path = str(answer_quality_dataset_path()) if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv"
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
    pred_score = np.array(readiness_pred).reshape(-1).clip(0, 1)

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
        "dataset_path": str(csv_path or (answer_quality_dataset_path() if answer_quality_dataset_path else "answer_quality_dataset_synthetic.csv")),
        "n_samples": int(len(y_label)),
        "answer_quality_accuracy": accuracy,
        "readiness_score_mae": mae,
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
