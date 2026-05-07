"""
model_builder.py — Deep Learning Model Architecture & Role Matching
Road2Work AI | CC26-PSU050
Author  : Muhammad Adil Imamul Haq Mubarak (AI Engineer – NLP Engine)
Role    : Arsitektur model TensorFlow, training pipeline, role-fit scoring
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# --------------------------------------------------------------------------- #
#  ROLE-SKILL MATRIX (akan diganti dataset dari Addya setelah kurasi selesai)  #
#  Format: { role_name: [required_canonical_skills] }                          #
# --------------------------------------------------------------------------- #
ROLE_SKILL_MATRIX: dict[str, list[str]] = {
    "Data Analyst": [
        "python", "sql", "pandas", "numpy", "data analysis",
        "data visualization", "statistics", "data wrangling",
    ],
    "AI Engineer": [
        "python", "tensorflow", "pytorch", "deep learning", "nlp",
        "machine learning", "scikit-learn", "rest api", "fastapi",
    ],
    "Data Scientist": [
        "python", "r", "statistics", "machine learning", "deep learning",
        "data analysis", "data visualization", "scikit-learn", "pandas",
    ],
    "ML Engineer": [
        "python", "tensorflow", "pytorch", "scikit-learn", "docker",
        "rest api", "git", "deep learning", "machine learning",
    ],
    "Backend Developer": [
        "python", "javascript", "typescript", "sql", "rest api",
        "express", "fastapi", "docker", "git", "linux",
    ],
}

# Label index (sinkronkan dengan num_classes di create_nlp_model)
ROLE_LABELS: list[str] = list(ROLE_SKILL_MATRIX.keys())


# --------------------------------------------------------------------------- #
#  1. CUSTOM CALLBACK (Syarat Wajib Main Quest)                                #
# --------------------------------------------------------------------------- #
class TargetAccuracyCallback(tf.keras.callbacks.Callback):
    """
    Menghentikan training otomatis saat akurasi target tercapai,
    lalu menyimpan model ke format .keras.

    Args:
        target_acc (float): Ambang akurasi yang menjadi trigger stop. Default 0.85.
        save_path  (str)  : Path file .keras tempat model disimpan.
    """

    def __init__(self, target_acc: float = 0.85, save_path: str = "road2work_model.keras"):
        super().__init__()
        self.target_acc = target_acc
        self.save_path  = save_path

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", None)

        if acc >= self.target_acc:
            print(f"\n🎯 Target akurasi {self.target_acc*100:.0f}% tercapai "
                  f"(train={acc*100:.2f}%"
                  + (f", val={val_acc*100:.2f}%" if val_acc else "")
                  + ").")
            print(f"   Menyimpan model ke '{self.save_path}' dan menghentikan training...")
            self.model.stop_training = True
            self.model.save(self.save_path)


# --------------------------------------------------------------------------- #
#  2. MODEL ARSITEKTUR (Functional API — Syarat Wajib Main Quest)              #
# --------------------------------------------------------------------------- #
def create_nlp_model(
    vocab_size: int    = 10_000,
    embedding_dim: int = 64,
    max_length: int    = 200,
    num_classes: int   = len(ROLE_LABELS),
) -> tf.keras.Model:
    """
    Membangun model Deep Learning NLP menggunakan TensorFlow Functional API.

    Input : Urutan token integer (hasil tokenisasi + padding teks CV).
    Output: Probabilitas kecocokan ke setiap role (softmax).

    Args:
        vocab_size    : Ukuran kamus tokenizer.
        embedding_dim : Dimensi vektor embedding setiap token.
        max_length    : Panjang maksimum urutan token (padding/truncate).
        num_classes   : Jumlah role yang diklasifikasikan.

    Returns:
        tf.keras.Model: Model siap di-train (.compile sudah dipanggil).
    """
    # --- Input ---
    inputs = tf.keras.Input(shape=(max_length,), name="cv_text_input")

    # --- Embedding ---
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="token_embedding",
    )(inputs)

    # --- Feature Extraction ---
    # Catatan untuk Addya: setelah data tersedia, kita bisa bandingkan
    # GlobalAveragePooling1D vs Bidirectional LSTM di sini.
    x = layers.GlobalAveragePooling1D(name="pooling")(x)
    x = layers.Dense(128, activation="relu", name="hidden_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)   # Cegah overfitting
    x = layers.Dense(64,  activation="relu", name="hidden_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # --- Output ---
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="role_prediction",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Road2Work_Engine")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# --------------------------------------------------------------------------- #
#  3. ROLE-FIT SCORING (Rule-based — digunakan sebelum model dilatih)          #
# --------------------------------------------------------------------------- #
def compute_role_fit_scores(
    user_skills: list[str],
    top_n: int = 3,
) -> list[dict]:
    """
    Menghitung role-fit score berbasis overlap skill pengguna vs role-skill matrix.
    Digunakan sebagai fallback / explainable baseline sebelum model ML siap.

    Args:
        user_skills : Daftar canonical skill pengguna (output nlp_utils.extract_skills).
        top_n       : Jumlah rekomendasi role yang dikembalikan.

    Returns:
        list[dict]: Top-N role diurutkan dari skor tertinggi, format:
            [
              {
                "role": "Data Analyst",
                "score": 88,          # persentase (0–100)
                "matched_skills": [...],
                "gap_skills": [...],
              },
              ...
            ]
    """
    user_set = set(user_skills)
    results = []

    for role, required in ROLE_SKILL_MATRIX.items():
        required_set = set(required)
        matched  = sorted(user_set & required_set)
        missing  = sorted(required_set - user_set)
        score    = round(len(matched) / len(required_set) * 100) if required_set else 0

        results.append({
            "role":           role,
            "score":          score,
            "matched_skills": matched,
            "gap_skills":     missing,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# --------------------------------------------------------------------------- #
#  4. MODEL-BASED PREDICTION (digunakan setelah model dilatih)                 #
# --------------------------------------------------------------------------- #
def predict_role_from_model(
    model: tf.keras.Model,
    tokenized_input: np.ndarray,
    top_n: int = 3,
) -> list[dict]:
    """
    Memprediksi role menggunakan model TF yang sudah dilatih.

    Args:
        model          : Model hasil load atau training.
        tokenized_input: Array shape (1, max_length) hasil tokenizer.
        top_n          : Jumlah top role yang dikembalikan.

    Returns:
        list[dict]: Top-N prediksi role dengan probabilitas.
    """
    probs = model.predict(tokenized_input, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]

    return [
        {
            "role":        ROLE_LABELS[i],
            "probability": round(float(probs[i]) * 100, 2),
        }
        for i in top_indices
    ]


# --------------------------------------------------------------------------- #
#  SELF-TEST                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 55)
    print("  MODEL BUILDER — SELF TEST")
    print("=" * 55)

    print("\n[1] ARSITEKTUR MODEL")
    model = create_nlp_model()
    model.summary()

    print("\n[2] ROLE-FIT SCORING (Rule-based)")
    dummy_skills = ["python", "tensorflow", "nlp", "machine learning", "fastapi", "git"]
    recommendations = compute_role_fit_scores(dummy_skills, top_n=3)
    for rec in recommendations:
        print(f"\n  Role   : {rec['role']}")
        print(f"  Score  : {rec['score']}%")
        print(f"  Matched: {rec['matched_skills']}")
        print(f"  Gap    : {rec['gap_skills']}")