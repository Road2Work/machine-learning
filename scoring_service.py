import tensorflow as tf
import numpy as np
import os

# Menentukan jalur file model .keras milik Diva
MODEL_PATH = "answer_scoring_model.keras"

# Proteksi awal: Cek apakah file model sudah ada di folder lokal
if os.path.exists(MODEL_PATH):
    try:
        # Memuat model TensorFlow siap produksi
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ [AI-Engine: AnswerScoring] Model berhasil dimuat!")
    except Exception as e:
        print(f"❌ [AI-Engine: AnswerScoring] Gagal memuat model: {str(e)}")
        model = None
else:
    print(f"⚠️  [AI-Engine: AnswerScoring] Berkas '{MODEL_PATH}' tidak ditemukan. Silakan lakukan 'git pull'.")
    model = None

def get_answer_quality_score(user_answer: str) -> dict:
    """
    Fungsi inferensi untuk memprediksi kualitas jawaban wawancara user.
    Dibuat oleh Diva untuk di-import dan dikonsumsi oleh FastAPI milik Adil.
    """
    if model is None:
        return {"status": "error", "pesan": "Model tidak siap atau belum dimuat."}
    
    # 1. Konversi teks menjadi Tensor String murni (Proteksi Error str3968)
    input_tensor = tf.constant([user_answer], dtype=tf.string)
    
    # 2. Jalankan proses prediksi AI
    predictions = model.predict(input_tensor, verbose=0)[0]
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx]) * 100
    
    labels = ["Weak", "Average", "Strong"]
    
    # 3. Mengembalikan format Dictionary utuh yang siap dikirim oleh API
    return {
        "status": "success",
        "data": {
            "predicted_class": labels[class_idx],
            "confidence_percentage": round(confidence, 2),
            "score_breakdown": {
                "Weak": round(float(predictions[0]) * 100, 2),
                "Average": round(float(predictions[1]) * 100, 2),
                "Strong": round(float(predictions[2]) * 100, 2)
            }
        }
    }

# ===========================================================================
# BLOK PENGUJIAN MANDIRI LOKAL
# ===========================================================================
if __name__ == "__main__":
    print("\n[Testing Lokal] Mencoba inferensi kalimat baru...")
    test_jawaban = "saya mengoptimasi database menggunakan sql sehingga mempercepat efisiensi waktu laporan tim"
    
    hasil = get_answer_quality_score(test_jawaban)
    import json
    print(json.dumps(hasil, indent=4))