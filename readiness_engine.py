import requests
import json

# =====================================================================
# READINESS ENGINE - ROAD2WORK
# Menggunakan REST API untuk menghindari konflik library lokal
# =====================================================================
API_KEY = "API_KEY_DISINI" # Masukkan API Key aslimu di sini

# Kita gunakan model yang sudah terbukti ADA di daftar akunmu!
MODEL_NAME = "gemini-2.5-flash"
URL_API = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"

def tanya_gemini(prompt):
    """Fungsi dasar untuk menembak API Google."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(URL_API, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    else:
        return f"ERROR SISTEM: {response.text}"

def buat_pertanyaan_interview(role):
    """Tugas 1: Generator Pertanyaan"""
    prompt = f"""
    Kamu adalah HRD profesional di perusahaan teknologi.
    Buatlah 1 pertanyaan wawancara behavioral (menggali pengalaman) untuk posisi {role}.
    Pertanyaan harus menantang. HANYA berikan teks pertanyaannya saja tanpa kalimat pembuka/penutup.
    """
    return tanya_gemini(prompt)

def evaluasi_jawaban(pertanyaan, jawaban_user):
    """Tugas 2 & 3: Feedback Engine & Scoring Kesiapan"""
    prompt = f"""
    Kamu adalah AI Penilai Wawancara.
    Pertanyaan HRD: {pertanyaan}
    Jawaban Kandidat: {jawaban_user}
    
    Berikan evaluasi objektif. Format HANYA dalam JSON murni seperti ini:
    {{
        "readiness_score": 85,
        "hal_yang_sudah_baik": "Penjelasan di sini...",
        "saran_perbaikan_dengan_metode_STAR": "Penjelasan di sini..."
    }}
    """
    return tanya_gemini(prompt)

# ---------------------------------------------------------
# SIMULASI END-TO-END (INTERVIEW READINESS)
# ---------------------------------------------------------
if __name__ == "__main__":
    target_role = "Data Analyst"
    
    print(f"\n🚀 --- MEMULAI SIMULASI INTERVIEW: {target_role.upper()} ---")
    
    # 1. AI Membuat Pertanyaan
    print("⏳ Menyiapkan pertanyaan...")
    pertanyaan_hr = buat_pertanyaan_interview(target_role)
    print(f"\n🤖 HRD (AI): {pertanyaan_hr}")
    
    # 2. User Menjawab (Simulasi suara yang sudah diubah jadi teks oleh stt_utils.py)
    print("-" * 50)
    jawaban = input("🗣️ Jawaban Kamu (Ketik & Enter): ")
    print("-" * 50)
    
    # 3. AI Menganalisis dan Memberi Skor
    print("\n⏳ AI sedang menganalisis kesiapan wawancaramu...")
    hasil_evaluasi = evaluasi_jawaban(pertanyaan_hr, jawaban)
    
    print("\n✅ === HASIL READINESS SCORE ===")
    print(hasil_evaluasi)
    print("================================\n")