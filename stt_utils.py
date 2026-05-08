import os
import time
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# INISIALISASI MODEL AI
# Ditaruh di luar fungsi agar AI hanya melakukan "loading" satu kali saja 
# saat server FastAPI pertama kali dinyalakan oleh tim.
# ---------------------------------------------------------------------------
print("Memuat Model AI STT (Road2Work Engine)...")
# Menggunakan model "base" untuk keseimbangan terbaik antara akurasi kalimat dan kecepatan
model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)

def proses_audio_ke_teks(jalur_file):
    """
    Fungsi utama Engine STT Road2Work.
    Menerima jalur file audio (dari user via FastAPI) dan mengembalikan teks dalam format Dictionary (JSON-ready).
    """
    # 1. PROTEKSI ERROR: Cek apakah file audio dari user benar-benar ada/terkirim
    if not os.path.exists(jalur_file):
        return {
            "status": "error",
            "pesan": f"File tidak ditemukan di server: {jalur_file}",
            "data_transkrip": None
        }

    start_time = time.time()
    
    try:
        # 2. PROSES TRANSKRIPSI AI
        # vad_filter=True sangat penting untuk memotong hening/suara napas user agar lebih ngebut
        # beam_size=5 membuat AI berpikir lebih akurat dalam menyusun tata bahasa
        segments, info = model.transcribe(jalur_file, beam_size=5, language="id", vad_filter=True)
        
        # 3. PENGGABUNGAN TEKS
        # Mengumpulkan semua potongan kata yang didengar AI menjadi satu paragraf utuh
        kumpulan_teks = [segment.text.strip() for segment in segments]
        teks_final = " ".join(kumpulan_teks)
        
        waktu_proses = round(time.time() - start_time, 2)
        
        # 4. OUTPUT SUKSES
        # Format terstruktur ini yang akan diterima oleh FastAPI Adil dan diteruskan ke HP User
        return {
            "status": "success",
            "pesan": "Audio berhasil ditranskripsi.",
            "data_transkrip": teks_final,
            "waktu_proses_detik": waktu_proses
        }
        
    except Exception as e:
        # 5. PROTEKSI CRASH SYSTEM
        # Jika tiba-tiba mesin AI gagal (misal file suara korup), server tidak akan mati
        return {
            "status": "error",
            "pesan": f"Terjadi kesalahan internal pada mesin AI: {str(e)}",
            "data_transkrip": None
        }

# ===========================================================================
# BLOK PENGUJIAN LOKAL KETIKA NGULIK
# Blok ini HANYA berjalan kalau kamu menjalankan file ini langsung di komputermu.
# Blok ini TIDAK AKAN berjalan/mengganggu saat file ini di-import oleh Adil nanti.
# ===========================================================================
if __name__ == "__main__":
    import json
    
    # Ini nama file audio dummy yang ada di komputermu untuk testing
    file_audio_tes = "test_suara.ogg" # (Atau sesuaikan dengan nama file audio yang kamu punya) 
    
    print(f"\n[TESTING LOKAL] Menerima request file: {file_audio_tes}...")
    hasil_respon = proses_audio_ke_teks(file_audio_tes)
    
    print("\n=== RESPON API STT (YANG AKAN DIKIRIM KE FULLSTACK) ===")
    print(json.dumps(hasil_respon, indent=4))
    print("========================================================\n")