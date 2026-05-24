"""
stt_utils.py — Speech-to-Text Engine (Diva)
Road2Work AI | CC26-PSU050
Author  : Diva (AI Engineer 2 – Speech & Interview Intelligence)

CHANGELOG v0.5.0:
  [NEW]  MAX_AUDIO_DURATION_SECONDS=120 & MAX_AUDIO_FILE_SIZE_MB=25 (14.5: Batasi durasi audio)
  [NEW]  Cek durasi nyata dari info.duration setelah Whisper membuka file
  [FIX]  Lazy model loading dipertahankan
  [NEW]  get_model_info() untuk health check
"""

import os
import time
from typing import Any

MAX_AUDIO_DURATION_SECONDS: int = 150
MAX_AUDIO_FILE_SIZE_MB:     int = 25

_MODEL_SIZE    = "base"
_MODEL_DEVICE  = "cpu"
_MODEL_COMPUTE = "int8"
_MODEL_THREADS = 4
_whisper_model: Any = None


def _get_model() -> Any:
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[stt_utils] Memuat Whisper '{_MODEL_SIZE}'...")
            _whisper_model = WhisperModel(_MODEL_SIZE, device=_MODEL_DEVICE,
                                          compute_type=_MODEL_COMPUTE, cpu_threads=_MODEL_THREADS)
            print("[stt_utils] ✅ Model Whisper siap.")
        except ImportError:
            raise RuntimeError("[stt_utils] faster-whisper tidak terinstall. pip install faster-whisper")
    return _whisper_model


def get_model_info() -> dict:
    return {"loaded": _whisper_model is not None, "model_size": _MODEL_SIZE,
            "device": _MODEL_DEVICE, "max_duration_seconds": MAX_AUDIO_DURATION_SECONDS}


def proses_audio_ke_teks(jalur_file: str) -> dict:
    """
    Transkripsi audio ke teks. Dipanggil endpoint POST /v1/stt/transcribe.
    Validasi: file ada, ukuran ≤ 25MB, durasi ≤ 120 detik.
    """
    if not os.path.exists(jalur_file):
        return {"status": "error", "pesan": f"File tidak ditemukan: '{jalur_file}'", "data_transkrip": None}

    file_size_mb = os.path.getsize(jalur_file) / (1024 * 1024)
    if file_size_mb == 0:
        return {"status": "error", "pesan": "File audio kosong (0 byte).", "data_transkrip": None}
    if file_size_mb > MAX_AUDIO_FILE_SIZE_MB:
        return {"status": "error", "pesan": f"File terlalu besar ({file_size_mb:.1f}MB). Maks {MAX_AUDIO_FILE_SIZE_MB}MB.", "data_transkrip": None}

    start_time = time.time()
    try:
        model = _get_model()
        segments_gen, info = model.transcribe(jalur_file, beam_size=5, language="id", vad_filter=True)

        audio_duration = round(info.duration, 2) if hasattr(info, "duration") else None
        if audio_duration and audio_duration > MAX_AUDIO_DURATION_SECONDS:
            return {"status": "error", "pesan": f"Durasi audio ({audio_duration:.0f}s) melebihi batas {MAX_AUDIO_DURATION_SECONDS}s.", "data_transkrip": None}

        kumpulan_teks = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
        teks_final    = " ".join(kumpulan_teks)

        if not teks_final:
            return {"status": "error", "pesan": "Tidak ada suara terdeteksi. Periksa mikrofon.", "data_transkrip": None}

        return {
            "status": "success", "pesan": "Audio berhasil ditranskripsi.",
            "data_transkrip": teks_final,
            "durasi_audio_detik": audio_duration,
            "waktu_proses_detik": round(time.time() - start_time, 2),
        }
    except RuntimeError as e:
        return {"status": "error", "pesan": str(e), "data_transkrip": None}
    except Exception as e:
        return {"status": "error", "pesan": f"Kesalahan memproses audio: {str(e)}", "data_transkrip": None}


if __name__ == "__main__":
    import json
    hasil = proses_audio_ke_teks("tes_suara.wav")
    print(json.dumps(hasil, indent=4, ensure_ascii=False))