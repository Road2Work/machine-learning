"""
stt_utils.py - Speech-to-Text Engine
Road2Work AI | CC26-PSU050
"""

import os
import time
from typing import Any

MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "90"))
MAX_AUDIO_FILE_SIZE_MB: int = int(os.getenv("MAX_AUDIO_FILE_SIZE_MB", "25"))

_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
_MODEL_DEVICE = os.getenv("STT_MODEL_DEVICE", "cpu")
_MODEL_COMPUTE = os.getenv("STT_MODEL_COMPUTE", "int8")
_MODEL_THREADS = int(os.getenv("STT_MODEL_THREADS", "4"))
_STT_BEAM_SIZE = int(os.getenv("STT_BEAM_SIZE", "5"))
_STT_VAD_FILTER = os.getenv("STT_VAD_FILTER", "true").lower() in {"1", "true", "yes", "on"}
_STT_INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    "Wawancara kerja Bahasa Indonesia dengan istilah teknologi Inggris: AI Engineer, Data Analyst, Backend Developer, Frontend Developer, computer vision, machine learning, dashboard, SQL, Python, JavaScript, Next.js, API, model, project, role, magang, kontribusi, dampak, hasil terukur.",
)
_whisper_model: Any = None


def _get_model() -> Any:
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel

            print(f"[stt_utils] Memuat Whisper '{_MODEL_SIZE}'...")
            _whisper_model = WhisperModel(
                _MODEL_SIZE,
                device=_MODEL_DEVICE,
                compute_type=_MODEL_COMPUTE,
                cpu_threads=_MODEL_THREADS,
            )
            print("[stt_utils] Model Whisper siap.")
        except ImportError:
            raise RuntimeError("[stt_utils] faster-whisper tidak terinstall. pip install faster-whisper")
    return _whisper_model


def get_model_info() -> dict:
    return {
        "loaded": _whisper_model is not None,
        "model_size": _MODEL_SIZE,
        "device": _MODEL_DEVICE,
        "max_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
        "vad_filter": _STT_VAD_FILTER,
    }


def proses_audio_ke_teks(jalur_file: str, language: str | None = None, max_duration_seconds: int | None = None) -> dict:
    """Transkripsi audio ke teks untuk endpoint POST /v1/stt/transcribe."""
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
        language_code = "id" if not language or language in {"id-ID", "id", "auto"} else language.split("-")[0]
        max_duration = int(max_duration_seconds or MAX_AUDIO_DURATION_SECONDS)
        segments_gen, info = model.transcribe(
            jalur_file,
            beam_size=_STT_BEAM_SIZE,
            language=language_code,
            vad_filter=_STT_VAD_FILTER,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=_STT_INITIAL_PROMPT,
            condition_on_previous_text=False,
            no_speech_threshold=0.65,
            log_prob_threshold=-1.2,
        )

        audio_duration = round(info.duration, 2) if hasattr(info, "duration") else None
        if audio_duration and audio_duration > max_duration:
            return {
                "status": "error",
                "kode": "AUDIO_TOO_LONG",
                "pesan": f"Durasi audio ({audio_duration:.0f}s) melebihi batas {max_duration}s.",
                "data_transkrip": None,
                "durasi_audio_detik": audio_duration,
            }

        segments = list(segments_gen)
        kumpulan_teks = [seg.text.strip() for seg in segments if seg.text.strip()]
        teks_final = " ".join(kumpulan_teks)

        log_probs = [float(getattr(seg, "avg_logprob", 0.0)) for seg in segments if getattr(seg, "avg_logprob", None) is not None]
        no_speech_probs = [float(getattr(seg, "no_speech_prob", 0.0)) for seg in segments if getattr(seg, "no_speech_prob", None) is not None]
        avg_logprob = sum(log_probs) / len(log_probs) if log_probs else None
        max_no_speech_prob = max(no_speech_probs) if no_speech_probs else 0.0
        confidence = 0.94
        if avg_logprob is not None:
            confidence = max(0.25, min(0.99, 0.95 + (avg_logprob * 0.35)))
        if max_no_speech_prob > 0.65:
            confidence = min(confidence, 0.55)

        if not teks_final:
            return {"status": "error", "pesan": "Tidak ada suara terdeteksi. Periksa mikrofon.", "data_transkrip": None}

        return {
            "status": "success",
            "pesan": "Audio berhasil ditranskripsi.",
            "data_transkrip": teks_final,
            "durasi_audio_detik": audio_duration,
            "waktu_proses_detik": round(time.time() - start_time, 2),
            "silence_detected": False,
            "silence_duration_seconds": 0,
            "confidence": round(confidence, 2),
            "avg_logprob": avg_logprob,
            "no_speech_probability": round(max_no_speech_prob, 2),
        }
    except RuntimeError as e:
        return {"status": "error", "pesan": str(e), "data_transkrip": None}
    except Exception as e:
        return {"status": "error", "pesan": f"Kesalahan memproses audio: {str(e)}", "data_transkrip": None}


if __name__ == "__main__":
    import json

    hasil = proses_audio_ke_teks("tes_suara.wav")
    print(json.dumps(hasil, indent=4, ensure_ascii=False))
