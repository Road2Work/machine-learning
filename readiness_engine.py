"""
readiness_engine.py — Interview Readiness Engine (Diva)
Road2Work AI | CC26-PSU050

v1.0.0 — aligned with Updated Product Overview
- Prompt evaluasi mengembalikan full schema Section 8.11.
- Tetap menggunakan REST API Gemini agar kompatibel dengan implementasi Diva.
- Key legacy tidak lagi menjadi output utama.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests


API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
URL_API = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
MAX_RETRIES = 3

if not API_KEY:
    print("[readiness_engine] ⚠️ GEMINI_API_KEY belum tersedia. Evaluasi REST Gemini akan fallback error string.")


def _tanya_gemini_safe(prompt: str) -> str:
    """Wrapper retry-safe untuk panggilan REST API Gemini."""
    if not API_KEY:
        return json.dumps(_fallback_evaluation("", "GEMINI_API_KEY belum tersedia."), ensure_ascii=False)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(URL_API, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

            if response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"[readiness_engine] HTTP {response.status_code}. Retry dalam {wait}s...")
                time.sleep(wait)
                continue

            return json.dumps(_fallback_evaluation("", f"Gemini HTTP {response.status_code}"), ensure_ascii=False)

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return json.dumps(_fallback_evaluation("", "Request Gemini timeout."), ensure_ascii=False)
        except requests.exceptions.RequestException as e:
            return json.dumps(_fallback_evaluation("", f"Koneksi Gemini gagal: {e}"), ensure_ascii=False)

    return json.dumps(_fallback_evaluation("", "Kesalahan tidak terduga."), ensure_ascii=False)


def tanya_gemini(prompt: str) -> str:
    return _tanya_gemini_safe(prompt)


def buat_pertanyaan_interview(role: str) -> str:
    """Generate satu pertanyaan behavioral sederhana."""
    prompt = f"""
Kamu adalah HRD profesional.
Buat 1 pertanyaan interview behavioral untuk posisi {role}.
Pertanyaan harus mendorong kandidat menjelaskan pengalaman nyata dengan evidence.
HANYA berikan teks pertanyaannya saja.
"""
    return _tanya_gemini_safe(prompt)


def evaluasi_jawaban(
    pertanyaan: str,
    jawaban_user: str,
    role: str = "posisi yang dipilih",
    context_summary: str = "",
) -> str:
    """
    Evaluasi jawaban wawancara user.

    Output JSON string schema:
    {
      "score_breakdown": {...},
      "final_score": 0-100,
      "evidence_level": 1-5,
      "weakness": [...],
      "need_clarification": true/false,
      "clarification_type": "tools|impact|contribution|specificity|null",
      "feedback": "...",
      "stronger_answer": "..."
    }
    """
    prompt = f"""
Kamu adalah AI Penilai Wawancara Road2Work.id yang objektif.
Role target: {role}
Konteks kandidat: {context_summary}
Pertanyaan HRD: {pertanyaan}
Jawaban kandidat: {jawaban_user}

Nilai dengan rubric:
- role_relevance 25%
- star_structure 20%
- evidence_specificity 20%
- technical_accuracy 15%
- communication_clarity 10%
- self_awareness 10%

Evidence Ladder:
1 Claim saja
2 Skill/tools disebut
3 Ada konteks pengalaman/project
4 Ada impact/hasil kualitatif
5 Ada hasil terukur/angka

Clarification diperlukan jika jawaban terlalu umum, kontribusi tidak jelas,
tools belum disebut, atau dampak belum dijelaskan.

Aturan stronger_answer:
Jangan mengarang angka, tools, company, project, atau hasil yang tidak disebut kandidat.
Gunakan placeholder [sebutkan ... jika ada] bila datanya tidak tersedia.

Balas HANYA JSON murni valid:
{{
  "score_breakdown": {{
    "role_relevance": 0,
    "star_structure": 0,
    "evidence_specificity": 0,
    "technical_accuracy": 0,
    "communication_clarity": 0,
    "self_awareness": 0
  }},
  "final_score": 0,
  "evidence_level": 1,
  "weakness": ["..."],
  "need_clarification": true,
  "clarification_type": "tools|impact|contribution|specificity|null",
  "feedback": "maksimal 3 kalimat",
  "stronger_answer": "versi jawaban yang lebih kuat tanpa mengarang data"
}}
"""
    raw = _tanya_gemini_safe(prompt)
    # Jika model mengembalikan non-JSON/error, main.py tetap punya normalizer.
    return raw


def _fallback_evaluation(answer: str, reason: str = "") -> dict[str, Any]:
    """Fallback local agar output tetap berbentuk schema baru."""
    text = (answer or "").lower()
    words = len(text.split())
    has_tools = bool(re.search(r"\b(python|sql|excel|tensorflow|pytorch|fastapi|api|dashboard|model)\b", text))
    has_impact = bool(re.search(r"\b(dampak|hasil|meningkat|mengurangi|mempercepat|membantu|akurasi|efisiensi)\b", text))
    has_metric = bool(re.search(r"\d+|%|persen|jam|menit|hari|bulan", text))

    evidence_level = 1 + int(has_tools) + int(words > 25) + int(has_impact) + int(has_metric)
    evidence_level = max(1, min(5, evidence_level))
    base = 30 + evidence_level * 10 + min(words, 80) // 4
    base = max(0, min(100, base))

    breakdown = {
        "role_relevance": base,
        "star_structure": max(0, base - 10),
        "evidence_specificity": max(0, base - (0 if has_metric else 15)),
        "technical_accuracy": base if has_tools else max(0, base - 20),
        "communication_clarity": min(100, base + 5),
        "self_awareness": max(0, base - 5),
    }
    weakness = [key for key, value in breakdown.items() if value < 60]
    if not has_tools:
        weakness.append("tools")
    if not has_impact:
        weakness.append("impact")

    clarification_type = "tools" if not has_tools else "impact" if not has_impact else "specificity"
    return {
        "score_breakdown": breakdown,
        "final_score": base,
        "evidence_level": evidence_level,
        "weakness": list(dict.fromkeys(weakness)),
        "need_clarification": base < 75,
        "clarification_type": clarification_type if base < 75 else None,
        "feedback": reason or "Jawaban sudah diterima, tetapi perlu detail evidence yang lebih kuat.",
        "stronger_answer": "Tambahkan konteks, kontribusi pribadi, tools yang digunakan, dan dampak nyata tanpa mengarang data.",
    }


if __name__ == "__main__":
    q = "Ceritakan project yang paling relevan untuk role AI Engineer."
    a = input("Jawaban: ")
    print(evaluasi_jawaban(q, a, role="AI Engineer"))
