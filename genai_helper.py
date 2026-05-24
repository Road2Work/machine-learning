"""
genai_helper.py — Generative AI Integration (Google Gemini)
Road2Work AI | CC26-PSU050

v1.0.0 — aligned with Updated Product Overview
- Natural adaptive question generation with DS guardrail fallback.
- Clarifying question generation.
- Full answer evaluation schema: score_breakdown, final_score, evidence_level,
  weakness, need_clarification, clarification_type, feedback, stronger_answer.
- Result dashboard generation: strengths, improvement_areas,
  before_after_answer_improvement, next_practice_recommendation.

Catatan penting:
GenAI dipakai untuk wording natural dan feedback. Skor tetap dinormalisasi lagi
oleh main.py agar response stabil untuk frontend.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    from google import genai
except Exception:  # pragma: no cover - environment lokal bisa belum install SDK
    genai = None


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_RETRIES: int = 3

client = None
if genai is not None and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as exc:  # pragma: no cover
        print(f"[genai_helper] ⚠️ Gagal inisialisasi Gemini client: {exc}")


EVALUATION_WEIGHTS: dict[str, float] = {
    "role_relevance": 0.25,
    "star_structure": 0.20,
    "evidence_specificity": 0.20,
    "technical_accuracy": 0.15,
    "communication_clarity": 0.10,
    "self_awareness": 0.10,
}

_ALLOWED_CLARIFICATION_TYPES = {"tools", "impact", "contribution", "specificity", "context", "metric", "structure", "role_relevance", "technical", "clarity", "self_awareness", "professionalism", None, "null"}


def _call_gemini(prompt: str) -> str:
    """Panggil Gemini dengan retry. Jika API key belum ada, return marker aman."""
    if client is None:
        return "GENAI_UNAVAILABLE: GEMINI_API_KEY belum tersedia atau SDK belum terinstall."

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return (response.text or "").strip()
        except Exception as e:  # pragma: no cover - tergantung API eksternal
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err or "429" in err:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(
                        f"⚠️ [genai_helper] Gemini sibuk. "
                        f"Retry {attempt + 1}/{MAX_RETRIES} dalam {wait}s..."
                    )
                    time.sleep(wait)
                    continue
            return f"GENAI_ERROR: {err}"

    return "GENAI_ERROR: Terjadi kesalahan tidak terduga."


def _parse_json_response(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse JSON dari respons GenAI, termasuk jika dibungkus markdown fence."""
    if not isinstance(raw, str) or not raw.strip():
        return fallback

    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Kadang model memberi teks sebelum/sesudah JSON. Ambil blok JSON pertama.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return fallback


def _clamp_score(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _weighted_score(score_breakdown: dict[str, Any]) -> int:
    total = 0.0
    for key, weight in EVALUATION_WEIGHTS.items():
        total += _clamp_score(score_breakdown.get(key), 0) * weight
    return _clamp_score(total)


def _normalize_score_breakdown(score_breakdown: dict[str, Any] | None, fallback_score: int = 0) -> dict[str, int]:
    score_breakdown = score_breakdown or {}
    normalized: dict[str, int] = {}
    for key in EVALUATION_WEIGHTS:
        normalized[key] = _clamp_score(score_breakdown.get(key), fallback_score)
    return normalized


def _fallback_evidence_level(answer: str) -> int:
    """Evidence Ladder heuristic: 1 claim → 5 measurable result."""
    text = (answer or "").lower()
    has_skill = bool(re.search(r"\b(python|sql|excel|tableau|power bi|tensorflow|pytorch|fastapi|api|dashboard|model|analisis|data)\b", text))
    has_context = bool(re.search(r"\b(project|proyek|magang|organisasi|freelance|tugas|client|tim|kampus|perusahaan|penjualan|user)\b", text))
    has_impact = bool(re.search(r"\b(membantu|meningkat|mengurangi|mempercepat|dampak|hasil|efisien|akurasi|rekomendasi)\b", text))
    has_metric = bool(re.search(r"\b\d+\b|%|persen|jam|menit|hari|bulan|x\b", text))

    level = 1
    if has_skill:
        level = 2
    if has_skill and has_context:
        level = 3
    if has_skill and has_context and has_impact:
        level = 4
    if has_skill and has_context and has_impact and has_metric:
        level = 5
    return level


def _fallback_weakness(score_breakdown: dict[str, int], answer: str) -> list[str]:
    weakness: list[str] = []
    for key, score in score_breakdown.items():
        if score < 60:
            weakness.append(key)

    text = (answer or "").lower()
    if not re.search(r"\b(python|sql|excel|tableau|power bi|tensorflow|pytorch|fastapi|api|dashboard|model)\b", text):
        weakness.append("tools")
    if not re.search(r"\b(dampak|hasil|meningkat|mengurangi|mempercepat|akurasi|efisiensi|membantu)\b", text):
        weakness.append("impact")
    if not re.search(r"\b(saya|bagian saya|kontribusi|bertanggung jawab|mengerjakan|membuat|mengembangkan)\b", text):
        weakness.append("contribution")
    if _fallback_evidence_level(answer) <= 2:
        weakness.append("specificity")

    # unique preserving order
    seen = set()
    unique = []
    for item in weakness:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:5]


def _fallback_clarification_type(weakness: list[str]) -> str | None:
    priority = ["tools", "impact", "contribution", "specificity"]
    for item in priority:
        if item in weakness or f"evidence_{item}" in weakness:
            return item
    if "evidence_specificity" in weakness:
        return "specificity"
    return None


def _safe_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return default or []


# --------------------------------------------------------------------------- #
# 1. EXISTING HELPER — tetap dipertahankan untuk formalisasi profil
# --------------------------------------------------------------------------- #
def formalize_narrative(raw_narrative: str, role: str = "profesional") -> dict[str, Any]:
    """Ubah narasi/CV menjadi ringkasan profil profesional terstruktur."""
    empty = {
        "professional_summary": "",
        "experience_bullets": [],
        "highlighted_skills": [],
    }
    if not raw_narrative or not raw_narrative.strip():
        return empty

    prompt = f"""
Kamu adalah career coach profesional. Bantu kandidat menargetkan posisi {role}.

Narasi/CV kandidat:
---
{raw_narrative[:6000]}
---

Balas HANYA JSON valid:
{{
  "professional_summary": "ringkasan profesional maksimal 3 kalimat",
  "experience_bullets": ["maksimal 4 bullet action-result"],
  "highlighted_skills": ["3-7 skill paling relevan"]
}}
"""
    raw_response = _call_gemini(prompt)
    if raw_response.startswith("GENAI_"):
        # Fallback lokal agar extraction tetap jalan tanpa API key.
        text = " ".join(raw_narrative.split())
        return {
            "professional_summary": text[:400],
            "experience_bullets": [],
            "highlighted_skills": [],
        }
    return _parse_json_response(raw_response, empty)


# --------------------------------------------------------------------------- #
# 2. NATURAL ADAPTIVE QUESTION
# --------------------------------------------------------------------------- #
def generate_natural_question(
    role: str,
    interview_context: dict[str, Any],
    interview_state: dict[str, Any],
    role_skill_matrix: dict[str, list[str]] | None = None,
    competency_map: dict[str, Any] | None = None,
    question_seed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate 1 pertanyaan utama secara adaptif.
    Guardrail berasal dari role-skill matrix, competency map, question seed,
    context user, dan state interview.

    Support format DS terbaru:
    competency_map[role] = {"domain": ..., "role_family": ..., "competencies": [{...}]}
    question_seed[role] = [{"competency": ..., "question_seed": ..., "guardrail_signals": [...]}]
    """
    role_skill_matrix = role_skill_matrix or {}
    competency_map = competency_map or {}
    question_seed = question_seed or {}

    asked = interview_state.get("asked_questions", []) or []
    question_index = int(interview_state.get("main_question_index", 0))

    raw_competency = competency_map.get(role) or competency_map.get("default") or {}
    if isinstance(raw_competency, dict):
        competencies = raw_competency.get("competencies", [])
    elif isinstance(raw_competency, list):
        competencies = raw_competency
    else:
        competencies = []
    if not competencies:
        competencies = [
            {"competency": "Role Relevance", "description": "Kesesuaian pengalaman dengan target role."},
            {"competency": "Evidence Specificity", "description": "Kekuatan bukti, kontribusi, tools, dan impact."},
            {"competency": "Technical Accuracy", "description": "Ketepatan teknis sesuai target role."},
        ]

    selected_competency = competencies[question_index % len(competencies)]
    if isinstance(selected_competency, dict):
        target_competency = str(selected_competency.get("competency") or selected_competency.get("competency_id") or "role_relevance")
        competency_detail = selected_competency
    else:
        target_competency = str(selected_competency)
        competency_detail = {"competency": target_competency}

    seeds = question_seed.get(role) or question_seed.get("default") or []
    if not isinstance(seeds, list):
        seeds = []
    # Prioritaskan seed dengan competency yang sama; fallback cycling.
    matching_seeds = [s for s in seeds if isinstance(s, dict) and str(s.get("competency", "")).lower() == target_competency.lower()]
    selected_seed = (matching_seeds or seeds or [{}])[question_index % len(matching_seeds or seeds or [{}])]
    seed_text = selected_seed.get("question_seed") if isinstance(selected_seed, dict) else str(selected_seed)
    guardrail_signals = selected_seed.get("guardrail_signals", []) if isinstance(selected_seed, dict) else []

    skills = interview_context.get("skills", []) or []
    summary = interview_context.get("profile_summary", "")
    experiences = interview_context.get("experience_summary", "")
    role_skills = role_skill_matrix.get(role, [])

    prompt = f"""
Kamu berperan sebagai HRD interviewer profesional untuk posisi {role}.
Buat tepat 1 pertanyaan interview utama yang natural, tidak kaku, dan relevan.

Guardrail dari Data Science:
- Target competency: {target_competency}
- Detail competency: {json.dumps(competency_detail, ensure_ascii=False)}
- Role skill matrix: {role_skills}
- Question seed terpilih: {seed_text}
- Guardrail signals: {guardrail_signals}

Konteks kandidat:
- Skill kandidat: {skills}
- Ringkasan kandidat: {summary}
- Pengalaman/evidence kandidat: {experiences}
- Pertanyaan yang sudah ditanyakan: {asked}

Aturan:
1. Jangan mengulang pertanyaan yang sudah ada.
2. Jangan menanyakan hal yang terlalu jauh dari konteks kandidat.
3. Dorong kandidat menjawab dengan pengalaman nyata, kontribusi pribadi, tools, dan evidence.
4. Bahasa Indonesia, profesional, singkat, seperti HRD sungguhan.
5. Kalau question seed tersedia, gunakan sebagai arah pertanyaan tetapi boleh diparafrase agar natural.

Balas HANYA JSON valid:
{{
  "question": "teks pertanyaan",
  "competency_target": "{target_competency}",
  "question_type": "main"
}}
"""
    raw = _call_gemini(prompt)
    fallback_question = seed_text or _fallback_natural_question(role, skills, target_competency, asked)
    fallback = {
        "question": fallback_question,
        "competency_target": target_competency,
        "question_type": "main",
    }
    if raw.startswith("GENAI_"):
        return fallback

    parsed = _parse_json_response(raw, fallback)
    question = str(parsed.get("question") or fallback_question).strip()
    return {
        "question": question,
        "competency_target": str(parsed.get("competency_target") or target_competency),
        "question_type": "main",
    }

def _fallback_natural_question(role: str, skills: list[str], competency: str, asked: list[str]) -> str:
    skills_text = ", ".join(skills[:4]) if skills else "skill yang kamu punya"
    options = {
        "role_relevance": f"Ceritakan pengalaman paling relevan yang menunjukkan kamu siap untuk posisi {role}.",
        "evidence_specificity": f"Bisa jelaskan satu project yang pernah kamu kerjakan dengan {skills_text}, termasuk konteks dan hasilnya?",
        "technical_accuracy": f"Pilih satu pengalaman teknis yang paling kuat. Bagaimana proses, tools, dan keputusan teknis yang kamu ambil?",
        "communication_clarity": "Coba jelaskan salah satu pengalamanmu secara runtut: situasinya apa, tugasmu apa, aksi yang kamu ambil, dan hasilnya.",
        "self_awareness": f"Menurut kamu, kekuatan dan area pengembanganmu untuk posisi {role} apa saja?",
    }
    question = options.get(competency, options["role_relevance"])
    if question in asked:
        question = f"Ceritakan pengalaman lain yang bisa membuktikan kesiapanmu untuk posisi {role}, dengan contoh konkret."
    return question


# Backward-compatible wrapper. Tidak dipakai flow baru, tapi aman untuk self-test lama.
def generate_interview_questions(
    skills: list[str],
    role: str = "posisi yang dipilih",
    num_questions: int = 3,
) -> str:
    questions = []
    state = {"asked_questions": [], "main_question_index": 0}
    ctx = {"skills": skills, "profile_summary": ""}
    for index in range(num_questions):
        state["main_question_index"] = index
        generated = generate_natural_question(role, ctx, state)
        q = generated["question"]
        questions.append(f"{index + 1}. {q}")
        state["asked_questions"].append(q)
    return "\n".join(questions)


# --------------------------------------------------------------------------- #
# 3. ANSWER EVALUATION — FULL SCHEMA
# --------------------------------------------------------------------------- #
def evaluate_interview_answer(
    question: str,
    answer: str,
    role: str = "posisi yang dipilih",
    interview_context: dict[str, Any] | None = None,
    competency_target: str | None = None,
) -> dict[str, Any]:
    """Evaluasi jawaban interview dengan schema sesuai overview Section 8.11."""
    interview_context = interview_context or {}
    answer = answer or ""

    prompt = f"""
Kamu adalah AI Answer Evaluation Engine untuk platform Road2Work.id.
Nilai jawaban kandidat untuk role {role} secara objektif.

Pertanyaan HRD:
{question}

Jawaban kandidat:
{answer}

Konteks kandidat:
{json.dumps(interview_context, ensure_ascii=False)[:5000]}

Competency target: {competency_target or "general"}

Rubric wajib:
- role_relevance bobot 25%
- star_structure bobot 20%
- evidence_specificity bobot 20%
- technical_accuracy bobot 15%
- communication_clarity bobot 10%
- self_awareness bobot 10%

Evidence Ladder:
1 Claim saja
2 Skill/tools disebut
3 Ada konteks project/pengalaman
4 Ada impact/hasil kualitatif
5 Ada hasil terukur/angka

Aturan stronger_answer:
- Jangan mengarang pengalaman, angka, tools, company, atau hasil yang tidak disebut kandidat.
- Boleh menulis template dengan placeholder seperti "[sebutkan angka jika ada]".

Balas HANYA JSON valid sesuai schema ini:
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
  "weakness": ["tag_kelemahan"],
  "need_clarification": true,
  "clarification_type": "tools|impact|contribution|specificity|null",
  "feedback": "maksimal 3 kalimat",
  "stronger_answer": "versi jawaban lebih kuat, tanpa mengarang data"
}}
"""
    raw_response = _call_gemini(prompt)

    fallback_score = _heuristic_base_score(answer)
    fallback_breakdown = _heuristic_breakdown(question, answer, role)
    fallback_weakness = _fallback_weakness(fallback_breakdown, answer)
    fallback_clarification_type = _fallback_clarification_type(fallback_weakness)
    fallback = {
        "score_breakdown": fallback_breakdown,
        "final_score": _weighted_score(fallback_breakdown),
        "evidence_level": _fallback_evidence_level(answer),
        "weakness": fallback_weakness,
        "need_clarification": bool(fallback_clarification_type and fallback_score < 78),
        "clarification_type": fallback_clarification_type,
        "feedback": _fallback_feedback(fallback_weakness),
        "stronger_answer": _fallback_stronger_answer(answer),
    }

    if raw_response.startswith("GENAI_"):
        return fallback

    parsed = _parse_json_response(raw_response, fallback)
    return normalize_evaluation_schema(parsed, original_answer=answer)


def normalize_evaluation_schema(raw: dict[str, Any], original_answer: str = "") -> dict[str, Any]:
    """Normalisasi response GenAI/legacy ke schema final yang stabil."""
    # Legacy key support dari versi lama readiness_engine/genai_helper.
    if "readiness_score" in raw:
        legacy_score = _clamp_score(raw.get("readiness_score"), 0)
        raw = {
            "score_breakdown": {key: legacy_score for key in EVALUATION_WEIGHTS},
            "final_score": legacy_score,
            "evidence_level": _fallback_evidence_level(original_answer),
            "weakness": _fallback_weakness({key: legacy_score for key in EVALUATION_WEIGHTS}, original_answer),
            "need_clarification": legacy_score < 70,
            "clarification_type": None,
            "feedback": raw.get("hal_yang_sudah_baik", ""),
            "stronger_answer": raw.get("saran_perbaikan_dengan_metode_STAR", ""),
        }

    if "score" in raw and "score_breakdown" not in raw:
        score = _clamp_score(raw.get("score"), 0)
        raw["score_breakdown"] = {key: score for key in EVALUATION_WEIGHTS}
        raw["final_score"] = score

    score_breakdown = _normalize_score_breakdown(raw.get("score_breakdown"), _clamp_score(raw.get("final_score"), 0))
    final_score = _clamp_score(raw.get("final_score"), _weighted_score(score_breakdown))
    if final_score == 0 and any(score_breakdown.values()):
        final_score = _weighted_score(score_breakdown)

    evidence_level = max(1, min(5, int(raw.get("evidence_level") or _fallback_evidence_level(original_answer))))
    weakness = [str(w).strip() for w in _safe_list(raw.get("weakness")) if str(w).strip()]
    if not weakness:
        weakness = _fallback_weakness(score_breakdown, original_answer)

    clarification_type = raw.get("clarification_type")
    if clarification_type == "null":
        clarification_type = None
    if clarification_type not in _ALLOWED_CLARIFICATION_TYPES:
        clarification_type = _fallback_clarification_type(weakness)

    need_clarification = bool(raw.get("need_clarification", False))
    if final_score < 70 and evidence_level <= 3 and clarification_type:
        need_clarification = True
    if final_score >= 82:
        need_clarification = False
        clarification_type = None

    return {
        "score_breakdown": score_breakdown,
        "final_score": final_score,
        "evidence_level": evidence_level,
        "weakness": weakness,
        "need_clarification": need_clarification,
        "clarification_type": clarification_type,
        "feedback": str(raw.get("feedback") or _fallback_feedback(weakness)),
        "stronger_answer": str(raw.get("stronger_answer") or _fallback_stronger_answer(original_answer)),
    }


def _heuristic_base_score(answer: str) -> int:
    answer = answer or ""
    words = len(answer.split())
    score = 25
    if words >= 25:
        score += 15
    if words >= 60:
        score += 15
    score += (_fallback_evidence_level(answer) - 1) * 10
    if re.search(r"\b(situasi|tugas|aksi|hasil|result|dampak)\b", answer.lower()):
        score += 10
    return _clamp_score(score)


def _heuristic_breakdown(question: str, answer: str, role: str) -> dict[str, int]:
    text = (answer or "").lower()
    evidence = _fallback_evidence_level(answer)
    words = len(text.split())
    role_tokens = [token.lower() for token in re.split(r"\W+", role or "") if len(token) > 2]
    role_hit = any(token in text for token in role_tokens)

    technical = 45
    if re.search(r"\b(python|sql|excel|tableau|power bi|tensorflow|pytorch|fastapi|api|dashboard|model|dataset|database)\b", text):
        technical += 30
    if re.search(r"\b(karena|sehingga|hasil|akurasi|error|deploy|validasi|evaluasi)\b", text):
        technical += 15

    return {
        "role_relevance": _clamp_score(60 + (15 if role_hit else 0) + (10 if evidence >= 3 else -10)),
        "star_structure": _clamp_score(35 + min(words, 80) * 0.4 + (15 if re.search(r"\b(situasi|tugas|aksi|hasil)\b", text) else 0)),
        "evidence_specificity": _clamp_score(25 + evidence * 15),
        "technical_accuracy": _clamp_score(technical),
        "communication_clarity": _clamp_score(45 + min(words, 80) * 0.3),
        "self_awareness": _clamp_score(55 + (15 if re.search(r"\b(belajar|menyadari|tantangan|improve|evaluasi|feedback)\b", text) else 0)),
    }


def _fallback_feedback(weakness: list[str]) -> str:
    if not weakness:
        return "Jawaban sudah cukup jelas dan relevan. Tinggal pertahankan struktur dan bukti konkret."
    return "Jawaban sudah bisa dipahami, tetapi masih perlu dibuat lebih spesifik. Tambahkan konteks, kontribusi pribadi, tools, dan dampak agar terdengar lebih kuat."


def _fallback_stronger_answer(answer: str) -> str:
    if not answer.strip():
        return "Saya pernah mengerjakan [nama project/pengalaman]. Dalam project itu, tugas saya adalah [kontribusi pribadi], menggunakan [tools/skill], dan hasilnya [impact yang benar-benar terjadi]."
    return (
        f"Berdasarkan jawabanmu: {answer.strip()[:180]}... "
        "Agar lebih kuat, tambahkan situasi, tugas spesifikmu, aksi yang kamu lakukan, tools yang digunakan, dan hasil/impact yang benar-benar terjadi."
    )


# --------------------------------------------------------------------------- #
# 4. CLARIFICATION QUESTION ENGINE
# --------------------------------------------------------------------------- #
def generate_clarification_question(
    original_question: str,
    transcript: str,
    weakness_tags: list[str],
    role: str,
    clarification_type: str | None = None,
) -> str:
    """Generate pertanyaan klarifikasi natural berdasarkan kelemahan jawaban."""
    weakness_tags = weakness_tags or []
    clarification_type = clarification_type or _fallback_clarification_type(weakness_tags)

    prompt = f"""
Kamu adalah HRD interviewer untuk role {role}.
Kandidat baru menjawab pertanyaan berikut:
Pertanyaan awal: {original_question}
Jawaban kandidat: {transcript}
Weakness terdeteksi: {weakness_tags}
Clarification type: {clarification_type}

Buat 1 pertanyaan klarifikasi yang natural dan singkat.
Tujuannya menggali detail yang belum jelas, bukan menghakimi.
Jangan menambahkan asumsi pengalaman baru.

Balas HANYA teks pertanyaannya saja.
"""
    raw = _call_gemini(prompt)
    if raw.startswith("GENAI_") or len(raw.strip()) < 8:
        return _fallback_clarification_question(clarification_type, role)
    return raw.strip().strip('"')


def _fallback_clarification_question(clarification_type: str | None, role: str) -> str:
    mapping = {
        "tools": "Bisa kamu jelaskan tools, metode, atau teknologi apa yang kamu gunakan dalam pengalaman itu?",
        "impact": "Apa hasil atau dampak dari pekerjaan yang kamu lakukan dalam pengalaman tersebut?",
        "contribution": "Bagian mana yang benar-benar kamu kerjakan sendiri, dan apa tanggung jawab utamamu?",
        "specificity": "Bisa kamu ceritakan contoh yang lebih spesifik dari pengalaman itu?",
        "context": "Bisa jelaskan konteks proyek atau masalah yang sedang kamu hadapi saat itu?",
        "metric": "Apakah ada angka, metrik, atau indikator yang menunjukkan keberhasilan pekerjaanmu?",
        "structure": "Bisa ceritakan ulang secara runtut dari situasi, tugasmu, aksi yang kamu lakukan, sampai hasilnya?",
        "role_relevance": f"Bagaimana pengalaman itu berhubungan langsung dengan posisi {role} yang kamu targetkan?",
        "technical": "Bisa jelaskan detail teknis atau pendekatan yang kamu gunakan?",
        "clarity": "Bisa jelaskan ulang dengan lebih terstruktur dan singkat?",
        "self_awareness": "Apa pembelajaran atau hal yang akan kamu perbaiki dari pengalaman tersebut?",
    }
    return mapping.get(
        clarification_type,
        f"Bisa kamu tambahkan detail yang lebih konkret agar saya bisa menilai kesiapanmu untuk posisi {role}?",
    )


# --------------------------------------------------------------------------- #
# 5. RESULT DASHBOARD GENERATION
# --------------------------------------------------------------------------- #
def generate_result_dashboard(
    role: str,
    interview_context: dict[str, Any],
    answers: list[dict[str, Any]],
    final_score: int,
) -> dict[str, Any]:
    """Generate 4 komponen dashboard hasil interview."""
    compact_answers = []
    for item in answers:
        evaluation = item.get("evaluation", {}) or {}
        compact_answers.append({
            "question_type": item.get("question_type"),
            "question": item.get("question_text") or item.get("question"),
            "transcript": item.get("transcript", "")[:900],
            "score_breakdown": evaluation.get("score_breakdown"),
            "final_score": evaluation.get("final_score"),
            "evidence_level": evaluation.get("evidence_level"),
            "weakness": evaluation.get("weakness"),
            "feedback": evaluation.get("feedback"),
            "stronger_answer": evaluation.get("stronger_answer"),
        })

    prompt = f"""
Kamu adalah AI career coach Road2Work.id.
Buat dashboard hasil interview yang ringkas dan actionable untuk role {role}.

Konteks kandidat:
{json.dumps(interview_context, ensure_ascii=False)[:4000]}

Data jawaban dan evaluasi:
{json.dumps(compact_answers, ensure_ascii=False)[:9000]}

Final score: {final_score}

Aturan:
- Jangan mengarang pengalaman, angka, tools, atau hasil yang tidak ada di transcript.
- Strengths harus berdasarkan score tertinggi/evidence yang muncul.
- Improvement areas harus berdasarkan score terendah/weakness.
- Before-after memakai jawaban user asli dan versi yang lebih kuat tanpa data palsu.
- Next practice recommendation mapping dari kelemahan terbesar.

Balas HANYA JSON valid:
{{
  "strengths": [
    {{"title": "...", "reason": "...", "evidence": "kutipan/parafrase singkat dari jawaban"}}
  ],
  "improvement_areas": [
    {{"title": "...", "cause": "...", "suggestion": "..."}}
  ],
  "before_after_answer_improvement": {{
    "before": "jawaban awal user",
    "problem": "masalah utama",
    "after": "versi lebih kuat tanpa mengarang data",
    "why_better": "alasan singkat"
  }},
  "next_practice_recommendation": {{
    "practice_type": "Behavioral STAR Practice|Evidence Booster Practice|Technical Interview Practice|Answer Clarity Practice|Role Understanding Practice",
    "reason": "...",
    "focus": ["..."],
    "cta": "..."
  }}
}}
"""
    raw = _call_gemini(prompt)
    fallback = _fallback_dashboard(role, answers, final_score)
    if raw.startswith("GENAI_"):
        return fallback

    parsed = _parse_json_response(raw, fallback)
    return normalize_dashboard_schema(parsed, fallback)


def normalize_dashboard_schema(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    strengths = _safe_list(raw.get("strengths"), fallback["strengths"])[:3]
    improvement_areas = _safe_list(raw.get("improvement_areas"), fallback["improvement_areas"])[:3]
    before_after = raw.get("before_after_answer_improvement")
    if not isinstance(before_after, dict):
        before_after = fallback["before_after_answer_improvement"]
    next_practice = raw.get("next_practice_recommendation")
    if not isinstance(next_practice, dict):
        next_practice = fallback["next_practice_recommendation"]

    return {
        "strengths": strengths,
        "improvement_areas": improvement_areas,
        "before_after_answer_improvement": before_after,
        "next_practice_recommendation": next_practice,
    }


def _fallback_dashboard(role: str, answers: list[dict[str, Any]], final_score: int) -> dict[str, Any]:
    main_answers = [a for a in answers if a.get("question_type") == "main"] or answers
    first_answer = main_answers[0] if main_answers else {}
    transcript = first_answer.get("transcript", "")

    all_breakdowns: dict[str, list[int]] = {key: [] for key in EVALUATION_WEIGHTS}
    all_weakness: list[str] = []
    for ans in answers:
        evaluation = ans.get("evaluation", {}) or {}
        for key, score in (evaluation.get("score_breakdown") or {}).items():
            if key in all_breakdowns:
                all_breakdowns[key].append(_clamp_score(score))
        all_weakness.extend(evaluation.get("weakness") or [])

    avg_breakdowns = {
        key: (sum(values) / len(values) if values else 0)
        for key, values in all_breakdowns.items()
    }
    top_components = sorted(avg_breakdowns.items(), key=lambda item: item[1], reverse=True)[:3]
    low_components = sorted(avg_breakdowns.items(), key=lambda item: item[1])[:3]

    practice_map = {
        "star_structure": "Behavioral STAR Practice",
        "evidence_specificity": "Evidence Booster Practice",
        "technical_accuracy": "Technical Interview Practice",
        "communication_clarity": "Answer Clarity Practice",
        "role_relevance": "Role Understanding Practice",
        "self_awareness": "Behavioral STAR Practice",
    }
    lowest = low_components[0][0] if low_components else "evidence_specificity"

    return {
        "strengths": [
            {
                "title": key.replace("_", " ").title(),
                "reason": f"Komponen ini relatif lebih kuat dalam simulasi untuk role {role}.",
                "evidence": "Terlihat dari jawaban yang sudah diberikan kandidat." if transcript else "Belum ada kutipan jawaban.",
            }
            for key, _ in top_components
        ] or [{"title": "Konteks Dasar", "reason": "Kandidat sudah mulai menjawab berdasarkan pengalaman.", "evidence": transcript[:160]}],
        "improvement_areas": [
            {
                "title": key.replace("_", " ").title(),
                "cause": "Skor komponen ini masih relatif rendah dibanding komponen lain.",
                "suggestion": "Tambahkan konteks, kontribusi pribadi, tools, dan hasil yang lebih konkret.",
            }
            for key, _ in low_components
        ],
        "before_after_answer_improvement": {
            "before": transcript[:500] if transcript else "Belum ada jawaban utama yang bisa dibandingkan.",
            "problem": "Jawaban masih perlu dibuat lebih terstruktur dan berbasis evidence.",
            "after": _fallback_stronger_answer(transcript),
            "why_better": "Versi after lebih jelas karena mengarahkan jawaban ke Situation, Task, Action, Result, dan evidence.",
        },
        "next_practice_recommendation": {
            "practice_type": practice_map.get(lowest, "Evidence Booster Practice"),
            "reason": f"Komponen terendah saat ini adalah {lowest.replace('_', ' ')}.",
            "focus": ["Struktur STAR", "Bukti konkret", "Dampak pengalaman"],
            "cta": "Ulangi latihan dengan satu pengalaman yang lebih spesifik.",
        },
    }
