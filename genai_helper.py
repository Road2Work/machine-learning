"""
genai_helper.py â€” Generative AI Integration (OpenAI GPT-5.4 mini)
Road2Work AI | CC26-PSU050

OpenAI Responses API digunakan untuk pertanyaan adaptif, evaluasi jawaban, dan feedback.
- Default GenAI model: gpt-5.4-mini.
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
    from openai import OpenAI
except Exception:  # pragma: no cover - environment lokal bisa belum install SDK
    OpenAI = None  # type: ignore


OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
OPENAI_MAX_OUTPUT_TOKENS: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "2048"))
MAX_RETRIES: int = int(os.getenv("GENAI_MAX_RETRIES", "3"))

client = None
if OpenAI is not None and OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:  # pragma: no cover
        print(f"[genai_helper] âš ï¸ Gagal inisialisasi OpenAI client: {exc}")


EVALUATION_WEIGHTS: dict[str, float] = {
    "role_relevance": 0.25,
    "star_structure": 0.20,
    "evidence_specificity": 0.20,
    "technical_accuracy": 0.15,
    "communication_clarity": 0.10,
    "self_awareness": 0.10,
}

_ALLOWED_CLARIFICATION_TYPES = {"tools", "impact", "contribution", "specificity", "context", "metric", "structure", "role_relevance", "technical", "clarity", "self_awareness", "professionalism", "unclear_audio", "weak_evidence", "missing_tools", "missing_impact", "missing_personal_contribution", "weak_star_structure", "low_role_relevance", "low_self_confidence", "weak_solution_skill", None, "null"}


def _call_openai(prompt: str) -> str:
    """Panggil OpenAI Responses API dengan retry. Jika API key belum ada, return marker aman."""
    if client is None:
        return "GENAI_UNAVAILABLE: OPENAI_API_KEY belum tersedia atau SDK openai belum terinstall."

    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict[str, Any] = {
                "model": OPENAI_MODEL,
                "input": prompt,
            }
            if OPENAI_MAX_OUTPUT_TOKENS > 0:
                kwargs["max_output_tokens"] = OPENAI_MAX_OUTPUT_TOKENS

            response = client.responses.create(**kwargs)
            return (getattr(response, "output_text", "") or "").strip()
        except Exception as e:  # pragma: no cover - tergantung API eksternal
            err = str(e)
            retryable = any(code in err for code in ["429", "500", "502", "503", "504", "rate_limit", "timeout"])
            if retryable and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(
                    f"âš ï¸ [genai_helper] OpenAI request sibuk/gagal sementara. "
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


def _looks_garbled_transcript(answer: str) -> bool:
    text = (answer or "").lower().strip()
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text)
    if len(words) < 3:
        return True
    common = {"saya", "aku", "pernah", "membuat", "menggunakan", "project", "proyek", "magang", "data", "role", "ai", "engineer", "backend", "frontend", "computer", "vision", "dashboard", "sql", "python", "java", "javascript", "model", "aplikasi", "tim", "hasil", "dampak", "belajar", "pengalaman"}
    known_ratio = sum(1 for word in words if word in common or len(word) <= 3) / max(1, len(words))
    very_long_unknown = sum(1 for word in words if len(word) > 14)
    repeated_noise = len(set(words)) <= max(2, len(words) // 4)
    return (len(words) >= 8 and known_ratio < 0.16) or very_long_unknown >= 2 or repeated_noise


def _fallback_evidence_level(answer: str) -> int:
    """Evidence Ladder heuristic: 1 claim â†’ 5 measurable result."""
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
    if _looks_garbled_transcript(answer):
        return ["unclear_audio"]
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
    priority = ["unclear_audio", "missing_tools", "tools", "missing_impact", "impact", "missing_personal_contribution", "contribution", "metric", "specificity"]
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
# 1. EXISTING HELPER â€” tetap dipertahankan untuk formalisasi profil
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
    raw_response = _call_openai(prompt)
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
    target_override = interview_state.get("target_competency_override")
    adaptive_memory = interview_state.get("adaptive_memory", {}) or {}
    weakness_history = interview_state.get("weakness_history", []) or []
    practice_mode = interview_state.get("practice_mode") or ("adaptive_from_history" if adaptive_memory.get("enabled") else "first_session")
    answer_memory = interview_state.get("answer_memory", {}) or {}

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
    if target_override:
        target_competency = str(target_override)
        competency_detail = {"competency": target_competency, "source": "adaptive_memory_or_session_state"}

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

Adaptive practice memory v2.3:
- Practice mode: {practice_mode}
- Previous weakness history: {weakness_history}
- Previous interview summary: {adaptive_memory.get("previous_interview_summary")}
- Latest interview feedback: {adaptive_memory.get("latest_interview_feedback")}
- Improvement focus: {adaptive_memory.get("improvement_focus")}
- Avoid repeated questions: {adaptive_memory.get("avoid_repeated_questions", True)}
- Retry mode: {adaptive_memory.get("retry_mode", False)}

Answer memory dari jawaban terakhir:
- Kutipan jawaban terakhir: {answer_memory.get("transcript_excerpt")}
- Tools/skill yang disebut: {answer_memory.get("mentioned_tools")}
- Anchor pengalaman yang bisa di-follow-up: {answer_memory.get("anchor_phrases")}
- Evidence level jawaban terakhir: {answer_memory.get("evidence_level")}
- Weakness jawaban terakhir: {answer_memory.get("detected_weaknesses")}

Aturan:
1. Jangan mengulang pertanyaan yang sudah ada secara sama persis.
2. Jika answer memory tersedia, buat follow-up dari detail yang benar-benar disebut kandidat. Contoh: jika kandidat menyebut Looker Studio, tanyakan keputusan, proses, validasi data, dampak, atau metrik dari penggunaan Looker Studio. Jangan pindah topik terlalu cepat.
3. Jika practice mode adaptive_from_history, prioritaskan weakness terbesar dan improvement focus.
4. Jika retry_mode false, jangan mengulang wording atau substansi pertanyaan lama. Kompetensi boleh sama, wording harus berbeda.
5. Jangan menanyakan hal yang terlalu jauh dari konteks kandidat.
6. Dorong kandidat menjawab dengan Evidence Ladder: skill/tools, konteks project, kontribusi pribadi, impact, lalu metrik jika ada.
7. Jangan mengulang substansi pertanyaan atau klarifikasi yang sudah ditanyakan, termasuk jika wording berbeda tetapi maksudnya sama.
8. Kalau pertanyaan sebelumnya sudah meminta tools, jangan lagi bertanya tools; naikkan ke impact/metrik atau kontribusi.
9. Bahasa Indonesia, profesional, singkat, seperti HRD sungguhan.
10. Kalau question seed tersedia, gunakan sebagai arah pertanyaan tetapi boleh diparafrase agar natural.

Balas HANYA JSON valid:
{{
  "question": "teks pertanyaan",
  "competency_target": "{target_competency}",
  "question_type": "main"
}}
"""
    raw = _call_openai(prompt)
    fallback_question = _fallback_answer_memory_question(role, answer_memory, asked) or seed_text or _fallback_natural_question(role, skills, target_competency, asked)
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


def _clean_follow_up_subject(value: str, *, allow_short_tool: bool = False) -> str | None:
    clean = re.sub(r"\s+", " ", str(value or "").strip(" .,:;!?\"'"))
    if not clean:
        return None
    lower = clean.lower()
    generic = {
        "dengan tim", "dalam tim", "sama tim", "ini sama", "ini sama sih", "ini juga",
        "ini juga sih", "pengalaman tadi", "jawaban tadi", "role", "posisi", "tim",
        "saya", "aku", "kami", "kita", "harus", "yang", "dan", "atau",
    }
    if lower in generic or any(lower.startswith(prefix) for prefix in ("dengan ", "sama ", "yang ", "untuk ")):
        return None
    if len(clean) < (2 if allow_short_tool else 6) or len(clean) > 80:
        return None
    stop_words = {"yang", "dan", "atau", "dengan", "untuk", "dalam", "pada", "saya", "aku", "kami", "kita", "itu", "ini"}
    words = re.findall(r"[a-zA-Z0-9.+#-]+", lower)
    if not allow_short_tool and words:
        meaningful = [word for word in words if word not in stop_words and len(word) > 2]
        if len(meaningful) == 0:
            return None
    return clean


def _is_repeated_question(candidate: str, asked: list[str]) -> bool:
    candidate_words = set(re.findall(r"[a-zA-Z0-9]+", candidate.lower()))
    for old in asked:
        old_words = set(re.findall(r"[a-zA-Z0-9]+", str(old).lower()))
        similarity = len(candidate_words & old_words) / max(1, len(candidate_words | old_words))
        if candidate.strip().lower() == str(old).strip().lower() or similarity > 0.74:
            return True
    return False


def _fallback_answer_memory_question(role: str, answer_memory: dict[str, Any], asked: list[str]) -> str | None:
    if not answer_memory:
        return None
    tools = []
    for item in answer_memory.get("mentioned_tools", []) or []:
        subject = _clean_follow_up_subject(str(item), allow_short_tool=True)
        if subject and subject.lower() not in {tool.lower() for tool in tools}:
            tools.append(subject)

    anchors = []
    for item in answer_memory.get("anchor_phrases", []) or []:
        subject = _clean_follow_up_subject(str(item))
        if subject and subject.lower() not in {anchor.lower() for anchor in anchors}:
            anchors.append(subject)

    weaknesses = [str(x).strip() for x in answer_memory.get("detected_weaknesses", []) or [] if str(x).strip()]
    evidence_level = int(answer_memory.get("evidence_level") or 0)
    tool_subject = tools[0] if tools else None
    anchor_subject = anchors[0] if anchors else None

    candidates: list[str] = []
    if tool_subject:
        if "missing_impact" in weaknesses or evidence_level >= 3:
            candidates.append(f"Tadi kamu menyebut {tool_subject}. Dari penggunaan itu, hasil apa yang berubah dan bagaimana kamu mengukurnya?")
        if "missing_personal_contribution" in weaknesses:
            candidates.append(f"Saat memakai {tool_subject}, bagian mana yang benar-benar kamu kerjakan sendiri dan keputusan apa yang kamu ambil?")
        candidates.append(f"Bisa jelaskan satu keputusan penting saat kamu memakai {tool_subject}, lalu dampaknya untuk project atau tim?")
    if anchor_subject:
        candidates.append(f"Dari pengalaman tentang {anchor_subject}, bagian mana yang paling menunjukkan kontribusi pribadimu dan hasil akhirnya?")
        candidates.append(f"Pada pengalaman {anchor_subject}, tantangan utamanya apa, aksi yang kamu ambil apa, dan perubahan apa yang terjadi setelahnya?")
    if "missing_tools" in weaknesses and not tool_subject:
        candidates.append("Dari pengalaman tadi, metode atau tools apa yang paling penting, dan kenapa itu kamu pilih?")
    candidates.append(f"Pilih satu pengalaman paling konkret untuk posisi {role}. Ceritakan konteksnya, kontribusimu, tools yang dipakai, dan hasil yang bisa dibuktikan.")

    for candidate in candidates:
        if not _is_repeated_question(candidate, asked):
            return candidate
    return None

def _fallback_natural_question(role: str, skills: list[str], competency: str, asked: list[str]) -> str:
    skills_text = ", ".join(skills[:4]) if skills else "skill yang kamu punya"
    options = {
        "role_relevance": f"Ceritakan pengalaman paling relevan yang menunjukkan kamu siap untuk posisi {role}.",
        "role_relevance_and_evidence": f"Ceritakan satu pengalaman yang paling membuktikan kesiapanmu untuk posisi {role}, termasuk tools, kontribusi pribadi, dan hasilnya.",
        "evidence_specificity": f"Bisa jelaskan satu project yang pernah kamu kerjakan dengan {skills_text}, termasuk konteks dan hasilnya?",
        "technical_accuracy": f"Pilih satu pengalaman teknis yang paling kuat. Bagaimana proses, tools, dan keputusan teknis yang kamu ambil?",
        "communication_clarity": "Coba jelaskan salah satu pengalamanmu secara runtut: situasinya apa, tugasmu apa, aksi yang kamu ambil, dan hasilnya.",
        "self_awareness": f"Menurut kamu, kekuatan dan area pengembanganmu untuk posisi {role} apa saja?",
        "self_introduction": f"Silakan perkenalkan diri kamu secara singkat dan jelaskan pengalaman yang paling relevan dengan role {role}.",
        "interest_need_of_learning": f"Apa yang membuat kamu tertarik pada posisi {role}, dan skill apa yang paling ingin kamu tingkatkan untuk role ini?",
        "self_confidence": f"Apa kekuatan utama kamu untuk posisi {role}, dan contoh pengalaman apa yang membuktikannya?",
        "skill": f"Tools, metode, atau skill apa yang pernah kamu gunakan dalam project yang relevan dengan posisi {role}?",
        "solution_skill": "Ceritakan situasi ketika kamu menyelesaikan masalah dalam project atau organisasi. Apa langkah yang kamu ambil dan hasilnya?",
        "predictive_based": "Ceritakan pengalaman ketika kamu memperkirakan risiko, dampak, atau hasil dari sebuah keputusan/solusi.",
        "predictive_based_recruitment": f"Bukti apa yang menunjukkan kamu siap menjalankan tanggung jawab posisi {role}?",
        "agile_culture": "Ceritakan pengalaman ketika kamu harus beradaptasi dengan perubahan cepat dalam tim atau project.",
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
# 3. ANSWER EVALUATION â€” FULL SCHEMA
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

Evidence Ladder wajib:
1 Claim saja: hanya klaim kemampuan.
2 Skill/tools: menyebut skill atau tools.
3 Context: menyebut project, organisasi, magang, perusahaan, atau masalah.
4 Impact: menjelaskan hasil/dampak kualitatif.
5 Measurable Result: ada angka, metrik, skala, waktu, persen, atau indikator terukur.

Aturan klarifikasi:
- Jika transkrip terdengar acak/garbled/tidak koheren, jangan nilai sebagai jawaban valid. Set need_clarification=true, clarification_type="unclear_audio", evidence_level=1.
- Kalau Evidence Level 1-2, klarifikasi harus menggali konteks atau tools.
- Kalau Evidence Level 3, klarifikasi harus menggali impact atau kontribusi pribadi.
- Kalau Evidence Level 4, klarifikasi harus menggali measurable result/metrik.
- Jangan meminta hal yang sudah dijawab kandidat.

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
  "clarification_type": "unclear_audio|tools|impact|contribution|specificity|metric|null",
  "feedback": "maksimal 3 kalimat",
  "stronger_answer": "versi jawaban lebih kuat, tanpa mengarang data"
}}
"""
    raw_response = _call_openai(prompt)

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

    if _looks_garbled_transcript(original_answer):
        return {
            "score_breakdown": _normalize_score_breakdown({}, 20),
            "final_score": 20,
            "evidence_level": 1,
            "weakness": ["unclear_audio"],
            "need_clarification": True,
            "clarification_type": "unclear_audio",
            "feedback": "Jawaban belum terdengar cukup jelas untuk dievaluasi.",
            "stronger_answer": "Ulangi jawaban dengan satu contoh pengalaman yang jelas: konteksnya apa, kontribusimu apa, tools yang dipakai, dan hasilnya.",
        }

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
        return "Saya pernah mengerjakan [nama project/pengalaman]. Konteksnya [situasi singkat], peran saya [kontribusi pribadi], tools yang saya gunakan [tools/skill], dan hasilnya [impact yang benar-benar terjadi]."
    concise = answer.strip().replace("\n", " ")[:220]
    return (
        f"{concise}. Untuk membuat jawaban ini naik di Evidence Ladder, jelaskan konteks project, "
        "kontribusi pribadi, tools/metode yang dipakai, lalu tutup dengan impact atau angka yang benar-benar terjadi."
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
Tujuannya menggali satu rung Evidence Ladder yang paling hilang, bukan menghakimi.
Jika audio/transkrip tidak jelas, minta kandidat mengulang jawaban dengan pelan dan konkret.
Jangan mengulang maksud pertanyaan awal. Jangan menambahkan asumsi pengalaman baru.

Balas HANYA teks pertanyaannya saja.
"""
    raw = _call_openai(prompt)
    if raw.startswith("GENAI_") or len(raw.strip()) < 8:
        return _fallback_clarification_question(clarification_type, role)
    return raw.strip().strip('"')


def _fallback_clarification_question(clarification_type: str | None, role: str) -> str:
    mapping = {
        "tools": "Bisa kamu jelaskan tools, metode, atau teknologi apa yang kamu gunakan dalam pengalaman itu?",
        "missing_tools": "Bisa kamu jelaskan tools, metode, atau teknologi apa yang kamu gunakan dalam pengalaman itu?",
        "impact": "Apa hasil atau dampak dari pekerjaan yang kamu lakukan dalam pengalaman tersebut?",
        "missing_impact": "Apa hasil atau dampak terukur dari pekerjaan yang kamu lakukan dalam pengalaman tersebut?",
        "contribution": "Bagian mana yang benar-benar kamu kerjakan sendiri, dan apa tanggung jawab utamamu?",
        "missing_personal_contribution": "Bagian mana yang benar-benar kamu kerjakan sendiri, dan apa tanggung jawab utamamu?",
        "specificity": "Bisa kamu ceritakan contoh yang lebih spesifik dari pengalaman itu?",
        "weak_evidence": "Bisa kamu ceritakan contoh yang lebih spesifik, termasuk konteks, aksi, dan bukti hasilnya?",
        "context": "Bisa jelaskan konteks proyek atau masalah yang sedang kamu hadapi saat itu?",
        "metric": "Apa hasil terukurnya? Kamu bisa sebutkan angka, waktu, skala data, akurasi, atau indikator lain yang benar-benar terjadi.",
        "structure": "Bisa ceritakan ulang secara runtut dari situasi, tugasmu, aksi yang kamu lakukan, sampai hasilnya?",
        "weak_star_structure": "Bisa kamu ceritakan ulang secara runtut dari situasi, tugasmu, aksi yang kamu lakukan, sampai hasilnya?",
        "role_relevance": f"Bagaimana pengalaman itu berhubungan langsung dengan posisi {role} yang kamu targetkan?",
        "low_role_relevance": f"Bagaimana pengalaman itu berhubungan langsung dengan posisi {role} yang kamu targetkan?",
        "technical": "Bisa jelaskan detail teknis atau pendekatan yang kamu gunakan?",
        "technical_accuracy": "Bisa jelaskan detail teknis, pendekatan, atau alasan pilihan solusi yang kamu gunakan?",
        "clarity": "Bisa jelaskan ulang dengan lebih terstruktur dan singkat?",
        "communication_clarity": "Bisa jelaskan ulang dengan lebih terstruktur dan singkat?",
        "self_awareness": "Apa pembelajaran atau hal yang akan kamu perbaiki dari pengalaman tersebut?",
        "low_self_confidence": "Apa bagian dari pengalaman itu yang paling kamu kuasai, dan apa yang masih ingin kamu tingkatkan?",
        "weak_solution_skill": "Bisa jelaskan langkah yang kamu ambil untuk menyelesaikan masalah dan alasan di balik keputusanmu?",
        "unclear_audio": "Maaf, jawabanmu belum tertangkap jelas. Bisa ulangi dengan lebih pelan, lalu ceritakan satu contoh pengalaman yang paling relevan?",
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
- Buat before-after untuk setiap jawaban utama yang punya transcript, maksimal 5 item.
- Gunakan Evidence Ladder: Level 1 claim, Level 2 skill/tools, Level 3 context, Level 4 impact, Level 5 measurable result.
- Next practice recommendation mapping dari kelemahan terbesar.

Balas HANYA JSON valid:
{{
  "strengths": [
    {{"title": "...", "reason": "...", "evidence": "kutipan/parafrase singkat dari jawaban"}}
  ],
  "improvement_areas": [
    {{"title": "...", "cause": "...", "suggestion": "..."}}
  ],
  "before_after_answer_improvement": [
    {{
      "question_text": "pertanyaan yang dijawab",
      "before": "jawaban awal user",
      "problem": "masalah utama berdasarkan Evidence Ladder",
      "after": "versi lebih kuat tanpa mengarang data",
      "why_better": "alasan singkat kenapa lebih kuat",
      "evidence_ladder_note": "Level jawaban saat ini dan cara naik level"
    }}
  ],
  "next_practice_recommendation": {{
    "practice_type": "Behavioral STAR Practice|Evidence Booster Practice|Technical Interview Practice|Answer Clarity Practice|Role Understanding Practice",
    "reason": "...",
    "focus": ["..."],
    "cta": "..."
  }}
}}
"""
    raw = _call_openai(prompt)
    fallback = _fallback_dashboard(role, answers, final_score)
    if raw.startswith("GENAI_"):
        return fallback

    parsed = _parse_json_response(raw, fallback)
    return normalize_dashboard_schema(parsed, fallback)


def normalize_dashboard_schema(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    strengths = _safe_list(raw.get("strengths"), fallback["strengths"])[:3]
    improvement_areas = _safe_list(raw.get("improvement_areas"), fallback["improvement_areas"])[:3]
    before_after = raw.get("before_after_answer_improvement")
    if isinstance(before_after, dict):
        before_after = [before_after]
    elif not isinstance(before_after, list):
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


def _fallback_before_after_items(answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for idx, answer in enumerate(answers[:5], start=1):
        transcript = str(answer.get("transcript") or "").strip()
        if not transcript:
            continue
        evaluation = answer.get("evaluation", {}) or {}
        breakdown = evaluation.get("score_breakdown") or {}
        lowest = "evidence_specificity"
        if isinstance(breakdown, dict) and breakdown:
            lowest = min(breakdown.items(), key=lambda item: _clamp_score(item[1]))[0]
        problem_map = {
            "star_structure": "Struktur jawaban belum runtut dari konteks, aksi, sampai hasil.",
            "evidence_specificity": "Bukti pengalaman masih perlu dibuat lebih spesifik.",
            "technical_accuracy": "Penjelasan teknis masih perlu diperjelas agar lebih kredibel.",
            "communication_clarity": "Alur cerita masih bisa dibuat lebih ringkas dan mudah diikuti.",
            "role_relevance": "Hubungan pengalaman dengan role target belum cukup terlihat.",
            "self_awareness": "Refleksi kekuatan dan area belajar masih perlu dipertegas.",
        }
        items.append({
            "question_text": answer.get("question_text") or answer.get("question") or f"Jawaban {idx}",
            "before": transcript[:700],
            "problem": problem_map.get(lowest, "Jawaban masih perlu dibuat lebih terstruktur dan berbasis evidence."),
            "after": evaluation.get("stronger_answer") or _fallback_stronger_answer(transcript),
            "why_better": "Versi ini lebih kuat karena menghubungkan konteks, kontribusi pribadi, tools/metode, dan hasil.",
            "evidence_ladder_note": "Naikkan jawaban minimal ke Level 4 dengan impact; jika punya angka, arahkan ke Level 5 measurable result.",
            "improvement_notes": ["Perjelas konteks", "Tunjukkan kontribusi pribadi", "Tambahkan impact atau hasil terukur"],
        })
    return items or [{
        "question_text": "Jawaban interview",
        "before": "Belum ada jawaban utama yang bisa dibandingkan.",
        "problem": "Data jawaban belum cukup untuk membuat perbaikan spesifik.",
        "after": _fallback_stronger_answer(""),
        "why_better": "Format ini membantu user menjawab dengan konteks, kontribusi, tools, dan hasil.",
        "evidence_ladder_note": "Targetkan Level 4-5 Evidence Ladder.",
        "improvement_notes": ["Tambahkan pengalaman konkret", "Sebutkan kontribusi pribadi", "Tutup dengan hasil"],
    }]


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
        "before_after_answer_improvement": _fallback_before_after_items(main_answers),
        "next_practice_recommendation": {
            "practice_type": practice_map.get(lowest, "Evidence Booster Practice"),
            "reason": f"Komponen terendah saat ini adalah {lowest.replace('_', ' ')}.",
            "focus": ["Struktur STAR", "Bukti konkret", "Dampak pengalaman"],
            "cta": "Ulangi latihan dengan satu pengalaman yang lebih spesifik.",
        },
    }









