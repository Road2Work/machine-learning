"""
main.py — FastAPI AI Service Road2Work.id

v2.3 — aligned with Road2Work Overview v2.3 Adaptive Session + API Contract v2.3

Backend ↔ FastAPI AI Service endpoints:
- POST /v1/profile/extract-cv              (v2.3 canonical)
- POST /v1/profile/extract-manual          (v2.3 canonical)
- POST /v1/role-fit/generate-ranking
- POST /v1/role-fit/calculate-score        (v2.3 canonical)
- POST /v1/interview/build-context         (supports adaptivePracticeMemory)
- POST /v1/interview/generate-question     (v2.3 canonical)
- POST /v1/stt/transcribe                  (90s, no silence auto-stop)
- POST /v1/interview/evaluate-answer
- POST /v1/interview/generate-clarification (v2.3 canonical)
- POST /v1/interview/generate-result
- POST /v1/model/predict-answer-quality
- POST /v1/dashboard/generate-summary

Backward-compatible aliases from v2.1 are still supported:
- POST /v1/context/extract-cv, /v1/context/extract-profile
- POST /v1/role-fit/score
- POST /v1/interview/next-question, /v1/interview/clarifying-question

Notes:
- Frontend tetap memanggil Express Backend. FastAPI menerima payload terkontrol dari Backend.
- STT mengikuti policy 90 detik: stop condition ditentukan frontend/backend (Mic Off atau timeout),
  FastAPI memvalidasi audio maksimal 90 detik dan tidak menerapkan silence auto-stop.
- Dataset TensorFlow tetap memakai dataset_train.csv, dataset_val.csv, dataset_test.csv dari Data Science.
- Adaptive interview antar session menggunakan practice memory dari Backend/PostgreSQL.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import io
import os
import re
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import pdfplumber
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ds_assets import (
    asset_status,
    get_competency_map as ds_get_competency_map,
    get_evidence_ladder_mapping,
    get_need_clarification_rule,
    get_question_seed as ds_get_question_seed,
    get_role_family_for_role,
    get_role_tree,
    get_scoring_rubric,
    get_target_roles,
    get_weakness_taxonomy,
    reload_all as reload_all_ds_assets,
)
from genai_helper import (
    EVALUATION_WEIGHTS,
    evaluate_interview_answer,
    formalize_narrative,
    generate_clarification_question,
    generate_natural_question,
    generate_result_dashboard,
    normalize_evaluation_schema,
)
from model_builder import (
    ROLE_LABELS,
    compute_role_fit_scores,
    evaluate_saved_model_detailed,
    evaluate_split_datasets,
    get_role_skill_matrix,
    manual_test_dataset_path,
    predict_answer_quality,
    reload_role_matrix,
    test_dataset_path,
)
from nlp_utils import (
    calculate_initial_evidence_score,
    clean_text,
    extract_evidence_signals,
    extract_experience_summary,
    extract_from_short_profile,
    extract_skills,
    normalize_skills,
    reload_taxonomy,
)
from stt_utils import get_model_info, proses_audio_ke_teks


# --------------------------------------------------------------------------- #
# CONSTANTS
# --------------------------------------------------------------------------- #
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}
MAX_MAIN_QUESTIONS = int(os.getenv("MAX_MAIN_QUESTIONS", "5"))
MIN_MAIN_QUESTIONS = int(os.getenv("MIN_MAIN_QUESTIONS", "3"))
MAX_CLARIFICATION_PER_MAIN = int(os.getenv("MAX_CLARIFICATION_PER_MAIN", "1"))
MAX_CLARIFICATION_PER_SESSION = int(os.getenv("MAX_CLARIFICATION_PER_SESSION", "3"))
MAX_AUDIO_DURATION_SECONDS = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "90"))
STT_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("STT_LOW_CONFIDENCE_THRESHOLD", "0.60"))

# Dev-only fallback context store. Express + PostgreSQL tetap source of truth.
contexts: dict[str, dict[str, Any]] = {}

CONTRACT_CLARIFICATION_TYPES = {
    "unclear_audio",
    "weak_evidence",
    "missing_tools",
    "missing_impact",
    "missing_personal_contribution",
    "weak_star_structure",
    "low_role_relevance",
    "low_confidence_answer",  # backward-compatible alias
    "low_self_confidence",
    "weak_solution_skill",
    "weak_learning_interest",
    "weak_agile_example",
}

DEFAULT_COMPETENCY_SEQUENCE = [
    "self_introduction",
    "interest_need_of_learning",
    "self_confidence",
    "skill",
    "solution_skill",
    "predictive_based",
    "predictive_based_recruitment",
    "agile_culture",
]


# --------------------------------------------------------------------------- #
# APP
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Road2Work FastAPI AI Service siap — API Contract v2.3 Adaptive Session")
    yield
    print("🛑 Road2Work FastAPI AI Service berhenti")


app = FastAPI(
    title="Road2Work.id AI Service",
    description="AI extraction, role fit, interview context, adaptive question, STT, evaluation, clarification, result generation, and TensorFlow inference.",
    version="2.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5000",
        "http://localhost:5173",
        "https://road2work.id",
        "https://api.road2work.id",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# SCHEMAS — tolerant to fullstack variations
# --------------------------------------------------------------------------- #
class RolePayload(BaseModel):
    id: str | None = None
    name: str | None = None
    role_name: str | None = None
    roleName: str | None = None
    role_family: str | None = None
    roleFamily: str | None = None
    core_skills: list[str] = Field(default_factory=list)
    coreSkills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)


class ProfilePayload(BaseModel):
    professional_summary: str | None = None
    professionalSummary: str | None = None
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    skill_evidence: list[dict[str, Any]] = Field(default_factory=list)
    skillEvidence: list[dict[str, Any]] = Field(default_factory=list)
    achievement_signals: list[str] = Field(default_factory=list)
    achievementSignals: list[str] = Field(default_factory=list)
    evidence_items: list[str] = Field(default_factory=list)
    evidenceItems: list[str] = Field(default_factory=list)
    evidence_score: int | None = None
    evidenceScore: int | None = None
    profile_completeness: int | None = None
    profileCompleteness: int | None = None


class ExtractManualProfileRequest(BaseModel):
    target_role: RolePayload | None = None
    target_role_id: str | None = None
    target_role_name: str | None = None
    most_relevant_experience: str | None = None
    skills_and_tools: str | None = None
    project_experience: str | None = None
    achievement_or_impact: str | None = None
    # legacy
    profile_text: str | None = None
    target_role_name_legacy: str | None = None
    target_role_legacy: str | None = None


class RoleFitRankingRequest(BaseModel):
    profile: ProfilePayload
    available_roles: list[RolePayload] = Field(default_factory=list)
    limit: int = Field(default=3, ge=1, le=10)


class RoleFitScoreRequest(BaseModel):
    profile: ProfilePayload
    selected_role: RolePayload


class BuildInterviewContextRequest(BaseModel):
    profile_id: str | None = None
    profile: ProfilePayload
    selected_role: RolePayload
    role_fit: dict[str, Any] = Field(default_factory=dict)
    practice_mode: str | None = None
    practiceMode: str | None = None
    adaptive_practice_memory: AdaptivePracticeMemoryPayload | None = None
    adaptivePracticeMemory: AdaptivePracticeMemoryPayload | None = None


class SessionStatePayload(BaseModel):
    current_question_index: int | None = None
    question_index: int | None = None
    question_count: int | None = None
    total_main_questions: int | None = None
    asked_questions: list[str] = Field(default_factory=list)
    detected_weaknesses: list[str] = Field(default_factory=list)
    first_question_required: bool = True
    competency_sequence: list[str] = Field(default_factory=list)
    clarification_count: int = 0
    max_clarification: int = MAX_CLARIFICATION_PER_SESSION
    current_main_question_clarification_count: int = 0
    current_state: str | None = None
    currentState: str | None = None
    practice_mode: str | None = None
    practiceMode: str | None = None


class InterviewContextPayload(BaseModel):
    summary: str | None = None
    strengths: list[str] = Field(default_factory=list)
    risk_areas: list[str] = Field(default_factory=list)
    recommended_competency_sequence: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    experience_summary: str | list[str] | None = None
    evidence_items: list[str] = Field(default_factory=list)
    profile_summary: str | None = None


class AskedQuestionHistoryItem(BaseModel):
    question_id: str | None = None
    questionId: str | None = None
    question_text: str | None = None
    questionText: str | None = None
    question_type: str | None = None
    questionType: str | None = None
    competency_target: str | None = None
    competencyTarget: str | None = None
    asked_at: str | None = None
    askedAt: str | None = None


class NextBestActionPayload(BaseModel):
    id: str | None = None
    title: str | None = None
    description: str | None = None
    impact_label: str | None = None
    impactLabel: str | None = None
    impact_score_text: str | None = None
    impactScoreText: str | None = None
    action_type: str | None = None
    actionType: str | None = None


class AdaptivePracticeMemoryPayload(BaseModel):
    enabled: bool = False
    previous_session_ids: list[str] = Field(default_factory=list)
    previousSessionIds: list[str] = Field(default_factory=list)
    previous_interview_summary: str | None = None
    previousInterviewSummary: str | None = None
    previous_score_breakdown: dict[str, Any] | None = None
    previousScoreBreakdown: dict[str, Any] | None = None
    previous_detected_weaknesses: list[str] = Field(default_factory=list)
    previousDetectedWeaknesses: list[str] = Field(default_factory=list)
    previous_evidence_levels: list[int] = Field(default_factory=list)
    previousEvidenceLevels: list[int] = Field(default_factory=list)
    asked_question_history: list[AskedQuestionHistoryItem] = Field(default_factory=list)
    askedQuestionHistory: list[AskedQuestionHistoryItem] = Field(default_factory=list)
    latest_interview_feedback: str | None = None
    latestInterviewFeedback: str | None = None
    next_best_actions: list[NextBestActionPayload] = Field(default_factory=list)
    nextBestActions: list[NextBestActionPayload] = Field(default_factory=list)
    improvement_focus: list[str] = Field(default_factory=list)
    improvementFocus: list[str] = Field(default_factory=list)
    avoid_repeated_questions: bool = True
    avoidRepeatedQuestions: bool | None = None
    retry_mode: bool = False
    retryMode: bool | None = None


class NextQuestionRequest(BaseModel):
    session_id: str | None = None
    selected_role: RolePayload | None = None
    target_role: RolePayload | None = None  # backward compatible
    interview_context: InterviewContextPayload | None = None
    session_state: SessionStatePayload = Field(default_factory=SessionStatePayload)
    question_seed: list[Any] | dict[str, Any] | None = None
    competency_map: list[Any] | dict[str, Any] | None = None
    adaptive_practice_memory: AdaptivePracticeMemoryPayload | None = None
    adaptivePracticeMemory: AdaptivePracticeMemoryPayload | None = None
    practice_mode: str | None = None
    practiceMode: str | None = None
    context_id: str | None = None


class QuestionPayload(BaseModel):
    id: str | None = None
    question_text: str | None = None
    question_type: str | None = None
    competency_target: str | None = None


class VoiceMetadataPayload(BaseModel):
    started_by: str | None = None
    startedBy: str | None = None
    stopped_by: str | None = None
    stoppedBy: str | None = None
    duration_seconds: float | None = None
    durationSeconds: float | None = None
    max_duration_seconds: int | None = None
    maxDurationSeconds: int | None = None
    silence_detected: bool | None = None
    silenceDetected: bool | None = None
    silence_duration_seconds: float | None = None
    silenceDurationSeconds: float | None = None
    audio_mime_type: str | None = None
    audioMimeType: str | None = None


class AnswerPayload(BaseModel):
    transcript_text: str = Field(..., min_length=1)
    stt_confidence: float | None = None
    voice_metadata: VoiceMetadataPayload | None = None
    voiceMetadata: VoiceMetadataPayload | None = None


class EvaluateAnswerRequest(BaseModel):
    question: QuestionPayload
    answer: AnswerPayload
    profile: ProfilePayload | None = None
    selected_role: RolePayload | None = None
    target_role: RolePayload | None = None
    session_state: SessionStatePayload = Field(default_factory=SessionStatePayload)
    # legacy optional
    session_id: str | None = None
    interview_context: InterviewContextPayload | None = None
    score_history: list[int] = Field(default_factory=list)


class ClarifyingQuestionRequest(BaseModel):
    question_text: str
    answer_text: str
    detected_weaknesses: list[str] = Field(default_factory=list)
    clarification_type: str | None = None
    selected_role: str | RolePayload | None = None
    target_role: str | RolePayload | None = None


class ResultAnswerPayload(BaseModel):
    question_text: str | None = None
    answer_text: str | None = None
    answer_score: int | None = None
    score_breakdown: dict[str, int] | None = None
    evidence_level: int | None = None
    detected_weaknesses: list[str] = Field(default_factory=list)
    stronger_answer: str | None = None
    feedback: str | None = None


class GenerateResultRequest(BaseModel):
    session_id: str | None = None
    selected_role: str | RolePayload | None = None
    target_role: str | RolePayload | None = None
    answers: list[ResultAnswerPayload] = Field(default_factory=list)


class DashboardSummaryRequest(BaseModel):
    career_readiness_score: int | None = None
    careerReadinessScore: int | None = None
    dashboard: dict[str, Any] = Field(default_factory=dict)
    user: dict[str, Any] = Field(default_factory=dict)
    selected_role: dict[str, Any] = Field(default_factory=dict)
    selectedRole: dict[str, Any] = Field(default_factory=dict)


class ModelPredictRequest(BaseModel):
    transcript_text: str | None = None
    answer_text: str | None = None
    answer: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    question: str | None = ""
    role: str | None = ""


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recording_policy(audio_format: str = "webm") -> dict[str, Any]:
    """RecordingPolicy v2.3. Legacy keys are included for older backend callers."""
    audio_format = (audio_format or "webm").lower().replace("audio/", "")
    if audio_format == "mpeg":
        audio_format = "mp3"
    if audio_format not in {"webm", "wav", "mp3"}:
        audio_format = "webm"
    return {
        # v2.3 canonical fields
        "autoStartMic": True,
        "autoStartTrigger": "after_hrd_question_finished",
        "answerLimitSeconds": MAX_AUDIO_DURATION_SECONDS,
        "silenceAutoStopEnabled": False,
        "userCanStopBeforeLimit": True,
        "stopReasons": ["user_mic_off", "timer_timeout"],
        "audioFormat": audio_format,
        # backward-compatible v2.1 fields
        "silenceAutoStop": False,
        "manualStopEnabled": True,
        "stopOptions": ["user_mic_off", "timer_timeout"],
    }


def _normalize_adaptive_memory(memory: AdaptivePracticeMemoryPayload | dict[str, Any] | None) -> dict[str, Any]:
    data = _model_dump(memory) if memory else {}
    asked_raw = data.get("asked_question_history") or data.get("askedQuestionHistory") or []
    asked_items: list[dict[str, Any]] = []
    for item in asked_raw:
        row = _model_dump(item)
        qtext = row.get("question_text") or row.get("questionText") or ""
        qid = row.get("question_id") or row.get("questionId")
        if qtext:
            asked_items.append({
                "question_id": qid,
                "question_text": qtext,
                "question_type": row.get("question_type") or row.get("questionType") or "main",
                "competency_target": row.get("competency_target") or row.get("competencyTarget"),
                "asked_at": row.get("asked_at") or row.get("askedAt"),
            })
    return {
        "enabled": bool(data.get("enabled", False)),
        "previous_session_ids": data.get("previous_session_ids") or data.get("previousSessionIds") or [],
        "previous_interview_summary": data.get("previous_interview_summary") or data.get("previousInterviewSummary"),
        "previous_score_breakdown": data.get("previous_score_breakdown") or data.get("previousScoreBreakdown"),
        "previous_detected_weaknesses": data.get("previous_detected_weaknesses") or data.get("previousDetectedWeaknesses") or [],
        "previous_evidence_levels": data.get("previous_evidence_levels") or data.get("previousEvidenceLevels") or [],
        "asked_question_history": asked_items,
        "latest_interview_feedback": data.get("latest_interview_feedback") or data.get("latestInterviewFeedback"),
        "next_best_actions": data.get("next_best_actions") or data.get("nextBestActions") or [],
        "improvement_focus": data.get("improvement_focus") or data.get("improvementFocus") or [],
        "avoid_repeated_questions": bool(data.get("avoid_repeated_questions", data.get("avoidRepeatedQuestions", True))),
        "retry_mode": bool(data.get("retry_mode", data.get("retryMode", False))),
    }


def _asked_question_texts(memory: dict[str, Any], session_state: dict[str, Any]) -> list[str]:
    asked = list(session_state.get("asked_questions") or [])
    for item in memory.get("asked_question_history", []) or []:
        q = item.get("question_text")
        if q:
            asked.append(q)
    return asked


def _simple_similarity(a: str, b: str) -> float:
    wa = {w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", (a or "").lower()).split() if len(w) > 2}
    wb = {w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", (b or "").lower()).split() if len(w) > 2}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa | wb))


def _find_repeated_question(question: str, memory: dict[str, Any]) -> dict[str, Any] | None:
    for item in memory.get("asked_question_history", []) or []:
        old = item.get("question_text", "")
        if old and (old.strip().lower() == question.strip().lower() or _simple_similarity(old, question) >= 0.88):
            return item
    return None


def _generated_from(memory: dict[str, Any]) -> str:
    if memory.get("retry_mode"):
        return "retry_focus"
    if memory.get("previous_detected_weaknesses") or memory.get("improvement_focus"):
        return "weakness_history"
    if memory.get("next_best_actions"):
        return "next_best_action"
    return "role_context"


def _focus_to_competency(focus: list[str]) -> str | None:
    text = " ".join(str(x).lower() for x in focus or [])
    if "star" in text or "structure" in text:
        return "role_relevance_and_evidence"
    if "evidence" in text or "impact" in text or "tools" in text:
        return "role_relevance_and_evidence"
    if "confidence" in text:
        return "self_confidence"
    if "solution" in text or "problem" in text:
        return "solution_skill"
    if "learning" in text or "interest" in text:
        return "interest_need_of_learning"
    if "agile" in text or "adapt" in text:
        return "agile_culture"
    if "technical" in text or "skill" in text:
        return "skill"
    return None


def _adaptive_fallback_question(role_name: str, competency_target: str, memory: dict[str, Any]) -> str:
    focus = memory.get("improvement_focus") or memory.get("previous_detected_weaknesses") or []
    focus_text = ", ".join(focus[:3]) if focus else "evidence dan struktur jawaban"
    if memory.get("retry_mode"):
        return f"Kita latihan ulang dengan fokus {focus_text}. Ceritakan satu pengalaman yang relevan untuk posisi {role_name}, lalu jelaskan konteks, kontribusi pribadi, tools, dan hasilnya."
    if competency_target == "self_confidence":
        return f"Apa kekuatan utama kamu untuk posisi {role_name}, dan bukti pengalaman apa yang mendukungnya?"
    if competency_target == "solution_skill":
        return f"Ceritakan satu masalah yang pernah kamu selesaikan. Jelaskan langkah solusi, keputusan yang kamu ambil, dan dampaknya."
    if competency_target == "agile_culture":
        return f"Ceritakan pengalaman ketika kamu harus beradaptasi dengan perubahan cepat dalam project atau tim."
    return f"Ceritakan satu pengalaman berbeda yang relevan untuk posisi {role_name}. Fokuskan pada {focus_text}, termasuk tools, kontribusi pribadi, dan dampak akhirnya."


def _model_dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip())
    return str(value).strip()


def _role_name(role: str | RolePayload | None, fallback: str = "posisi yang dipilih") -> str:
    if isinstance(role, str):
        return role.strip() or fallback
    if isinstance(role, RolePayload):
        return (role.name or role.role_name or role.roleName or role.id or fallback).strip()
    return fallback


def _role_id(role: RolePayload | None, fallback: str = "") -> str:
    if role and role.id:
        return role.id
    name = _role_name(role, fallback="")
    return name or fallback


def _profile_summary(profile: ProfilePayload | None) -> str:
    if not profile:
        return ""
    return profile.professional_summary or profile.professionalSummary or ""


def _profile_skills(profile: ProfilePayload | None) -> list[str]:
    return list(profile.skills or []) if profile else []


def _profile_tools(profile: ProfilePayload | None) -> list[str]:
    return list(profile.tools or []) if profile else []


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CV_INVALID_FORMAT: gagal membaca PDF: {exc}")
    if not text.strip():
        raise HTTPException(status_code=422, detail="CV_INVALID_FORMAT: tidak ada teks yang bisa diekstrak. Pastikan CV bukan scan gambar.")
    return text.strip()


def _validate_audio_file(audio: UploadFile) -> str:
    suffix = os.path.splitext(audio.filename or "")[-1].lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="AUDIO_INVALID_FORMAT")
    return suffix


def _skill_evidence_from_text(skills: list[str], raw_text: str, source: str) -> list[dict[str, Any]]:
    raw = raw_text.strip()
    signals = extract_evidence_signals(raw)
    level = 1
    if skills:
        level = 2
    if any(k in raw.lower() for k in ["project", "proyek", "magang", "tim", "kampus", "perusahaan", "dashboard", "model", "api"]):
        level = max(level, 3)
    if signals.get("impact_keywords"):
        level = max(level, 4)
    if signals.get("has_metric"):
        level = max(level, 5)
    evidence_text = raw[:220] or "Evidence belum tersedia."
    return [
        {
            "skill_name": skill,
            "evidence_text": evidence_text,
            "evidence_level": level,
            "source": source,
        }
        for skill in skills[:8]
    ]


def _achievement_signals(raw_text: str) -> list[str]:
    signals = []
    for sent in [s.strip() for s in raw_text.replace("\n", ". ").split(".") if s.strip()]:
        lower = sent.lower()
        if any(k in lower for k in ["meningkat", "mengurangi", "mempercepat", "akurasi", "efisiensi", "%", "persen", "hasil", "dampak"]):
            signals.append(sent[:180])
    return signals[:5]


def _profile_completeness(skills: list[str], tools: list[str], summary: str, evidence_items: list[Any]) -> int:
    score = 0
    if summary:
        score += 25
    if skills:
        score += min(25, len(skills) * 5)
    if tools:
        score += min(20, len(tools) * 5)
    if evidence_items:
        score += min(30, len(evidence_items) * 6)
    return max(0, min(100, score))


def _create_profile_context(raw_text: str, source: str, target_role: str | None = None, filename: str | None = None) -> dict[str, Any]:
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="PROFILE_CONTEXT_REQUIRED")
    role_for_formalize = target_role or "profesional"
    skills = normalize_skills(extract_skills(raw_text))
    tools = skills[:]  # MVP: taxonomy belum memisahkan skill/tool secara eksplisit
    professional_profile = formalize_narrative(raw_text, role=role_for_formalize)
    summary = professional_profile.get("professional_summary") or raw_text[:450]
    experience = extract_experience_summary(raw_text)
    evidence_items = experience if isinstance(experience, list) else [str(experience)]
    evidence_score = calculate_initial_evidence_score(raw_text, skills)
    skill_evidence = _skill_evidence_from_text(skills, raw_text, source)
    achievement_signals = _achievement_signals(raw_text)
    completeness = _profile_completeness(skills, tools, summary, skill_evidence or evidence_items)
    ai_confidence = max(50, min(95, int((evidence_score + completeness) / 2)))
    meta = get_role_family_for_role(target_role) if target_role else None
    context = {
        "context_id": str(uuid4()),
        "source": source,
        "filename": filename,
        "target_role": target_role,
        "domain": (meta or {}).get("domain"),
        "role_family": (meta or {}).get("role_family"),
        "raw_text": raw_text,
        "cleaned_text": clean_text(raw_text),
        "professional_summary": summary,
        "skills": skills,
        "tools": tools,
        "skill_evidence": skill_evidence,
        "achievement_signals": achievement_signals,
        "evidence_items": evidence_items,
        "evidence_score": evidence_score,
        "profile_completeness": completeness,
        "ai_confidence": ai_confidence,
        "created_at": _now_iso(),
    }
    contexts[context["context_id"]] = context
    return context


def _profile_response(context: dict[str, Any], include_context_id: bool = True) -> dict[str, Any]:
    resp = {
        "source": context.get("source"),
        "professional_summary": context.get("professional_summary", ""),
        "skills": context.get("skills", []),
        "tools": context.get("tools", []),
        "skill_evidence": context.get("skill_evidence", []),
        "achievement_signals": context.get("achievement_signals", []),
        "evidence_score": int(context.get("evidence_score", 0)),
        "profile_completeness": int(context.get("profile_completeness", 0)),
        "ai_confidence": int(context.get("ai_confidence", 0)),
    }
    if include_context_id:
        resp["context_id"] = context.get("context_id")
    return resp


def _normalize_score_breakdown(score_breakdown: dict[str, Any] | None) -> dict[str, int]:
    aliases = {
        "roleRelevance": "role_relevance",
        "starStructure": "star_structure",
        "evidenceSpecificity": "evidence_specificity",
        "technicalAccuracy": "technical_accuracy",
        "communicationClarity": "communication_clarity",
        "selfAwareness": "self_awareness",
    }
    score_breakdown = score_breakdown or {}
    normalized = {key: 0 for key in EVALUATION_WEIGHTS}
    for key, val in score_breakdown.items():
        k = aliases.get(key, key)
        if k in normalized:
            try:
                normalized[k] = max(0, min(100, int(round(float(val)))))
            except Exception:
                normalized[k] = 0
    return normalized


def _weighted_score(breakdown: dict[str, int]) -> int:
    total = 0.0
    for key, weight in EVALUATION_WEIGHTS.items():
        total += breakdown.get(key, 0) * weight
    return max(0, min(100, int(round(total))))


def _map_weakness(tag: str) -> str:
    tag = str(tag).strip().lower()
    mapping = {
        "tools": "missing_tools", "tools_missing": "missing_tools", "missing_tools": "missing_tools",
        "impact": "missing_impact", "impact_missing": "missing_impact", "missing_impact": "missing_impact",
        "contribution": "missing_personal_contribution", "contribution_unclear": "missing_personal_contribution",
        "specificity": "weak_evidence", "evidence_specificity": "weak_evidence", "weak_evidence": "weak_evidence", "metric": "weak_evidence", "measurable_result_missing": "weak_evidence",
        "star_structure": "weak_star_structure", "star_structure_missing": "weak_star_structure", "weak_star_structure": "weak_star_structure",
        "role_relevance": "low_role_relevance", "role_relevance_low": "low_role_relevance", "low_role_relevance": "low_role_relevance",
        "unclear_audio": "unclear_audio",
        "self_confidence": "low_confidence_answer",
        "learning_interest": "weak_learning_interest",
        "agile_culture": "weak_agile_example",
    }
    return mapping.get(tag, tag)


def _contract_weaknesses(tags: list[Any]) -> list[str]:
    result = []
    for tag in tags or []:
        mapped = _map_weakness(str(tag))
        if mapped and mapped not in result:
            result.append(mapped)
    return result


def _contract_clarification_type(value: str | None, weaknesses: list[str]) -> str | None:
    if value and value != "null":
        mapped = _map_weakness(value)
        if mapped in CONTRACT_CLARIFICATION_TYPES:
            return mapped
    for weakness in weaknesses:
        mapped = _map_weakness(weakness)
        if mapped in CONTRACT_CLARIFICATION_TYPES:
            return mapped
    return None


def _status_id(final_score: int) -> str:
    if final_score >= 85:
        return "Siap melamar"
    if final_score >= 70:
        return "Hampir siap"
    if final_score >= 50:
        return "Mulai siap"
    return "Belum siap"


def _practice_type_from_breakdown(breakdown: dict[str, int]) -> str:
    if not breakdown:
        return "Evidence Booster Practice"
    lowest = min(breakdown, key=breakdown.get)
    return {
        "star_structure": "Behavioral STAR Practice",
        "evidence_specificity": "Evidence Booster Practice",
        "technical_accuracy": "Technical Interview Practice",
        "communication_clarity": "Answer Clarity Practice",
        "role_relevance": "Role Understanding Practice",
        "self_awareness": "Reflection Practice",
    }.get(lowest, "Evidence Booster Practice")


def _role_fit_against_role(skills: list[str], selected_role: RolePayload) -> dict[str, Any]:
    role_name = _role_name(selected_role)
    matrix = get_role_skill_matrix()
    required = matrix.get(role_name, [])
    normalized = {s.lower() for s in skills}
    required_lower = [s.lower() for s in required]
    matched = sorted(normalized.intersection(required_lower))
    missing = sorted(set(required_lower).difference(normalized))
    score = int(round((len(matched) / len(required_lower)) * 100)) if required_lower else 0
    return {
        "role_id": _role_id(selected_role, role_name),
        "role_name": role_name,
        "fit_score": score,
        "reason": f"{len(matched)} dari {len(required_lower)} skill inti {role_name} tercermin di profil user." if required_lower else "Role-skill matrix belum tersedia untuk role ini.",
        "strengths": matched[:5],
        "gaps": missing[:5],
        "skill_overlap": {
            "matched": len(matched),
            "total": len(required_lower),
            "matched_skills": matched,
            "missing_skills": missing,
        },
    }


# --------------------------------------------------------------------------- #
# GENERAL / DEV ENDPOINTS
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["General"])
async def health_check():
    return {
        "status": "ok",
        "service": "Road2Work.id AI Service",
        "version": app.version,
        "recording_policy": _recording_policy(),
        "stt": get_model_info(),
        "ds_assets": asset_status(),
    }


@app.get("/v1/roles/tree", tags=["General"])
async def roles_tree():
    return {"status": "success", "data": get_role_tree()}


@app.get("/v1/roles", tags=["General"])
async def roles_list():
    return {"status": "success", "data": {"roles": get_target_roles() or ROLE_LABELS, "role_tree": get_role_tree()}}


# --------------------------------------------------------------------------- #
# 9.1 AI EXTRACT CV PROFILE — Upload CV path, role may be unknown
# --------------------------------------------------------------------------- #
@app.post("/v1/profile/extract-cv", tags=["Profile Extraction"])
@app.post("/v1/context/extract-cv", tags=["Context", "Backward Compatible"])
async def extract_cv_profile(
    cvFile: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    upload = cvFile or file
    if upload is None:
        raise HTTPException(status_code=400, detail="cvFile wajib dikirim.")
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="CV_INVALID_FORMAT")
    contents = await upload.read()
    raw_text = _extract_text_from_pdf(contents)
    context = _create_profile_context(raw_text=raw_text, source="cv", target_role=None, filename=upload.filename)
    return _profile_response(context)


# --------------------------------------------------------------------------- #
# 9.2 AI EXTRACT MANUAL PROFILE — Manual path already has selected role
# --------------------------------------------------------------------------- #
@app.post("/v1/profile/extract-manual", tags=["Profile Extraction"])
@app.post("/v1/context/extract-profile", tags=["Context", "Backward Compatible"])
async def extract_manual_profile(body: ExtractManualProfileRequest):
    role = body.target_role or RolePayload(id=body.target_role_id, name=body.target_role_name or body.target_role_legacy or body.target_role_name_legacy)
    role_name = _role_name(role, fallback="")
    if not role_name:
        raise HTTPException(status_code=400, detail="target_role wajib dikirim untuk manual path.")
    raw_text = body.profile_text or "\n".join(
        str(part).strip()
        for part in [body.most_relevant_experience, body.skills_and_tools, body.project_experience, body.achievement_or_impact]
        if part and str(part).strip()
    )
    try:
        extracted = extract_from_short_profile(raw_text, target_role=role_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    context = _create_profile_context(raw_text=extracted["raw_text"], source="manual", target_role=role_name)
    context.update({
        "skills": extracted.get("skills", context["skills"]),
        "tools": extracted.get("tools", context["tools"]),
        "evidence_score": extracted.get("initial_evidence_score", context["evidence_score"]),
        "professional_summary": extracted.get("profile_summary") or context["professional_summary"],
    })
    return _profile_response(context)


# --------------------------------------------------------------------------- #
# 9.3 AI GENERATE ROLE FIT RANKING — CV path only, but AI service stateless
# --------------------------------------------------------------------------- #
@app.post("/v1/role-fit/generate-ranking", tags=["Role Fit"])
async def generate_role_fit_ranking(body: RoleFitRankingRequest):
    skills = normalize_skills(_profile_skills(body.profile) + _profile_tools(body.profile))
    available = body.available_roles or []
    if available:
        recommendations = [_role_fit_against_role(skills, role) for role in available]
        recommendations = sorted(recommendations, key=lambda x: x["fit_score"], reverse=True)[: body.limit]
    else:
        recommendations = compute_role_fit_scores(skills, top_n=body.limit)
        recommendations = [
            {
                "role_id": rec.get("role"),
                "role_name": rec.get("role"),
                "fit_score": rec.get("score", 0),
                "reason": f"Skill yang cocok: {', '.join(rec.get('matched_skills', [])[:5]) or 'belum ada'}.",
                "strengths": rec.get("matched_skills", [])[:5],
                "gaps": rec.get("missing_skills", [])[:5],
                "skill_overlap": {
                    "matched": len(rec.get("matched_skills", [])),
                    "total": len(rec.get("matched_skills", [])) + len(rec.get("missing_skills", [])),
                    "matched_skills": rec.get("matched_skills", []),
                    "missing_skills": rec.get("missing_skills", []),
                },
            }
            for rec in recommendations
        ]
    for idx, rec in enumerate(recommendations, start=1):
        rec["rank"] = idx
    return {"recommended_roles": recommendations}


# --------------------------------------------------------------------------- #
# 9.4 AI CALCULATE ROLE FIT SCORE — CV and manual path
# --------------------------------------------------------------------------- #
@app.post("/v1/role-fit/calculate-score", tags=["Role Fit"])
@app.post("/v1/role-fit/score", tags=["Role Fit", "Backward Compatible"])
async def role_fit_score(body: RoleFitScoreRequest):
    skills = normalize_skills(_profile_skills(body.profile) + _profile_tools(body.profile))
    result = _role_fit_against_role(skills, body.selected_role)
    return result


# --------------------------------------------------------------------------- #
# 9.5 AI BUILD PERSONALIZED INTERVIEW CONTEXT
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/build-context", tags=["Interview Engine"])
async def build_interview_context(body: BuildInterviewContextRequest):
    role_name = _role_name(body.selected_role)
    skills = _profile_skills(body.profile)
    tools = _profile_tools(body.profile)
    gaps = body.role_fit.get("gaps", []) or body.role_fit.get("skill_overlap", {}).get("missing_skills", []) or []
    memory = _normalize_adaptive_memory(body.adaptive_practice_memory or body.adaptivePracticeMemory)
    practice_mode = body.practice_mode or body.practiceMode or ("adaptive_from_history" if memory.get("enabled") else "first_session")

    sequence = list(DEFAULT_COMPETENCY_SEQUENCE[:5])
    focus_competency = _focus_to_competency(memory.get("improvement_focus") or memory.get("previous_detected_weaknesses") or [])
    if focus_competency and focus_competency in sequence:
        sequence = ["self_introduction", focus_competency] + [c for c in sequence if c not in {"self_introduction", focus_competency}]

    summary = _profile_summary(body.profile) or f"Candidate has profile context for {role_name}."
    risk_areas = gaps[:5] or ["STAR Structure", "Evidence Specificity"]
    if memory.get("previous_detected_weaknesses"):
        risk_areas = list(dict.fromkeys(list(memory.get("previous_detected_weaknesses", [])) + risk_areas))[:8]

    return {
        "interview_context": {
            "summary": summary,
            "strengths": (skills + tools)[:5],
            "risk_areas": risk_areas,
            "recommended_competency_sequence": sequence,
            "previous_interview_summary": memory.get("previous_interview_summary"),
            "latest_interview_feedback": memory.get("latest_interview_feedback"),
        },
        "practice_mode": practice_mode,
        "adaptive_memory": memory,
        "recording_policy": _recording_policy(),
    }


# --------------------------------------------------------------------------- #
# 9.6 AI GENERATE INTERVIEW QUESTION — first question must be introduction
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/generate-question", tags=["Interview Engine"])
@app.post("/v1/interview/next-question", tags=["Interview Engine", "Backward Compatible"])
async def next_question(body: NextQuestionRequest):
    if body.context_id and body.context_id in contexts:
        ctx = contexts[body.context_id]
        role_name = ctx.get("target_role") or "posisi yang dipilih"
        interview_context = {
            "summary": ctx.get("professional_summary"),
            "skills": ctx.get("skills", []),
            "tools": ctx.get("tools", []),
            "evidence_items": ctx.get("evidence_items", []),
        }
    else:
        role_payload = body.selected_role or body.target_role or RolePayload(name="posisi yang dipilih")
        role_name = _role_name(role_payload)
        interview_context = _model_dump(body.interview_context)

    state = _model_dump(body.session_state)
    memory = _normalize_adaptive_memory(body.adaptive_practice_memory or body.adaptivePracticeMemory)
    practice_mode = body.practice_mode or body.practiceMode or state.get("practice_mode") or state.get("practiceMode") or ("adaptive_from_history" if memory.get("enabled") else "first_session")
    question_order = int(state.get("current_question_index") or state.get("question_index") or 1)
    total_questions = int(state.get("question_count") or state.get("total_main_questions") or MAX_MAIN_QUESTIONS)
    total_questions = max(MIN_MAIN_QUESTIONS, min(MAX_MAIN_QUESTIONS, total_questions))
    asked_questions = _asked_question_texts(memory, state)
    first_required = bool(state.get("first_question_required", True))
    generated_from = _generated_from(memory)

    if first_required and question_order <= 1 and not (state.get("asked_questions") or []):
        return {
            "question_text": f"Silakan perkenalkan diri kamu secara singkat dan jelaskan pengalaman yang paling relevan dengan role {role_name}.",
            "question_type": "main",
            "parent_question_id": None,
            "competency_target": "self_introduction",
            "clarification_type": None,
            "question_order": 1,
            "generated_from": "role_context",
            "repeated_from_question_id": None,
            "hrd_state": "asking",
            "reason": "Pertanyaan pertama wajib self introduction/perkenalan diri.",
            "recording_policy": _recording_policy(),
        }

    sequence = state.get("competency_sequence") or (interview_context.get("recommended_competency_sequence") if isinstance(interview_context, dict) else None) or DEFAULT_COMPETENCY_SEQUENCE
    focus_competency = _focus_to_competency(memory.get("improvement_focus") or memory.get("previous_detected_weaknesses") or [])
    if focus_competency and question_order > 1:
        competency_target = focus_competency
    else:
        competency_target = sequence[(question_order - 1) % len(sequence)] if sequence else "role_relevance_and_evidence"

    gen_state = {
        "main_question_index": max(0, question_order - 1),
        "asked_questions": asked_questions,
        "weakness_history": list(dict.fromkeys((state.get("detected_weaknesses", []) or []) + (memory.get("previous_detected_weaknesses", []) or []))),
        "target_competency_override": competency_target,
        "adaptive_memory": memory,
        "practice_mode": practice_mode,
    }
    competency_map = body.competency_map if body.competency_map else ds_get_competency_map()
    question_seed = body.question_seed if body.question_seed else ds_get_question_seed()
    generated = generate_natural_question(
        role=role_name,
        interview_context=interview_context if isinstance(interview_context, dict) else {},
        interview_state=gen_state,
        role_skill_matrix=get_role_skill_matrix(),
        competency_map=competency_map if isinstance(competency_map, dict) else {role_name: {"competencies": competency_map}},
        question_seed=question_seed if isinstance(question_seed, dict) else {role_name: question_seed},
    )
    question_text = generated.get("question") or _adaptive_fallback_question(role_name, competency_target, memory)

    repeated = _find_repeated_question(question_text, memory)
    if repeated and memory.get("avoid_repeated_questions", True) and not memory.get("retry_mode", False):
        question_text = _adaptive_fallback_question(role_name, competency_target, memory)
        repeated = _find_repeated_question(question_text, memory)
        if repeated:
            # Last safety rewrite to block exact repetition.
            question_text = f"Berikan contoh pengalaman lain untuk posisi {role_name} yang belum kamu ceritakan sebelumnya. Jelaskan konteks, aksi, tools, kontribusi pribadi, dan hasilnya."
            repeated = None

    return {
        "question_text": question_text,
        "question_type": "main",
        "parent_question_id": None,
        "competency_target": str(generated.get("competency_target") or competency_target),
        "clarification_type": None,
        "question_order": question_order,
        "generated_from": generated_from,
        "repeated_from_question_id": repeated.get("question_id") if repeated and memory.get("retry_mode") else None,
        "hrd_state": "asking",
        "reason": "Pertanyaan diarahkan dari practice memory." if generated_from != "role_context" else "Pertanyaan diarahkan dari role context dan interview state.",
        "practice_mode": practice_mode,
        "recording_policy": _recording_policy(),
    }


# --------------------------------------------------------------------------- #
# 9.7 AI SPEECH-TO-TEXT — maxDurationSeconds 90, no silence auto-stop
# --------------------------------------------------------------------------- #
@app.post("/v1/stt/transcribe", tags=["Speech-to-Text"])
async def transcribe_audio(
    audioFile: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    language: str | None = Form("id-ID"),
    maxDurationSeconds: int | None = Form(None),
    maxDurationSec: int | None = Form(None),
    silenceAutoStopEnabled: bool | None = Form(False),
    audioFormat: str | None = Form(None),
):
    upload = audioFile or audio
    if upload is None:
        raise HTTPException(status_code=400, detail="audioFile wajib dikirim.")
    if bool(silenceAutoStopEnabled):
        raise HTTPException(status_code=400, detail="SILENCE_AUTOSTOP_NOT_ALLOWED")
    max_duration = int(maxDurationSec or maxDurationSeconds or MAX_AUDIO_DURATION_SECONDS)
    if max_duration > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(status_code=400, detail="AUDIO_TOO_LONG")
    suffix = _validate_audio_file(upload)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name
        stt_result = proses_audio_ke_teks(tmp_path, language=language, max_duration_seconds=max_duration)
        if stt_result.get("status") == "error":
            code = stt_result.get("kode") or "STT_FAILED"
            status_code = 400 if code == "AUDIO_TOO_LONG" else 422
            raise HTTPException(status_code=status_code, detail=code if code == "AUDIO_TOO_LONG" else stt_result.get("pesan"))
        transcript = stt_result.get("data_transkrip") or ""
        confidence = stt_result.get("confidence")
        if confidence is None:
            confidence = 0.94 if len(transcript.split()) >= 8 else (0.80 if len(transcript.split()) >= 4 else 0.50)
        confidence = round(float(confidence), 2)
        needs_clarification = confidence < STT_LOW_CONFIDENCE_THRESHOLD or len(transcript.split()) < 3
        return {
            "status": "success",
            "transcript_text": transcript,
            "stt_confidence": confidence,
            "duration_seconds": stt_result.get("durasi_audio_detik"),
            "silence_detected": bool(stt_result.get("silence_detected", False)),
            "silence_duration_seconds": float(stt_result.get("silence_duration_seconds", 0) or 0),
            "needs_clarification": needs_clarification,
            "clarification_type": "unclear_audio" if needs_clarification else None,
            "recording_policy": _recording_policy(audioFormat or suffix.replace(".", "")),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# 9.8 AI EVALUATE ANSWER
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/evaluate-answer", tags=["Interview Engine"])
async def evaluate_answer(body: EvaluateAnswerRequest):
    question_text = body.question.question_text or ""
    question_type = body.question.question_type or "main"
    competency_target = body.question.competency_target
    transcript = body.answer.transcript_text
    stt_confidence = body.answer.stt_confidence
    voice_meta = body.answer.voice_metadata or body.answer.voiceMetadata
    role_payload = body.selected_role or body.target_role
    role_name = _role_name(role_payload)
    if not question_text:
        raise HTTPException(status_code=400, detail="question.question_text wajib dikirim.")
    if not transcript:
        raise HTTPException(status_code=400, detail="answer.transcript_text wajib dikirim.")

    profile_context = _model_dump(body.profile)
    raw_eval = evaluate_interview_answer(
        question=question_text,
        answer=transcript,
        role=role_name,
        interview_context=profile_context,
        competency_target=competency_target,
    )
    evaluation = normalize_evaluation_schema(raw_eval, original_answer=transcript)
    breakdown = _normalize_score_breakdown(evaluation.get("score_breakdown"))
    answer_score = int(evaluation.get("final_score") or _weighted_score(breakdown))
    weaknesses = _contract_weaknesses(evaluation.get("weakness", []))
    clarification_type = _contract_clarification_type(evaluation.get("clarification_type"), weaknesses)
    needs_clarification = bool(evaluation.get("need_clarification", False))

    state = _model_dump(body.session_state)
    if int(state.get("clarification_count") or 0) >= int(state.get("max_clarification") or MAX_CLARIFICATION_PER_SESSION):
        needs_clarification = False
        clarification_type = None
    if stt_confidence is not None and stt_confidence < STT_LOW_CONFIDENCE_THRESHOLD:
        needs_clarification = True
        clarification_type = "unclear_audio"
        if "unclear_audio" not in weaknesses:
            weaknesses.insert(0, "unclear_audio")
    if needs_clarification and not clarification_type:
        clarification_type = _contract_clarification_type(None, weaknesses) or "weak_evidence"

    model_support = predict_answer_quality(question=question_text, answer=transcript, role=role_name)
    response = {
        "id": str(uuid4()),
        "session_id": body.session_id,
        "question_id": body.question.id,
        "question_type": question_type,
        "transcript_text": transcript,
        "stt_confidence": stt_confidence,
        "score_breakdown": breakdown,
        "answer_score": answer_score,
        "evidence_level": max(1, min(5, int(evaluation.get("evidence_level", 1)))),
        "detected_weaknesses": weaknesses,
        "needs_clarification": needs_clarification,
        "clarification_type": clarification_type if needs_clarification else None,
        "feedback": str(evaluation.get("feedback", "")),
        "stronger_answer": str(evaluation.get("stronger_answer", "")),
        "model_support": {
            "label": str(model_support.get("label", "Average")).lower(),
            "predicted_quality": str(model_support.get("label", "Average")),
            "confidence": model_support.get("confidence"),
            "supporting_score": model_support.get("supporting_readiness_score"),
        },
        "created_at": _now_iso(),
    }
    if voice_meta:
        vm = _model_dump(voice_meta)
        response["voice_metadata"] = {
            "started_by": vm.get("started_by") or vm.get("startedBy") or "system_auto_after_question",
            "stopped_by": vm.get("stopped_by") or vm.get("stoppedBy") or "user_mic_off",
            "duration_seconds": vm.get("duration_seconds") or vm.get("durationSeconds"),
            "max_duration_seconds": vm.get("max_duration_seconds") or vm.get("maxDurationSeconds") or MAX_AUDIO_DURATION_SECONDS,
            "silence_detected": vm.get("silence_detected") or vm.get("silenceDetected") or False,
            "silence_duration_seconds": vm.get("silence_duration_seconds") or vm.get("silenceDurationSeconds") or 0,
            "audio_mime_type": vm.get("audio_mime_type") or vm.get("audioMimeType"),
        }
    return response


# --------------------------------------------------------------------------- #
# 9.9 AI GENERATE CLARIFYING QUESTION
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/generate-clarification", tags=["Interview Engine"])
@app.post("/v1/interview/clarifying-question", tags=["Interview Engine", "Backward Compatible"])
async def clarifying_question(body: ClarifyingQuestionRequest):
    role_name = _role_name(body.selected_role or body.target_role, fallback="posisi yang dipilih")
    clarification_type = _contract_clarification_type(body.clarification_type, body.detected_weaknesses) or "weak_evidence"
    q = generate_clarification_question(
        original_question=body.question_text,
        transcript=body.answer_text,
        weakness_tags=body.detected_weaknesses,
        role=role_name,
        clarification_type=clarification_type,
    )
    return {
        "question_text": q,
        "question_type": "clarification",
        "parent_question_id": None,
        "clarification_type": clarification_type,
        "competency_target": "role_relevance_and_evidence",
        "generated_from": "weakness_history",
        "repeated_from_question_id": None,
        "hrd_state": "clarifying",
        "recording_policy": _recording_policy(),
    }


# --------------------------------------------------------------------------- #
# 9.10 AI GENERATE INTERVIEW RESULT
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/generate-result", tags=["Interview Engine"])
async def generate_interview_result(body: GenerateResultRequest):
    role_name = _role_name(body.selected_role or body.target_role)
    normalized_answers: list[dict[str, Any]] = []
    for ans in body.answers:
        data = _model_dump(ans)
        breakdown = _normalize_score_breakdown(data.get("score_breakdown"))
        score = int(data.get("answer_score") or _weighted_score(breakdown))
        normalized_answers.append({
            "question_text": data.get("question_text"),
            "answer_text": data.get("answer_text"),
            "transcript": data.get("answer_text"),
            "answer_score": score,
            "score_breakdown": breakdown,
            "evidence_level": int(data.get("evidence_level") or 1),
            "detected_weaknesses": data.get("detected_weaknesses") or [],
            "stronger_answer": data.get("stronger_answer") or "",
            "feedback": data.get("feedback") or "",
            "evaluation": {
                "score_breakdown": breakdown,
                "final_score": score,
                "evidence_level": int(data.get("evidence_level") or 1),
                "weakness": data.get("detected_weaknesses") or [],
                "stronger_answer": data.get("stronger_answer") or "",
                "feedback": data.get("feedback") or "",
            },
        })
    if normalized_answers:
        interview_score = int(round(sum(a["answer_score"] for a in normalized_answers) / len(normalized_answers)))
        avg_evidence = int(round(sum(a["evidence_level"] for a in normalized_answers) / len(normalized_answers)))
    else:
        interview_score = 0
        avg_evidence = 1

    dashboard = generate_result_dashboard(role=role_name, interview_context={}, answers=normalized_answers, final_score=interview_score)
    strengths_raw = dashboard.get("strengths") or []
    improvements_raw = dashboard.get("improvement_areas") or []
    strengths = [str(x.get("title") or x.get("description") or x) if isinstance(x, dict) else str(x) for x in strengths_raw[:3]] or ["Komunikasi cukup jelas"]
    improvements = [str(x.get("title") or x.get("description") or x) if isinstance(x, dict) else str(x) for x in improvements_raw[:3]] or ["Gunakan struktur STAR dan evidence yang lebih spesifik"]
    before_after = dashboard.get("before_after_answer_improvement") or dashboard.get("before_after_improvement") or []
    avg_breakdown = {k: 0 for k in EVALUATION_WEIGHTS}
    if normalized_answers:
        for k in avg_breakdown:
            avg_breakdown[k] = int(round(sum(a["score_breakdown"].get(k, 0) for a in normalized_answers) / len(normalized_answers)))
    next_raw = dashboard.get("next_practice_recommendation") or {}
    next_practice = {
        "practice_type": next_raw.get("practice_type") or _practice_type_from_breakdown(avg_breakdown),
        "reason": next_raw.get("reason") or "Area terendah perlu dilatih ulang agar jawaban lebih siap untuk interview.",
        "focus_areas": next_raw.get("focus_areas") or next_raw.get("focusAreas") or improvements,
    }
    return {
        "interview_readiness_score": interview_score,
        "readiness_status": _status_id(interview_score),
        "summary": dashboard.get("summary") or "Interview selesai. Jawaban sudah dievaluasi berdasarkan relevansi role, STAR, evidence, akurasi teknis, komunikasi, dan self-awareness.",
        "evidence_level": max(1, min(5, avg_evidence)),
        "score_breakdown": avg_breakdown,
        "strengths": strengths,
        "improvement_areas": improvements,
        "before_after_improvement": before_after if isinstance(before_after, list) else [],
        "next_practice_recommendation": next_practice,
        "adaptive_session_suggestion": {
            "recommended_focus": next_practice.get("focus_areas", []),
            "avoid_repeated_questions": True,
            "suggested_practice_mode": "adaptive_from_history",
        },
    }


# --------------------------------------------------------------------------- #
# 9.11 AI PREDICT ANSWER QUALITY — TensorFlow supporting model
# --------------------------------------------------------------------------- #
@app.post("/v1/model/predict-answer-quality", tags=["Model"])
async def predict_answer_quality_endpoint(body: ModelPredictRequest):
    transcript = body.transcript_text or body.answer_text or body.answer
    if not transcript:
        raise HTTPException(status_code=400, detail="transcript_text wajib dikirim.")
    result = predict_answer_quality(
        question=body.question or "",
        answer=transcript,
        role=body.role or "",
        features_override=body.features or None,
    )
    label = str(result.get("label", "Average"))
    return {
        "label": label.lower(),
        "predicted_quality": label,
        "confidence": result.get("confidence"),
        "supporting_score": result.get("supporting_readiness_score"),
    }


# --------------------------------------------------------------------------- #
# 9.12 DASHBOARD CAREER SUMMARY — unlocked when score >= 90
# --------------------------------------------------------------------------- #
@app.post("/v1/dashboard/generate-summary", tags=["Dashboard"])
async def generate_dashboard_summary(body: DashboardSummaryRequest):
    data = body.dashboard or {}
    score = int(body.career_readiness_score or body.careerReadinessScore or data.get("careerReadinessScore") or data.get("career_readiness_score") or 0)
    if score < 90:
        raise HTTPException(status_code=403, detail="CAREER_SUMMARY_LOCKED")
    role = body.selected_role or body.selectedRole or data.get("selectedRole") or {}
    user = body.user or data.get("user") or {}
    summary = {
        "title": "Road2Work Career Readiness Summary",
        "user_name": user.get("name"),
        "selected_role": role.get("name") or role.get("roleName") or role.get("role_name"),
        "career_readiness_score": score,
        "readiness_status": data.get("readinessStatus") or data.get("readiness_status") or _status_id(score),
        "strengths": data.get("strengths") or [],
        "gaps": data.get("gaps") or [],
        "next_best_actions": data.get("nextBestActions") or data.get("next_best_actions") or [],
        "generated_at": _now_iso(),
    }
    return {
        "status": "success",
        "summary": summary,
        "download_ready": True,
        "message": "Career summary unlocked because Career Readiness Score >= 90.",
    }


# --------------------------------------------------------------------------- #
# ADMIN / DEV HELPERS
# --------------------------------------------------------------------------- #
@app.get("/v1/admin/ds-assets/status", tags=["Admin"])
async def ds_assets_status():
    return {"status": "success", "data": asset_status()}


@app.post("/v1/admin/reload-ds-assets", tags=["Admin"])
async def reload_ds_assets():
    status = reload_all_ds_assets()
    matrix_source = reload_role_matrix()
    taxonomy_source = reload_taxonomy()
    return {
        "status": "success",
        "message": "Data Science assets reloaded. Jika memakai multi-worker production, restart service tetap disarankan.",
        "data": {
            "asset_status": status,
            "matrix_source": matrix_source,
            "taxonomy_source": taxonomy_source,
            "role_tree_roles": get_target_roles(),
            "competency_map_roles": list(ds_get_competency_map().keys()),
            "question_seed_roles": list(ds_get_question_seed().keys()),
            "role_skill_matrix_roles": list(get_role_skill_matrix().keys()),
            "scoring_rubric_components": list(get_scoring_rubric().get("components", {}).keys()),
            "weakness_tags": list(get_weakness_taxonomy().keys()),
            "evidence_levels": get_evidence_ladder_mapping().get("levels", []),
            "need_clarification_rule": get_need_clarification_rule(),
        },
    }


@app.get("/v1/model/evaluation-report", tags=["Model"])
async def model_evaluation_report(dataset: str = "test", include_predictions: bool = False):
    try:
        key = dataset.lower()
        if key in {"split", "all"}:
            result = evaluate_split_datasets(include_predictions=include_predictions)
        elif key in {"manual", "realistic", "external"}:
            csv_path = manual_test_dataset_path()
            if not csv_path or not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="answer_quality_manual_test.csv belum tersedia.")
            result = evaluate_saved_model_detailed(csv_path=csv_path, include_predictions=include_predictions)
        elif key in {"test", "ds_test"}:
            csv_path = test_dataset_path()
            if not csv_path or not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="dataset_test.csv belum tersedia.")
            result = evaluate_saved_model_detailed(csv_path=csv_path, include_predictions=include_predictions)
        elif key in {"synthetic", "train", "ds"}:
            result = evaluate_saved_model_detailed(csv_path=None, include_predictions=include_predictions)
        else:
            raise HTTPException(status_code=400, detail="dataset harus test, split, manual, atau synthetic.")
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
