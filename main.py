"""
main.py — FastAPI AI Service Road2Work.id

v1.2.0 — aligned with Fullstack API Contract (Backend ↔ FastAPI AI Service)

Contract endpoints called by Express Backend:
- POST /v1/context/extract-cv
- POST /v1/context/extract-profile
- POST /v1/interview/next-question
- POST /v1/stt/transcribe
- POST /v1/interview/evaluate-answer
- POST /v1/interview/clarifying-question
- POST /v1/model/predict-answer-quality
- POST /v1/interview/generate-result

Catatan:
- Response endpoint contract dibuat FLAT sesuai dokumen fullstack, bukan wrapper {data: ...}.
- Endpoint admin/dev tetap memakai wrapper sederhana agar tidak mengganggu contract.
- FastAPI tetap menyediakan fallback saat Gemini/model belum tersedia.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import io
import os
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
    get_role_skill_matrix,
    manual_test_dataset_path,
    predict_answer_quality,
    reload_role_matrix,
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
# CONSTANTS & OPTIONAL IN-MEMORY STORE
# --------------------------------------------------------------------------- #
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}
MAX_MAIN_QUESTIONS = int(os.getenv("MAX_MAIN_QUESTIONS", "5"))
STT_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("STT_LOW_CONFIDENCE_THRESHOLD", "0.60"))

# In production, Express + PostgreSQL menjadi source of truth.
# Store ini hanya fallback/dev helper jika AI service dipanggil mandiri.
contexts: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- #
# FALLBACK DS ASSETS
# --------------------------------------------------------------------------- #
_COMPETENCY_FALLBACK: dict[str, list[str]] = {
    "Data Analyst": ["role_relevance", "evidence_specificity", "technical_accuracy"],
    "Data Scientist": ["technical_accuracy", "evidence_specificity", "self_awareness"],
    "AI Engineer": ["technical_accuracy", "evidence_specificity", "role_relevance"],
    "ML Engineer": ["technical_accuracy", "star_structure", "evidence_specificity"],
    "Backend Developer": ["technical_accuracy", "role_relevance", "communication_clarity"],
    "default": ["role_relevance", "star_structure", "evidence_specificity"],
}

_QUESTION_SEED_FALLBACK: dict[str, list[str]] = {
    "Data Analyst": [
        "Ceritakan pengalaman membuat analisis/dashboard dari data mentah.",
        "Bagaimana kamu memastikan insight yang kamu berikan benar-benar berguna?",
    ],
    "AI Engineer": [
        "Ceritakan pengalaman membangun model atau fitur AI dari awal sampai bisa digunakan.",
        "Bagaimana kamu mengevaluasi kualitas output model atau pipeline AI?",
    ],
    "default": [
        "Ceritakan pengalaman paling relevan dengan role target.",
        "Jelaskan kontribusi pribadi dan hasil dari pengalaman tersebut.",
    ],
}


def _get_competency_map() -> dict[str, Any]:
    data = ds_get_competency_map()
    return data if isinstance(data, dict) and data else _COMPETENCY_FALLBACK


def _get_question_seed() -> dict[str, Any]:
    data = ds_get_question_seed()
    return data if isinstance(data, dict) and data else _QUESTION_SEED_FALLBACK


# --------------------------------------------------------------------------- #
# APP SETUP
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Road2Work FastAPI AI Service siap — API Contract v1.2.0")
    yield
    print("🛑 Road2Work FastAPI AI Service berhenti.")


app = FastAPI(
    title="Road2Work.id AI Service",
    description="AI service for context extraction, STT, adaptive question generation, evaluation, clarification, and model inference.",
    version="1.2.0",
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
# REQUEST SCHEMAS — aligned with fullstack contract
# --------------------------------------------------------------------------- #
class ExtractShortProfileRequest(BaseModel):
    # Contract fields
    target_role_id: str | None = None
    target_role_name: str | None = None
    most_relevant_experience: str | None = None
    skills_and_tools: str | None = None
    project_experience: str | None = None
    achievement_or_impact: str | None = None

    # Backward-compatible fields from previous AI implementation
    profile_text: str | None = None
    target_role: str | None = None
    domain: str | None = None
    role_family: str | None = None


class TargetRolePayload(BaseModel):
    id: str | None = None
    role_name: str | None = None
    role_family: str | None = None

    # Tolerate backend variations
    roleName: str | None = None
    roleFamily: str | None = None


class InterviewContextPayload(BaseModel):
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    experience_summary: str | list[str] | None = None
    evidence_items: list[str] = Field(default_factory=list)
    profile_summary: str | None = None


class SessionStatePayload(BaseModel):
    question_index: int = 1
    total_main_questions: int = MAX_MAIN_QUESTIONS
    asked_questions: list[str] = Field(default_factory=list)
    clarification_count: int = 0
    detected_weaknesses: list[str] = Field(default_factory=list)


class NextQuestionRequest(BaseModel):
    session_id: str | None = None
    target_role: TargetRolePayload | None = None
    interview_context: InterviewContextPayload | None = None
    session_state: SessionStatePayload = Field(default_factory=SessionStatePayload)
    question_seed: list[Any] | dict[str, Any] | None = None
    competency_map: list[Any] | dict[str, Any] | None = None

    # Backward-compatible legacy fields
    context_id: str | None = None
    max_main_questions: int | None = None


class QuestionPayload(BaseModel):
    id: str | None = None
    question_text: str | None = None
    question_type: str | None = None
    competency_target: str | None = None


class AnswerPayload(BaseModel):
    transcript_text: str = Field(..., min_length=1)
    stt_confidence: float | None = None


class EvaluateAnswerRequest(BaseModel):
    session_id: str | None = None
    question: QuestionPayload | None = None
    answer: AnswerPayload | None = None
    target_role: TargetRolePayload | None = None
    interview_context: InterviewContextPayload | None = None
    score_history: list[int] = Field(default_factory=list)
    clarification_count: int = 0

    # Backward-compatible legacy fields
    question_id: str | None = None
    transcript: str | None = None
    question_type: str | None = None
    stt_confidence: float | None = None


class ClarifyingQuestionRequest(BaseModel):
    target_role: str | TargetRolePayload
    question_text: str
    answer_text: str
    detected_weaknesses: list[str] = Field(default_factory=list)
    clarification_type: str | None = None
    clarification_goal: str | None = None


class ModelPredictRequest(BaseModel):
    # Contract fields
    answer_text: str | None = None
    features: dict[str, Any] | None = None

    # Backward-compatible fields
    question: str | None = ""
    answer: str | None = None
    role: str | None = ""


class ResultAnswerPayload(BaseModel):
    question_text: str | None = None
    answer_text: str | None = None
    score_breakdown: dict[str, int] | None = None
    evidence_level: int | None = None
    detected_weaknesses: list[str] = Field(default_factory=list)
    stronger_answer: str | None = None
    feedback: str | None = None


class GenerateResultRequest(BaseModel):
    session_id: str | None = None
    target_role: str | TargetRolePayload | None = None
    answers: list[ResultAnswerPayload] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Gagal membaca PDF: {exc}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Tidak ada teks yang bisa diekstrak. Pastikan CV bukan scan gambar.")
    return text.strip()


def _validate_audio_file(audio: UploadFile) -> str:
    suffix = os.path.splitext(audio.filename or "")[-1].lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Format audio '{suffix}' tidak didukung. Gunakan: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}")
    return suffix


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def _role_name(payload: str | TargetRolePayload | None, fallback: str = "posisi yang dipilih") -> str:
    if isinstance(payload, str):
        return payload.strip() or fallback
    if isinstance(payload, TargetRolePayload):
        return (payload.role_name or payload.roleName or payload.id or fallback).strip()
    return fallback


def _role_family(payload: TargetRolePayload | None, role_name: str) -> str | None:
    if payload:
        family = payload.role_family or payload.roleFamily
        if family:
            return family
    meta = get_role_family_for_role(role_name) or {}
    return meta.get("role_family")


def _experience_to_string(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return _safe_text(value)


def _evidence_items_from_context(context: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("evidence_items", "experience_summary"):
        value = context.get(key)
        if isinstance(value, list):
            items.extend(str(v).strip() for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            items.append(value.strip())
    if not items:
        signals = context.get("evidence_signals", {}) or {}
        if signals.get("has_metric"):
            items.append("Answer contains measurable result or numeric signal")
        if signals.get("impact_keywords"):
            items.append("Answer contains impact signal")
    # unique
    seen = set()
    unique = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:6]


def _create_context(raw_text: str, target_role: str, source: str, filename: str | None = None) -> dict[str, Any]:
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Konteks user wajib ada. Upload CV atau isi profil singkat.")
    if not target_role.strip():
        raise HTTPException(status_code=400, detail="target_role_name wajib dikirim.")

    meta = get_role_family_for_role(target_role.strip()) or {}
    skills = normalize_skills(extract_skills(raw_text))
    role_recs = compute_role_fit_scores(skills, top_n=3)
    professional_profile = formalize_narrative(raw_text, role=target_role)
    experience_summary = extract_experience_summary(raw_text)
    evidence_signals = extract_evidence_signals(raw_text)
    initial_evidence_score = calculate_initial_evidence_score(raw_text, skills)

    context_id = str(uuid4())
    context = {
        "context_id": context_id,
        "source": source,
        "filename": filename,
        "target_role": target_role.strip(),
        "target_role_id": None,
        "domain": meta.get("domain"),
        "role_family": meta.get("role_family"),
        "raw_text": raw_text.strip(),
        "cleaned_text": clean_text(raw_text),
        "profile_summary": professional_profile.get("professional_summary") or raw_text.strip()[:500],
        "skills": skills,
        "tools": skills,
        "experience_summary": experience_summary,
        "evidence_signals": evidence_signals,
        "evidence_items": experience_summary,
        "initial_evidence_score": initial_evidence_score,
        "professional_profile": professional_profile,
        "role_recommendations": role_recs,
        "created_at": _now_iso(),
    }
    contexts[context_id] = context
    return context


def _contract_context_response(context: dict[str, Any], source: str, include_raw_preview: bool = False) -> dict[str, Any]:
    response = {
        "status": "success",
        "source": source,
        "profile_summary": context.get("profile_summary", ""),
        "skills": context.get("skills", []),
        "tools": context.get("tools", []),
        "experience_summary": _experience_to_string(context.get("experience_summary")),
        "evidence_items": _evidence_items_from_context(context),
        "initial_evidence_score": int(context.get("initial_evidence_score", 0)),
    }
    # Extra debug field, useful for backend if needed; harmless for non-strict consumers.
    response["context_id"] = context.get("context_id")
    if include_raw_preview:
        response["raw_text_preview"] = _safe_text(context.get("raw_text"))[:700]
    return response


_SCORE_KEY_ALIASES = {
    "role_relevance": "role_relevance",
    "roleRelevance": "role_relevance",
    "star_structure": "star_structure",
    "starStructure": "star_structure",
    "evidence_specificity": "evidence_specificity",
    "evidenceSpecificity": "evidence_specificity",
    "technical_accuracy": "technical_accuracy",
    "technicalAccuracy": "technical_accuracy",
    "communication_clarity": "communication_clarity",
    "communicationClarity": "communication_clarity",
    "self_awareness": "self_awareness",
    "selfAwareness": "self_awareness",
}


def _normalize_score_breakdown_contract(score_breakdown: dict[str, Any] | None) -> dict[str, int]:
    score_breakdown = score_breakdown or {}
    normalized = {key: 0 for key in EVALUATION_WEIGHTS}
    for raw_key, value in score_breakdown.items():
        key = _SCORE_KEY_ALIASES.get(raw_key, raw_key)
        if key in normalized:
            try:
                normalized[key] = max(0, min(100, int(round(float(value)))))
            except (TypeError, ValueError):
                normalized[key] = 0
    return normalized


def _contract_weakness_tag(tag: str) -> str:
    tag = str(tag).strip().lower()
    mapping = {
        "tools": "missing_tools",
        "tools_missing": "missing_tools",
        "missing_tools": "missing_tools",
        "impact": "missing_impact",
        "impact_missing": "missing_impact",
        "missing_impact": "missing_impact",
        "contribution": "missing_personal_contribution",
        "contribution_unclear": "missing_personal_contribution",
        "missing_personal_contribution": "missing_personal_contribution",
        "specificity": "weak_evidence",
        "evidence_specificity": "weak_evidence",
        "weak_evidence": "weak_evidence",
        "measurable_result_missing": "weak_evidence",
        "metric": "weak_evidence",
        "star_structure": "weak_star_structure",
        "star_structure_missing": "weak_star_structure",
        "weak_star_structure": "weak_star_structure",
        "role_relevance": "low_role_relevance",
        "role_relevance_low": "low_role_relevance",
        "low_role_relevance": "low_role_relevance",
        "unclear_audio": "unclear_audio",
    }
    return mapping.get(tag, tag)


def _contract_weaknesses(tags: list[Any]) -> list[str]:
    result: list[str] = []
    for tag in tags or []:
        mapped = _contract_weakness_tag(str(tag))
        if mapped and mapped not in result:
            result.append(mapped)
    return result


def _contract_clarification_type(value: str | None, weaknesses: list[str] | None = None) -> str | None:
    if not value or value == "null":
        value = None
    if value:
        mapped = _contract_weakness_tag(value)
        if mapped in {
            "unclear_audio",
            "weak_evidence",
            "missing_tools",
            "missing_impact",
            "missing_personal_contribution",
            "weak_star_structure",
            "low_role_relevance",
        }:
            return mapped
    for weakness in weaknesses or []:
        mapped = _contract_weakness_tag(weakness)
        if mapped in {
            "weak_evidence",
            "missing_tools",
            "missing_impact",
            "missing_personal_contribution",
            "weak_star_structure",
            "low_role_relevance",
        }:
            return mapped
    return None


def _contract_evaluation_response(evaluation: dict[str, Any], model_support: dict[str, Any]) -> dict[str, Any]:
    score_breakdown = _normalize_score_breakdown_contract(evaluation.get("score_breakdown"))
    weaknesses = _contract_weaknesses(evaluation.get("weakness", []))
    clarification_type = _contract_clarification_type(evaluation.get("clarification_type"), weaknesses)
    needs_clarification = bool(evaluation.get("need_clarification", False))
    if needs_clarification and not clarification_type:
        clarification_type = _contract_clarification_type(None, weaknesses) or "weak_evidence"

    return {
        "score_breakdown": score_breakdown,
        "answer_score": int(evaluation.get("final_score", 0)),
        "detected_weaknesses": weaknesses,
        "evidence_level": int(evaluation.get("evidence_level", 1)),
        "needs_clarification": needs_clarification,
        "clarification_type": clarification_type if needs_clarification else None,
        "feedback": str(evaluation.get("feedback", "")),
        "stronger_answer": str(evaluation.get("stronger_answer", "")),
        "model_support": {
            "predicted_quality": model_support.get("label") or model_support.get("predicted_quality"),
            "confidence": model_support.get("confidence"),
        },
    }


def _interview_context_to_dict(payload: InterviewContextPayload | None) -> dict[str, Any]:
    if payload is None:
        return {}
    data = payload.model_dump()
    data["experience_summary"] = _experience_to_string(data.get("experience_summary"))
    return data


def _score_breakdown_average(answers: list[dict[str, Any]]) -> dict[str, int]:
    if not answers:
        return {key: 0 for key in EVALUATION_WEIGHTS}
    totals = {key: 0.0 for key in EVALUATION_WEIGHTS}
    counts = {key: 0 for key in EVALUATION_WEIGHTS}
    for ans in answers:
        breakdown = _normalize_score_breakdown_contract(ans.get("score_breakdown"))
        for key, value in breakdown.items():
            totals[key] += value
            counts[key] += 1
    return {key: int(round(totals[key] / counts[key])) if counts[key] else 0 for key in EVALUATION_WEIGHTS}


def _weighted_score_from_breakdown(breakdown: dict[str, int]) -> int:
    total = 0.0
    for key, weight in EVALUATION_WEIGHTS.items():
        total += int(breakdown.get(key, 0)) * weight
    return max(0, min(100, int(round(total))))


def _readiness_status(final_score: int) -> str:
    if final_score >= 80:
        return "Ready"
    if final_score >= 60:
        return "Almost Ready"
    return "Needs Practice"


def _normalize_result_dashboard(dashboard: dict[str, Any], answers: list[dict[str, Any]], final_score: int) -> dict[str, Any]:
    strengths_raw = dashboard.get("strengths") or []
    improvement_raw = dashboard.get("improvement_areas") or []

    strengths = []
    for item in strengths_raw[:3] if isinstance(strengths_raw, list) else []:
        if isinstance(item, dict):
            strengths.append({
                "title": str(item.get("title") or "Strength"),
                "description": str(item.get("description") or item.get("reason") or "Jawaban memiliki bagian yang sudah cukup kuat."),
                "evidence": item.get("evidence"),
            })

    improvement_areas = []
    for item in improvement_raw[:3] if isinstance(improvement_raw, list) else []:
        if isinstance(item, dict):
            improvement_areas.append({
                "title": str(item.get("title") or "Improvement Area"),
                "description": str(item.get("description") or item.get("cause") or item.get("suggestion") or "Perlu diperkuat dengan detail yang lebih spesifik."),
                "evidence": item.get("evidence"),
            })

    if not strengths:
        strengths = [{"title": "Role relevance", "description": "Jawaban sudah menunjukkan keterkaitan dengan target role.", "evidence": None}]
    if not improvement_areas:
        improvement_areas = [{"title": "Evidence specificity", "description": "Tambahkan detail tools, kontribusi pribadi, dan hasil yang lebih terukur.", "evidence": None}]

    before_after_raw = dashboard.get("before_after_answer_improvement") or dashboard.get("before_after_improvement")
    before_after_list: list[dict[str, Any]] = []
    if isinstance(before_after_raw, list):
        for item in before_after_raw:
            if isinstance(item, dict):
                before_after_list.append({
                    "question_text": str(item.get("question_text") or item.get("questionText") or ""),
                    "before_answer": str(item.get("before_answer") or item.get("beforeAnswer") or item.get("before") or ""),
                    "after_answer": str(item.get("after_answer") or item.get("afterAnswer") or item.get("after") or ""),
                    "improvement_notes": item.get("improvement_notes") or item.get("improvementNotes") or [str(item.get("why_better") or item.get("problem") or "Jawaban lebih terstruktur.")],
                })
    elif isinstance(before_after_raw, dict):
        first = answers[0] if answers else {}
        before_after_list.append({
            "question_text": str(first.get("question_text") or before_after_raw.get("question_text") or ""),
            "before_answer": str(before_after_raw.get("before") or first.get("answer_text") or ""),
            "after_answer": str(before_after_raw.get("after") or first.get("stronger_answer") or ""),
            "improvement_notes": [str(before_after_raw.get("problem") or "Masalah utama diperbaiki."), str(before_after_raw.get("why_better") or "Jawaban menjadi lebih jelas dan berbasis evidence.")],
        })

    next_raw = dashboard.get("next_practice_recommendation") or {}
    practice_type = str(next_raw.get("practice_type") or next_raw.get("practiceType") or "Evidence Booster Practice")
    valid_practices = {
        "Behavioral STAR Practice",
        "Evidence Booster Practice",
        "Technical Interview Practice",
        "Answer Clarity Practice",
        "Role Understanding Practice",
        "Reflection Practice",
    }
    if practice_type not in valid_practices:
        practice_type = "Evidence Booster Practice"

    focus = next_raw.get("focus_areas") or next_raw.get("focusAreas") or next_raw.get("focus") or ["Tambahkan tools yang digunakan", "Jelaskan kontribusi pribadi", "Sebutkan hasil atau impact"]
    if not isinstance(focus, list):
        focus = [str(focus)]

    return {
        "strengths": strengths,
        "improvement_areas": improvement_areas,
        "before_after_improvement": before_after_list,
        "next_practice_recommendation": {
            "practice_type": practice_type,
            "reason": str(next_raw.get("reason") or "Jawaban perlu diperkuat dengan evidence yang lebih spesifik."),
            "focus_areas": [str(x) for x in focus],
        },
    }


# --------------------------------------------------------------------------- #
# GENERAL / DEV ENDPOINTS
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["General"])
async def health_check():
    ds_status = asset_status()
    existing_assets = {key: value["exists"] for key, value in ds_status.get("assets", {}).items()}
    return {
        "status": "ok",
        "service": "Road2Work.id AI Service",
        "version": app.version,
        "stt": get_model_info(),
        "ds_resources_dir": ds_status.get("resources_dir"),
        "ds_assets_loaded": existing_assets,
    }


@app.get("/v1/roles", tags=["General"])
async def list_roles_v1():
    return {"status": "success", "data": {"roles": get_target_roles() or ROLE_LABELS, "role_tree": get_role_tree()}}


@app.get("/v1/roles/tree", tags=["General"])
async def role_tree_dropdown():
    return {"status": "success", "data": get_role_tree()}


# --------------------------------------------------------------------------- #
# 6.1 EXTRACT CV CONTEXT — Contract: cvFile, targetRoleId, targetRoleName
# --------------------------------------------------------------------------- #
@app.post("/v1/context/extract-cv", tags=["Context"])
async def extract_cv_context(
    cvFile: UploadFile | None = File(None),
    targetRoleId: str | None = Form(None),
    targetRoleName: str | None = Form(None),
    # Backward-compatible aliases
    file: UploadFile | None = File(None),
    target_role: str | None = Form(None),
    target_role_id: str | None = Form(None),
    target_role_name: str | None = Form(None),
):
    upload = cvFile or file
    role_name = targetRoleName or target_role_name or target_role
    role_id = targetRoleId or target_role_id

    if upload is None:
        raise HTTPException(status_code=400, detail="cvFile wajib dikirim.")
    if not role_name:
        raise HTTPException(status_code=400, detail="targetRoleName wajib dikirim.")
    if not upload.filename or not upload.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="CV_INVALID_FORMAT: hanya file PDF yang diterima.")

    contents = await upload.read()
    raw_text = _extract_text_from_pdf(contents)
    context = _create_context(raw_text=raw_text, target_role=role_name, source="cv", filename=upload.filename)
    context["target_role_id"] = role_id
    return _contract_context_response(context, source="cv", include_raw_preview=True)


# --------------------------------------------------------------------------- #
# 6.2 EXTRACT SHORT PROFILE CONTEXT
# --------------------------------------------------------------------------- #
@app.post("/v1/context/extract-profile", tags=["Context"])
async def extract_short_profile_context(body: ExtractShortProfileRequest):
    role_name = body.target_role_name or body.target_role
    if not role_name:
        raise HTTPException(status_code=400, detail="target_role_name wajib dikirim.")

    if body.profile_text:
        raw_text = body.profile_text
    else:
        parts = [
            body.most_relevant_experience,
            body.skills_and_tools,
            body.project_experience,
            body.achievement_or_impact,
        ]
        raw_text = "\n".join(str(part).strip() for part in parts if part and str(part).strip())

    try:
        extracted = extract_from_short_profile(raw_text, target_role=role_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    context = _create_context(raw_text=extracted["raw_text"], target_role=role_name, source="short_profile")
    context["target_role_id"] = body.target_role_id
    context.update({
        "skills": extracted["skills"],
        "tools": extracted["tools"],
        "experience_summary": extracted["experience_summary"],
        "evidence_signals": extracted["evidence_signals"],
        "evidence_items": extracted["experience_summary"],
        "initial_evidence_score": extracted["initial_evidence_score"],
        "profile_summary": extracted.get("profile_summary") or context.get("profile_summary"),
    })
    return _contract_context_response(context, source="short_profile", include_raw_preview=False)


# --------------------------------------------------------------------------- #
# 6.3 GENERATE NEXT QUESTION — stateless contract
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/next-question", tags=["Interview Engine"])
async def next_question(body: NextQuestionRequest):
    # If legacy context_id is used, hydrate context from in-memory store; otherwise use contract payload.
    if body.context_id:
        context = contexts.get(body.context_id)
        if not context:
            raise HTTPException(status_code=404, detail=f"context_id '{body.context_id}' tidak ditemukan.")
        role = context.get("target_role", "posisi yang dipilih")
        role_payload = TargetRolePayload(id=None, role_name=role, role_family=context.get("role_family"))
        interview_context = {
            "skills": context.get("skills", []),
            "tools": context.get("tools", []),
            "experience_summary": _experience_to_string(context.get("experience_summary")),
            "evidence_items": _evidence_items_from_context(context),
            "profile_summary": context.get("profile_summary"),
        }
    else:
        role_payload = body.target_role or TargetRolePayload(role_name="posisi yang dipilih")
        role = _role_name(role_payload)
        interview_context = _interview_context_to_dict(body.interview_context)

    session_state = body.session_state.model_dump() if body.session_state else {}
    session_state["main_question_index"] = max(0, int(session_state.get("question_index", 1)) - 1)
    session_state["asked_questions"] = session_state.get("asked_questions", [])
    session_state["weakness_history"] = session_state.get("detected_weaknesses", [])

    competency_map = body.competency_map if body.competency_map else _get_competency_map()
    question_seed = body.question_seed if body.question_seed else _get_question_seed()

    generated = generate_natural_question(
        role=role,
        interview_context=interview_context,
        interview_state=session_state,
        role_skill_matrix=get_role_skill_matrix(),
        competency_map=competency_map if isinstance(competency_map, dict) else {role: {"competencies": competency_map}},
        question_seed=question_seed if isinstance(question_seed, dict) else {role: question_seed},
    )

    return {
        "question_text": generated.get("question", f"Ceritakan pengalaman paling relevan dengan role {role}."),
        "question_type": "main",
        "parent_question_id": None,
        "competency_target": generated.get("competency_target") or "role_relevance_and_experience",
        "clarification_type": None,
        "hrd_state": "asking",
    }


# --------------------------------------------------------------------------- #
# 6.4 TRANSCRIBE AUDIO — audioFile, language
# --------------------------------------------------------------------------- #
@app.post("/v1/stt/transcribe", tags=["Speech-to-Text"])
async def transcribe_audio(
    audioFile: UploadFile | None = File(None),
    language: str | None = Form(None),
    # Backward-compatible alias
    audio: UploadFile | None = File(None),
):
    upload = audioFile or audio
    if upload is None:
        raise HTTPException(status_code=400, detail="audioFile wajib dikirim.")

    suffix = _validate_audio_file(upload)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        stt_result = proses_audio_ke_teks(tmp_path)
        if stt_result.get("status") == "error":
            raise HTTPException(status_code=422, detail=stt_result.get("pesan"))

        transcript = stt_result.get("data_transkrip") or ""
        # faster-whisper result in stt_utils currently does not expose confidence.
        confidence = stt_result.get("confidence")
        if confidence is None:
            confidence = 0.90 if len(transcript.split()) >= 4 else 0.50
        confidence = round(float(confidence), 2)
        needs_clarification = confidence < STT_LOW_CONFIDENCE_THRESHOLD or len(transcript.split()) < 3

        return {
            "status": "success",
            "transcript_text": transcript,
            "stt_confidence": confidence,
            "duration_seconds": stt_result.get("durasi_audio_detik"),
            "needs_clarification": needs_clarification,
            "clarification_type": "unclear_audio" if needs_clarification else None,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# --------------------------------------------------------------------------- #
# 6.5 EVALUATE ANSWER — nested contract body
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/evaluate-answer", tags=["Interview Engine"])
async def evaluate_answer(body: EvaluateAnswerRequest):
    question_text = ""
    question_type = "main"
    competency_target = None

    if body.question:
        question_text = body.question.question_text or ""
        question_type = body.question.question_type or "main"
        competency_target = body.question.competency_target
    elif isinstance(body.question, str):
        question_text = body.question

    transcript = ""
    stt_confidence = body.stt_confidence
    if body.answer:
        transcript = body.answer.transcript_text
        stt_confidence = body.answer.stt_confidence
    elif body.transcript:
        transcript = body.transcript
        question_type = body.question_type or question_type

    role = _role_name(body.target_role)
    if not question_text:
        raise HTTPException(status_code=400, detail="question.question_text wajib dikirim.")
    if not transcript:
        raise HTTPException(status_code=400, detail="answer.transcript_text wajib dikirim.")

    interview_context = _interview_context_to_dict(body.interview_context)
    raw_eval = evaluate_interview_answer(
        question=question_text,
        answer=transcript,
        role=role,
        interview_context=interview_context,
        competency_target=competency_target,
    )
    evaluation = normalize_evaluation_schema(raw_eval, original_answer=transcript)
    model_support = predict_answer_quality(question=question_text, answer=transcript, role=role)
    contract_eval = _contract_evaluation_response(evaluation, model_support)

    # STT low confidence should force unclear-audio clarification.
    if stt_confidence is not None and stt_confidence < STT_LOW_CONFIDENCE_THRESHOLD:
        contract_eval["needs_clarification"] = True
        contract_eval["clarification_type"] = "unclear_audio"
        if "unclear_audio" not in contract_eval["detected_weaknesses"]:
            contract_eval["detected_weaknesses"].insert(0, "unclear_audio")

    return contract_eval


# --------------------------------------------------------------------------- #
# 6.6 GENERATE CLARIFYING QUESTION
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/clarifying-question", tags=["Interview Engine"])
async def clarifying_question(body: ClarifyingQuestionRequest):
    role = _role_name(body.target_role if isinstance(body.target_role, TargetRolePayload) else str(body.target_role))
    clarification_type = _contract_clarification_type(body.clarification_type, body.detected_weaknesses) or "weak_evidence"

    # genai_helper expects internal-ish types; contract type still works via fallback mapping.
    question_text = generate_clarification_question(
        original_question=body.question_text,
        transcript=body.answer_text,
        weakness_tags=body.detected_weaknesses,
        role=role,
        clarification_type=clarification_type,
    )
    return {
        "question_text": question_text,
        "question_type": "clarification",
        "clarification_type": clarification_type,
        "hrd_state": "clarifying",
    }


# --------------------------------------------------------------------------- #
# 6.7 PREDICT ANSWER QUALITY WITH TENSORFLOW MODEL
# --------------------------------------------------------------------------- #
@app.post("/v1/model/predict-answer-quality", tags=["Model"])
async def predict_answer_quality_endpoint(body: ModelPredictRequest):
    answer_text = body.answer_text or body.answer
    if not answer_text:
        raise HTTPException(status_code=400, detail="answer_text wajib dikirim.")

    # Contract does not send question/role. Keep them optional for better model context when available.
    result = predict_answer_quality(question=body.question or "", answer=answer_text, role=body.role or "")
    return {
        "predicted_quality": result.get("label"),
        "confidence": result.get("confidence"),
        "supporting_score": result.get("supporting_readiness_score"),
    }


# --------------------------------------------------------------------------- #
# 6.8 GENERATE FINAL RESULT
# --------------------------------------------------------------------------- #
@app.post("/v1/interview/generate-result", tags=["Interview Engine"])
async def generate_final_result(body: GenerateResultRequest):
    role = _role_name(body.target_role)
    answers_raw: list[dict[str, Any]] = []
    for item in body.answers:
        data = item.model_dump()
        score_breakdown = _normalize_score_breakdown_contract(data.get("score_breakdown"))
        answer_score = _weighted_score_from_breakdown(score_breakdown)
        answers_raw.append({
            "question_type": "main",
            "question_text": data.get("question_text"),
            "answer_text": data.get("answer_text"),
            "transcript": data.get("answer_text"),
            "stronger_answer": data.get("stronger_answer"),
            "feedback": data.get("feedback"),
            "evaluation": {
                "score_breakdown": score_breakdown,
                "final_score": answer_score,
                "evidence_level": data.get("evidence_level") or 1,
                "weakness": data.get("detected_weaknesses") or [],
                "stronger_answer": data.get("stronger_answer") or "",
                "feedback": data.get("feedback") or "",
            },
            "score_breakdown": score_breakdown,
            "evidence_level": data.get("evidence_level") or 1,
            "detected_weaknesses": data.get("detected_weaknesses") or [],
        })

    avg_breakdown = _score_breakdown_average([a for a in answers_raw])
    if answers_raw:
        final_score = int(round(sum(_weighted_score_from_breakdown(a["score_breakdown"]) for a in answers_raw) / len(answers_raw)))
        evidence_level = int(round(sum(int(a.get("evidence_level") or 1) for a in answers_raw) / len(answers_raw)))
    else:
        final_score = 0
        evidence_level = 1

    dashboard_raw = generate_result_dashboard(role=role, interview_context={}, answers=answers_raw, final_score=final_score)
    dashboard = _normalize_result_dashboard(dashboard_raw, answers_raw, final_score)

    return {
        "final_score": final_score,
        "readiness_status": _readiness_status(final_score),
        "evidence_level": max(1, min(5, evidence_level)),
        "score_breakdown": avg_breakdown,
        **dashboard,
    }


# --------------------------------------------------------------------------- #
# OPTIONAL DEV / ADMIN HELPERS
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
            "competency_map_roles": list(_get_competency_map().keys()),
            "question_seed_roles": list(_get_question_seed().keys()),
            "role_skill_matrix_roles": list(get_role_skill_matrix().keys()),
            "scoring_rubric_components": list(get_scoring_rubric().get("components", {}).keys()),
            "weakness_tags": list(get_weakness_taxonomy().keys()),
            "evidence_levels": get_evidence_ladder_mapping().get("levels", []),
            "need_clarification_rule": get_need_clarification_rule(),
        },
    }


@app.get("/v1/model/evaluation-report", tags=["Model"])
async def model_evaluation_report(dataset: str = "manual", include_predictions: bool = False):
    try:
        if dataset.lower() in {"manual", "realistic", "external"}:
            csv_path = manual_test_dataset_path()
            if not csv_path or not os.path.exists(csv_path):
                raise HTTPException(status_code=404, detail="answer_quality_manual_test.csv belum tersedia.")
        elif dataset.lower() in {"synthetic", "train", "ds"}:
            csv_path = None
        else:
            raise HTTPException(status_code=400, detail="dataset harus manual atau synthetic.")
        result = evaluate_saved_model_detailed(csv_path=csv_path, include_predictions=include_predictions)
        return {"status": "success", "data": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
