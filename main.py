"""
main.py — FastAPI Service: NLP Engine & Role Matching
Road2Work AI | CC26-PSU050
Author  : Muhammad Adil Imamul Haq Mubarak (AI Engineer – NLP Engine)
Role    : REST API endpoint untuk ekstraksi profil, role matching, interview prep
"""

import io
import json
from contextlib import asynccontextmanager

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from genai_helper import (
    evaluate_interview_answer,
    formalize_narrative,
    generate_interview_questions,
)
from model_builder import ROLE_LABELS, compute_role_fit_scores
from nlp_utils import clean_text, extract_skills, normalize_skills


# --------------------------------------------------------------------------- #
#  APP SETUP                                                                    #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown handler — bisa ditambah load model di sini."""
    print("🚀 Road2Work AI Service siap!")
    yield
    print("🛑 Service dihentikan.")


app = FastAPI(
    title="Road2Work AI — NLP Engine",
    description=(
        "API untuk ekstraksi profil CV, role matching, formalisasi narasi, "
        "dan persiapan wawancara berbasis AI."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — izinkan frontend Next.js (dev & prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://road2work.id"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
#  REQUEST / RESPONSE SCHEMAS                                                   #
# --------------------------------------------------------------------------- #
class NarrativeRequest(BaseModel):
    narrative: str
    role: str = "profesional"


class InterviewEvalRequest(BaseModel):
    question: str
    answer: str
    role: str = "posisi yang dipilih"


class SkillsRequest(BaseModel):
    skills: list[str]
    role: str = "posisi yang dipilih"
    num_questions: int = 3


# --------------------------------------------------------------------------- #
#  HELPERS                                                                      #
# --------------------------------------------------------------------------- #
def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Ekstrak teks dari bytes PDF menggunakan pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Gagal membaca PDF: {str(e)}")

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="Tidak ada teks yang dapat diekstrak dari PDF. "
                   "Pastikan file bukan scan gambar.",
        )
    return text


# --------------------------------------------------------------------------- #
#  ENDPOINTS                                                                    #
# --------------------------------------------------------------------------- #

# ── Health check ────────────────────────────────────────────────────────────
@app.get("/health", tags=["General"])
async def health_check():
    """Cek apakah service berjalan normal."""
    return {"status": "ok", "service": "Road2Work AI", "version": "0.2.0"}


# ── 1. Extract Profile dari CV PDF ──────────────────────────────────────────
@app.post("/extract-profile", tags=["NLP Engine"])
async def extract_profile(file: UploadFile = File(...)):
    """
    Endpoint utama: menerima CV PDF, lalu:
      1. Ekstrak teks dari PDF
      2. Bersihkan teks (NLP preprocessing)
      3. Ekstrak & normalisasi skill
      4. Hitung role-fit score (rule-based)
      5. Formalisasi narasi via Gemini
      6. Generate pertanyaan interview via Gemini

    Returns:
        JSON dengan profil lengkap, rekomendasi role, dan pertanyaan interview.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Hanya file PDF yang diterima.")

    contents = await file.read()
    raw_text = _extract_text_from_pdf(contents)

    # --- Pipeline NLP ---
    cleaned_text = clean_text(raw_text)
    skills       = extract_skills(raw_text)          # Gunakan raw agar multi-word terbaca
    skills       = normalize_skills(skills)

    # --- Role Matching (rule-based, akan diganti model setelah data Addya siap) ---
    role_recommendations = compute_role_fit_scores(skills, top_n=3)
    top_role = role_recommendations[0]["role"] if role_recommendations else "profesional"

    # --- GenAI: Formalisasi Narasi ---
    formalized = formalize_narrative(raw_text, role=top_role)

    # --- GenAI: Pertanyaan Interview ---
    interview_questions = generate_interview_questions(skills, role=top_role)

    return {
        "status":   "success",
        "filename": file.filename,
        "data": {
            "extracted_skills":       skills,
            "role_recommendations":   role_recommendations,
            "professional_profile":   formalized,
            "ai_interview_questions": interview_questions,
        },
    }


# ── 2. Formalize Narrative ───────────────────────────────────────────────────
@app.post("/formalize-narrative", tags=["NLP Engine"])
async def formalize_narrative_endpoint(body: NarrativeRequest):
    """
    Memformalkan narasi pengalaman user (teks bebas) menjadi profil profesional.
    Cocok untuk input narasi / pencapaian nonformal (bukan PDF).
    """
    result = formalize_narrative(body.narrative, role=body.role)
    return {"status": "success", "data": result}


# ── 3. Role Fit dari Daftar Skill ───────────────────────────────────────────
@app.post("/role-fit", tags=["Role Matching"])
async def role_fit(body: SkillsRequest):
    """
    Menghitung role-fit score dari daftar skill yang diberikan langsung.
    Berguna untuk input manual / dari fitur tambah skill di frontend.
    """
    normalized_skills = normalize_skills(body.skills)
    recommendations   = compute_role_fit_scores(normalized_skills, top_n=3)
    return {
        "status": "success",
        "data": {
            "input_skills":         normalized_skills,
            "role_recommendations": recommendations,
        },
    }


# ── 4. Generate Interview Questions ─────────────────────────────────────────
@app.post("/interview/questions", tags=["Interview Readiness"])
async def get_interview_questions(body: SkillsRequest):
    """
    Menghasilkan pertanyaan wawancara berbasis skill dan role target.
    """
    questions = generate_interview_questions(
        skills=body.skills,
        role=body.role,
        num_questions=body.num_questions,
    )
    return {"status": "success", "data": {"questions": questions}}


# ── 5. Evaluate Interview Answer ────────────────────────────────────────────
@app.post("/interview/evaluate", tags=["Interview Readiness"])
async def evaluate_answer(body: InterviewEvalRequest):
    """
    Mengevaluasi jawaban interview user dan memberikan skor + feedback AI.
    Jawaban bisa berupa teks langsung atau hasil transkripsi STT dari Diva.
    """
    result = evaluate_interview_answer(
        question=body.question,
        answer=body.answer,
        role=body.role,
    )
    return {"status": "success", "data": result}


# ── 6. List Available Roles ──────────────────────────────────────────────────
@app.get("/roles", tags=["General"])
async def list_roles():
    """Mengembalikan daftar role yang didukung sistem saat ini."""
    return {"status": "success", "data": {"roles": ROLE_LABELS}}