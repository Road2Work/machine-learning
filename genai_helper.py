"""
genai_helper.py — Generative AI Integration (Google Gemini)
Road2Work AI | CC26-PSU050
Author  : Muhammad Adil Imamul Haq Mubarak (AI Engineer – NLP Engine)
Role    : Interview question generation, narrative formalization, feedback scoring
"""

import time
import json
import re
from google import genai


# --------------------------------------------------------------------------- #
#  CONFIG                                                                       #
# --------------------------------------------------------------------------- #
GEMINI_API_KEY: str = "AIzaSyCSl79XibjRhGbQqTLLZI3zECcgPznnigE"   # Isi via environment variable di produksi
GEMINI_MODEL:   str = "gemini-2.5-flash"
MAX_RETRIES:    int = 3


# --------------------------------------------------------------------------- #
#  CLIENT                                                                       #
# --------------------------------------------------------------------------- #
client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------------------------------- #
#  INTERNAL: RETRY WRAPPER                                                      #
# --------------------------------------------------------------------------- #
def _call_gemini(prompt: str) -> str:
    """
    Wrapper retry-safe untuk semua panggilan Gemini.
    Menerapkan exponential backoff saat server sibuk (503 / UNAVAILABLE).

    Returns:
        str: Teks respons Gemini, atau pesan error yang dapat ditampilkan ke user.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text

        except Exception as e:
            err = str(e)
            if "503" in err or "UNAVAILABLE" in err:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt   # 1s, 2s, 4s
                    print(f"⚠️  Server Google sibuk. Retry dalam {wait}s "
                          f"(percobaan {attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(wait)
                else:
                    return "Maaf, AI sedang sangat sibuk. Silakan coba beberapa saat lagi."
            else:
                return f"Gagal menghubungi AI: {err}"

    return "Terjadi kesalahan tidak terduga."


# --------------------------------------------------------------------------- #
#  1. GENERATE INTERVIEW QUESTIONS                                              #
# --------------------------------------------------------------------------- #
def generate_interview_questions(
    skills: list[str],
    role: str = "posisi yang dipilih",
    num_questions: int = 3,
) -> str:
    """
    Membuat pertanyaan wawancara kontekstual berdasarkan skill dan role target.

    Args:
        skills       : Daftar canonical skill yang terdeteksi dari CV.
        role         : Nama role yang dilamar pengguna.
        num_questions: Jumlah pertanyaan yang dihasilkan.

    Returns:
        str: Daftar pertanyaan dalam format bullet points (Bahasa Indonesia).
    """
    if not skills:
        return "Skill tidak terdeteksi. Silakan lengkapi profil CV Anda."

    skills_text = ", ".join(skills)

    prompt = f"""
Kamu adalah seorang HR Senior dan Tech Recruiter berpengalaman di perusahaan teknologi top Indonesia.

Seorang kandidat melamar posisi **{role}** dengan keahlian utama: {skills_text}.

Tugasmu:
Buatkan tepat {num_questions} pertanyaan wawancara (kombinasi teknis dan studi kasus) 
yang spesifik menguji keahlian tersebut untuk posisi {role}. 
Gunakan Bahasa Indonesia yang profesional namun ramah.

Format: Langsung daftar pertanyaan bernomor (1. 2. 3.) tanpa salam pembuka atau penutup.
"""
    return _call_gemini(prompt)


# --------------------------------------------------------------------------- #
#  2. FORMALIZE NARRATIVE (Narasi Informal → Profil Profesional)               #
# --------------------------------------------------------------------------- #
def formalize_narrative(raw_narrative: str, role: str = "profesional") -> dict:
    """
    Mengubah narasi pengalaman informal menjadi profil profesional terstruktur.

    Args:
        raw_narrative: Teks pengalaman user (bebas/informal).
        role         : Role target yang menjadi konteks formalisasi.

    Returns:
        dict dengan kunci:
          - "professional_summary" (str)
          - "experience_bullets"   (list[str])
          - "highlighted_skills"   (list[str])
    """
    if not raw_narrative.strip():
        return {
            "professional_summary": "",
            "experience_bullets":   [],
            "highlighted_skills":   [],
        }

    prompt = f"""
Kamu adalah career coach profesional yang membantu kandidat memperkuat profil CV mereka.

Berikut narasi pengalaman mentah dari seorang kandidat yang menargetkan posisi **{role}**:
---
{raw_narrative}
---

Tugasmu:
1. Tulis "professional_summary": satu paragraf ringkasan profesional (maks 3 kalimat, formal, orang ketiga).
2. Tulis "experience_bullets": maksimal 4 poin pencapaian dalam format action-result (mulai kata kerja aktif).
3. Tulis "highlighted_skills": daftar 3-5 skill teknis maupun soft skill yang paling menonjol.

Balas HANYA dalam format JSON berikut (tanpa markdown, tanpa penjelasan tambahan):
{{
  "professional_summary": "...",
  "experience_bullets": ["...", "..."],
  "highlighted_skills": ["...", "..."]
}}
"""
    raw_response = _call_gemini(prompt)

    # Parse JSON; fallback ke teks mentah jika gagal
    try:
        cleaned = re.sub(r"```json|```", "", raw_response).strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {
            "professional_summary": raw_response,
            "experience_bullets":   [],
            "highlighted_skills":   [],
        }


# --------------------------------------------------------------------------- #
#  3. EVALUATE INTERVIEW ANSWER                                                 #
# --------------------------------------------------------------------------- #
def evaluate_interview_answer(
    question: str,
    answer: str,
    role: str = "posisi yang dipilih",
) -> dict:
    """
    Mengevaluasi jawaban wawancara pengguna secara kontekstual.

    Args:
        question: Pertanyaan interview yang diajukan.
        answer  : Jawaban pengguna (teks / hasil transkripsi).
        role    : Role target untuk konteks penilaian.

    Returns:
        dict dengan kunci:
          - "score"          (int, 0-100)
          - "feedback"       (str, evaluasi naratif)
          - "stronger_answer"(str, saran jawaban yang lebih baik)
    """
    prompt = f"""
Kamu adalah penilai wawancara berpengalaman untuk posisi **{role}**.

Pertanyaan: "{question}"
Jawaban kandidat: "{answer}"

Evaluasi jawaban tersebut dan balas HANYA dalam format JSON berikut (tanpa markdown):
{{
  "score": <angka 0–100>,
  "feedback": "<evaluasi singkat, maks 3 kalimat, Bahasa Indonesia>",
  "stronger_answer": "<contoh jawaban yang lebih baik, maks 4 kalimat>"
}}
"""
    raw_response = _call_gemini(prompt)

    try:
        cleaned = re.sub(r"```json|```", "", raw_response).strip()
        result  = json.loads(cleaned)
        # Pastikan score adalah int
        result["score"] = int(result.get("score", 0))
        return result
    except (json.JSONDecodeError, ValueError):
        return {
            "score":           0,
            "feedback":        raw_response,
            "stronger_answer": "",
        }


# --------------------------------------------------------------------------- #
#  SELF-TEST                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 55)
    print("  GENAI HELPER — SELF TEST")
    print("=" * 55)

    # -- Test 1: Interview Questions --
    print("\n[1] GENERATE INTERVIEW QUESTIONS")
    dummy_skills = ["Python", "TensorFlow", "NLP", "FastAPI"]
    print(f"  Skills : {dummy_skills}")
    print(f"  Role   : AI Engineer\n")
    questions = generate_interview_questions(dummy_skills, role="AI Engineer")
    print(questions)

    # -- Test 2: Narrative Formalization --
    print("\n[2] FORMALIZE NARRATIVE")
    raw = (
        "saya pernah bikin model machine learning buat deteksi spam email "
        "di project kuliah. pakai python dan scikit learn. hasilnya lumayan akurat sekitar 90%."
    )
    print(f"  Narasi mentah: {raw}\n")
    result = formalize_narrative(raw, role="Data Scientist")
    import json as _json
    print(_json.dumps(result, indent=2, ensure_ascii=False))

    # -- Test 3: Interview Evaluation --
    print("\n[3] EVALUATE INTERVIEW ANSWER")
    q = "Jelaskan bagaimana kamu menangani overfitting pada model deep learning?"
    a = "Saya biasanya pakai dropout dan early stopping."
    eval_result = evaluate_interview_answer(q, a, role="AI Engineer")
    print(_json.dumps(eval_result, indent=2, ensure_ascii=False))