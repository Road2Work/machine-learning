"""
nlp_utils.py — NLP Preprocessing & Skill Extraction Pipeline
Road2Work AI | CC26-PSU050
Author  : Muhammad Adil Imamul Haq Mubarak (AI Engineer – NLP Engine)
Role    : Pre-processing teks CV, ekstraksi skill, normalisasi ke taxonomy
"""

import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# --------------------------------------------------------------------------- #
#  SETUP                                                                        #
# --------------------------------------------------------------------------- #
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --------------------------------------------------------------------------- #
#  SKILL TAXONOMY (role-skill matrix — akan diperluas oleh Addya / Data Sci)  #
#  Format: { canonical_skill: [alias / variasi penulisan] }                   #
# --------------------------------------------------------------------------- #
SKILL_TAXONOMY: dict[str, list[str]] = {
    # --- Programming Languages ---
    "python":       ["python3", "py"],
    "javascript":   ["js", "javascript"],
    "typescript":   ["ts", "typescript"],
    "sql":          ["mysql", "postgresql", "sqlite", "postgres", "sql server"],
    "r":            ["rstudio", "r language"],
    "java":         ["java se", "java ee"],
    "c++":          ["cpp", "c plus plus"],
    # --- ML / AI ---
    "tensorflow":   ["tf", "keras", "tensorflow2"],
    "pytorch":      ["torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "pandas":       ["pd", "pandas"],
    "numpy":        ["np", "numpy"],
    "nlp":          ["natural language processing", "text mining", "text classification"],
    "deep learning":["dl", "deep learning", "neural network", "ann", "cnn", "rnn", "lstm"],
    "machine learning": ["ml", "machine learning", "supervised learning", "unsupervised learning"],
    "generative ai":["genai", "llm", "large language model", "gpt", "gemini", "claude"],
    # --- Data ---
    "data analysis": ["analisis data", "data analyst", "exploratory data analysis", "eda"],
    "data visualization": ["visualisasi data", "matplotlib", "seaborn", "plotly", "tableau", "power bi"],
    "data wrangling":     ["data cleaning", "data preprocessing", "pembersihan data"],
    "statistics":         ["statistika", "statistik", "probability", "probabilitas"],
    # --- Web & Backend ---
    "fastapi":      ["fast api"],
    "express":      ["expressjs", "express.js"],
    "next.js":      ["nextjs", "next js"],
    "rest api":     ["restful", "restful api", "api", "web service"],
    # --- Tools ---
    "git":          ["github", "gitlab", "version control"],
    "docker":       ["containerization", "container"],
    "linux":        ["ubuntu", "bash", "shell scripting"],
    "google colab": ["colab", "google colaboratory"],
    # --- Soft Skills ---
    "problem solving":  ["pemecahan masalah", "analytical thinking"],
    "communication":    ["komunikasi", "presentasi", "public speaking"],
    "teamwork":         ["kolaborasi", "kerja tim", "team player"],
}

# Inverted index: alias → canonical
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_TAXONOMY.items():
    _ALIAS_TO_CANONICAL[canonical] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


# --------------------------------------------------------------------------- #
#  1. TEXT CLEANING                                                             #
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """
    Membersihkan teks mentah CV untuk keperluan NLP.

    Pipeline:
      1. Case folding
      2. Hapus karakter non-alfabet
      3. Hapus whitespace berlebih
      4. Stopword removal (NLTK Indonesian)
      5. Stemming (Sastrawi)

    Returns:
        str: Teks bersih siap di-tokenize / di-vectorize.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    cleaned_words = []
    for word in text.split():
        if word not in stop_words:
            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)


# --------------------------------------------------------------------------- #
#  2. SKILL EXTRACTION                                                          #
# --------------------------------------------------------------------------- #
def extract_skills(raw_text: str) -> list[str]:
    """
    Mengekstrak canonical skill dari teks CV menggunakan exact-match taxonomy.

    Catatan: Ini adalah implementasi rule-based sebagai baseline.
    Akan diperkuat dengan model embedding pada iterasi berikutnya.

    Args:
        raw_text: Teks CV mentah (belum di-clean, agar multi-word tetap terbaca).

    Returns:
        list[str]: Daftar skill unik (canonical form), sudah di-deduplikasi & diurutkan.
    """
    if not raw_text:
        return []

    text_lower = raw_text.lower()
    found: set[str] = set()

    # Coba cocokkan setiap alias dari yang terpanjang lebih dulu
    # (supaya "machine learning" tidak terpotong jadi "machine" + "learning")
    all_aliases = sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)
    for alias in all_aliases:
        # Gunakan word-boundary (\b) agar tidak partial-match
        pattern = r'\b' + re.escape(alias) + r'\b'
        if re.search(pattern, text_lower):
            canonical = _ALIAS_TO_CANONICAL[alias]
            found.add(canonical)

    return sorted(found)


# --------------------------------------------------------------------------- #
#  3. SKILL NORMALIZATION                                                       #
# --------------------------------------------------------------------------- #
def normalize_skills(raw_skills: list[str]) -> list[str]:
    """
    Menormalisasi daftar skill raw (dari input pengguna / parser) ke
    canonical form dalam taxonomy.

    Args:
        raw_skills: Daftar skill belum ternormalisasi.

    Returns:
        list[str]: Daftar skill yang sudah dinormalisasi & unik.
    """
    normalized: set[str] = set()
    for skill in raw_skills:
        key = skill.lower().strip()
        canonical = _ALIAS_TO_CANONICAL.get(key)
        if canonical:
            normalized.add(canonical)
        else:
            # Skill tidak ada di taxonomy — tetap disimpan apa adanya
            normalized.add(skill.strip())
    return sorted(normalized)


# --------------------------------------------------------------------------- #
#  4. PROFILE FORMALIZATION (sederhana, akan digantikan oleh GenAI Helper)    #
# --------------------------------------------------------------------------- #
def formalize_experience(raw_experience: str) -> str:
    """
    Memformalkan narasi pengalaman informal menjadi bullet profesional singkat.
    (Rule-based versi awal; produksi akan menggunakan Gemini via genai_helper.py)

    Args:
        raw_experience: Narasi pengalaman user (bebas/informal).

    Returns:
        str: Bullet poin profesional.
    """
    if not raw_experience.strip():
        return ""

    sentences = re.split(r'[.\n;]+', raw_experience)
    bullets = []
    for s in sentences:
        s = s.strip()
        if len(s) > 10:
            # Kapitalisasi + pastikan diawali kata kerja aktif
            bullets.append(f"• {s[0].upper()}{s[1:]}")

    return "\n".join(bullets)


# --------------------------------------------------------------------------- #
#  SELF-TEST                                                                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    sample_cv = """
    Halo! Saya adalah seorang Data Analyst. Saya suka menganalisis data-data besar
    menggunakan Python dan Pandas sejak tahun 2020. Saya juga berpengalaman dengan
    TensorFlow, Scikit-learn, dan SQL. Saya sering berkolaborasi dalam tim menggunakan Git.
    """

    print("=" * 55)
    print("  NLP UTILS — SELF TEST")
    print("=" * 55)

    print("\n[1] TEXT CLEANING")
    print(f"  Input : {sample_cv.strip()[:80]}...")
    print(f"  Output: {clean_text(sample_cv)[:80]}...")

    print("\n[2] SKILL EXTRACTION")
    skills = extract_skills(sample_cv)
    print(f"  Skills ditemukan ({len(skills)}): {skills}")

    print("\n[3] SKILL NORMALIZATION")
    raw = ["tensorflow2", "sklearn", "github", "Keras"]
    normalized = normalize_skills(raw)
    print(f"  Raw      : {raw}")
    print(f"  Canonical: {normalized}")

    print("\n[4] EXPERIENCE FORMALIZATION")
    exp = "saya pernah menganalisis data penjualan perusahaan. membuat dashboard visualisasi data"
    print(formalize_experience(exp))