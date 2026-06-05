"""
nlp_utils.py â€” NLP Preprocessing & Context Extraction Pipeline
Road2Work AI | CC26-PSU050

v1.1.0 â€” connected to Data Science resources
- Membaca skill_taxonomy.json dari data_science_resources/ atau env SKILL_TAXONOMY_PATH.
- Bisa membaca format taxonomy DS yang nested per role_family.
- Skill dari role_skill_matrix.json selalu dijadikan canonical identity agar role-fit tidak rusak.
- Menyediakan extract_from_short_profile() untuk jalur Isi Profil Singkat.
"""

from __future__ import annotations

import re
from typing import Any

try:  # optional dependency agar service tetap bisa jalan saat package belum ada
    from nltk.corpus import stopwords
    _stop_words = set(stopwords.words("indonesian"))
except Exception:  # pragma: no cover
    _stop_words = {
        "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "atau", "pada",
        "saya", "adalah", "sebagai", "dalam", "ini", "itu", "karena",
    }

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except Exception:  # pragma: no cover
    _stemmer = None

try:
    from ds_assets import get_role_skill_matrix, get_skill_taxonomy
except Exception:  # pragma: no cover
    get_role_skill_matrix = None  # type: ignore
    get_skill_taxonomy = None  # type: ignore


_TAXONOMY_FALLBACK: dict[str, list[str]] = {
    "python": ["python3", "py"],
    "javascript": ["js", "javascript"],
    "typescript": ["ts", "typescript"],
    "sql": ["mysql", "postgresql", "sqlite", "postgres", "sql server"],
    "r": ["rstudio", "r language"],
    "java": ["java se", "java ee"],
    "c++": ["cpp", "c plus plus"],
    "tensorflow": ["tf", "keras", "tensorflow2"],
    "pytorch": ["torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "pandas": ["pd"],
    "numpy": ["np"],
    "nlp": ["natural language processing", "text mining", "text classification"],
    "deep learning": ["dl", "neural network", "ann", "cnn", "rnn", "lstm"],
    "machine learning": ["ml", "supervised learning", "unsupervised learning"],
    "generative ai": ["genai", "llm", "large language model", "prompt engineering"],
    "data analysis": ["analisis data", "data analyst", "exploratory data analysis", "eda"],
    "data visualization": ["visualisasi data", "matplotlib", "seaborn", "plotly", "tableau", "power bi", "dashboard"],
    "data wrangling": ["data cleaning", "data preprocessing", "pembersihan data", "data cleansing"],
    "statistics": ["statistika", "statistik", "probability", "probabilitas"],
    "fastapi": ["fast api"],
    "express": ["expressjs", "express.js"],
    "next.js": ["nextjs", "next js"],
    "react": ["react.js", "reactjs"],
    "rest api": ["restful", "restful api", "api", "web service"],
    "git": ["github", "gitlab", "version control"],
    "docker": ["containerization", "container"],
    "linux": ["ubuntu", "bash", "shell scripting"],
    "excel": ["microsoft excel", "spreadsheet"],
    "problem solving": ["pemecahan masalah", "analytical thinking"],
    "communication": ["komunikasi", "presentasi", "public speaking"],
    "teamwork": ["kolaborasi", "kerja tim", "team player"],
}

SKILL_TAXONOMY: dict[str, list[str]] = {}
_ALIAS_TO_CANONICAL: dict[str, str] = {}
_ROLE_MATRIX_SKILLS: set[str] = set()


def _safe_lower(value: Any) -> str:
    return str(value).strip().lower()


def _role_matrix_skill_set() -> set[str]:
    if get_role_skill_matrix is None:
        return set()
    try:
        matrix = get_role_skill_matrix()
        return {skill.lower() for skills in matrix.values() for skill in skills}
    except Exception:
        return set()


def _add_alias(index: dict[str, str], alias: str, canonical: str) -> None:
    alias = _safe_lower(alias)
    canonical = _safe_lower(canonical)
    if alias and canonical:
        index[alias] = canonical


def _build_alias_index(taxonomy: dict[str, Any]) -> dict[str, str]:
    """
    Support 2 format:
    1) {canonical_skill: [aliases]}
    2) {role_family: {skill_group: [aliases]}}

    Rule penting:
    - Skill yang ada di role_skill_matrix diprioritaskan sebagai identity canonical.
    - Alias yang juga merupakan skill matrix tidak dipaksa ke group besar.
      Contoh: "pandas" tetap "pandas", bukan "python".
    """
    index: dict[str, str] = {}

    # Baseline fallback taxonomy
    for canonical, aliases in _TAXONOMY_FALLBACK.items():
        _add_alias(index, canonical, canonical)
        for alias in aliases:
            _add_alias(index, alias, canonical)

    # Role matrix skills harus identity supaya role-fit akurat.
    for skill in _ROLE_MATRIX_SKILLS:
        _add_alias(index, skill, skill)

    for key, value in taxonomy.items():
        if isinstance(value, list):
            canonical = _safe_lower(key)
            _add_alias(index, canonical, canonical)
            for alias in value:
                alias_key = _safe_lower(alias)
                _add_alias(index, alias_key, alias_key if alias_key in _ROLE_MATRIX_SKILLS else canonical)
        elif isinstance(value, dict):
            # Nested role family -> skill group -> aliases
            for skill_group, aliases in value.items():
                group_canonical = _safe_lower(skill_group)
                _add_alias(index, group_canonical, group_canonical)
                if isinstance(aliases, list):
                    for alias in aliases:
                        alias_key = _safe_lower(alias)
                        _add_alias(index, alias_key, alias_key if alias_key in _ROLE_MATRIX_SKILLS else group_canonical)

    return index


def reload_taxonomy(path: str | None = None) -> str:
    """Reload skill taxonomy dari Data Science resources. Param path dipertahankan untuk kompatibilitas."""
    global SKILL_TAXONOMY, _ALIAS_TO_CANONICAL, _ROLE_MATRIX_SKILLS

    _ROLE_MATRIX_SKILLS = _role_matrix_skill_set()
    try:
        loaded = get_skill_taxonomy() if get_skill_taxonomy is not None and path is None else {}
        if not isinstance(loaded, dict) or not loaded:
            loaded = _TAXONOMY_FALLBACK
        SKILL_TAXONOMY = loaded
        _ALIAS_TO_CANONICAL = _build_alias_index(SKILL_TAXONOMY)
        print(f"[nlp_utils] âœ… Taxonomy aktif ({len(_ALIAS_TO_CANONICAL)} alias, {len(_ROLE_MATRIX_SKILLS)} role skills).")
        return "json" if loaded is not _TAXONOMY_FALLBACK else "fallback"
    except Exception as exc:
        print(f"[nlp_utils] âš ï¸ Gagal load taxonomy DS: {exc}. Pakai fallback.")
        SKILL_TAXONOMY = _TAXONOMY_FALLBACK.copy()
        _ALIAS_TO_CANONICAL = _build_alias_index(SKILL_TAXONOMY)
        return "fallback"


_taxonomy_source = reload_taxonomy()


def clean_text(text: str) -> str:
    """Case folding, remove symbol, stopword removal, optional stemming."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        if word in _stop_words:
            continue
        words.append(_stemmer.stem(word) if _stemmer else word)
    return " ".join(words)


def extract_skills(raw_text: str) -> list[str]:
    """Extract canonical skill dari CV/profile memakai exact alias matching."""
    if not raw_text:
        return []
    text = f" {raw_text.lower()} "
    found: set[str] = set()
    # Lebih panjang dulu agar "react native" tertangkap sebelum "react".
    for alias, canonical in sorted(_ALIAS_TO_CANONICAL.items(), key=lambda kv: len(kv[0]), reverse=True):
        escaped = re.escape(alias)
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, text):
            found.add(canonical)
    return sorted(found)


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalisasi list skill bebas menjadi canonical skill."""
    normalized: list[str] = []
    seen: set[str] = set()
    for skill in skills or []:
        key = _safe_lower(skill)
        if not key:
            continue
        canonical = _ALIAS_TO_CANONICAL.get(key, key)
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return normalized



_CONTACT_OR_HEADER_RE = re.compile(
    r"(\b[\w.+-]+@[\w.-]+\.\w+\b|https?://|www\.|linkedin\.com|github\.com|\+?\d[\d\s().-]{7,}|\bdepok\b|\bjakarta\b|\bbandung\b)",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(
    r"^(about|about me|profile|professional profile|professional summary|summary|objective|ringkasan|profil|tentang saya|pengalaman|experience|projects?|project experience|education|skills?|tools?|achievement|achievements?|pencapaian)\s*:?$",
    re.IGNORECASE,
)
_SUMMARY_HEADING_RE = re.compile(
    r"^(about|about me|profile|professional profile|professional summary|summary|objective|ringkasan|profil|tentang saya)\s*:?$",
    re.IGNORECASE,
)
_NEXT_SECTION_RE = re.compile(
    r"^(experience|pengalaman|projects?|project experience|education|pendidikan|skills?|tools?|achievement|achievements?|pencapaian|certifications?|sertifikasi|organizations?|organisasi)\s*:?$",
    re.IGNORECASE,
)
_TOOL_ALIASES: dict[str, list[str]] = {
    "python": ["python", "python3"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "java": ["java"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "sql server", "sqlite", "bigquery"],
    "excel": ["excel", "microsoft excel", "spreadsheet"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "google looker studio": ["google looker studio", "looker studio", "looker"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow": ["tensorflow", "keras"],
    "pytorch": ["pytorch", "torch"],
    "opencv": ["opencv", "open cv", "cv2"],
    "matplotlib": ["matplotlib"],
    "seaborn": ["seaborn"],
    "plotly": ["plotly"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs", "next js"],
    "node.js": ["node.js", "nodejs", "node"],
    "express": ["express", "express.js", "expressjs"],
    "fastapi": ["fastapi", "fast api"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": [" git ", "gitlab", "version control"],
    "figma": ["figma"],
    "tailwind": ["tailwind", "tailwind css"],
}
_TOOL_CANONICALS = set(_TOOL_ALIASES.keys())
_CONCEPT_SKILLS = {
    "data visualization",
    "data analysis",
    "data wrangling",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "statistics",
    "statistical analysis",
    "problem solving",
    "communication",
    "teamwork",
    "frontend",
    "backend",
    "database",
    "rest api",
    "generative ai",
    "mlops",
}
_VISUALIZATION_TOOLS = {"tableau", "power bi", "google looker studio", "matplotlib", "seaborn", "plotly"}
_DATA_TOOLS = {"python", "sql", "excel", "pandas", "numpy", "scikit-learn"}


def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        value = str(item).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if limit and len(result) >= limit:
            break
    return result


def _is_noise_line(line: str) -> bool:
    clean = line.strip(" -\\u2022\\t|,")
    if not clean:
        return True
    if _CONTACT_OR_HEADER_RE.search(clean):
        return True
    letters = re.sub(r"[^A-Za-z]", "", clean)
    if len(clean) <= 95 and len(letters) >= 8 and clean.upper() == clean and not _SECTION_HEADING_RE.match(clean):
        return True
    if len(clean.split()) <= 4 and not _SECTION_HEADING_RE.match(clean):
        return True
    return False


def remove_cv_header_and_contact(raw_text: str) -> str:
    """Buang header identitas/kontak agar extraction tidak salah membaca headline CV."""
    lines = [line.strip() for line in (raw_text or "").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    for index, line in enumerate(lines):
        if not line:
            continue
        if _CONTACT_OR_HEADER_RE.search(line):
            continue
        # Bagian 8 baris pertama biasanya nama, headline, lokasi, kontak. Jangan jadikan sumber skill/summary.
        if index < 8 and _is_noise_line(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _compact_summary(text: str, max_chars: int = 700) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip(" -•\t|")
    compact = _CONTACT_OR_HEADER_RE.sub("", compact)
    compact = re.sub(r"\s+", " ", compact).strip(" -•\t|")
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rsplit(" ", 1)[0].strip() + "..."


def extract_profile_summary(raw_text: str, ai_summary: str | None = None) -> str:
    """Ambil About/Profile/Summary dari CV, bukan header nama/kontak."""
    lines = [line.strip() for line in (raw_text or "").replace("\r", "\n").split("\n") if line.strip()]
    for index, line in enumerate(lines):
        if not _SUMMARY_HEADING_RE.match(line):
            continue
        collected: list[str] = []
        for next_line in lines[index + 1:]:
            if _NEXT_SECTION_RE.match(next_line):
                break
            if _is_noise_line(next_line):
                continue
            collected.append(next_line)
            if len(" ".join(collected)) >= 450:
                break
        candidate = _compact_summary(" ".join(collected))
        if len(candidate.split()) >= 12:
            return candidate

    if ai_summary and not _is_noise_line(ai_summary):
        candidate = _compact_summary(ai_summary)
        if len(candidate.split()) >= 12:
            return candidate

    searchable = remove_cv_header_and_contact(raw_text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=[.!?])\s+", searchable) if p.strip()]
    for paragraph in paragraphs:
        if _is_noise_line(paragraph):
            continue
        candidate = _compact_summary(paragraph)
        if len(candidate.split()) >= 12:
            return candidate
    return ""


def extract_tools(raw_text: str, extracted_skills: list[str] | None = None) -> list[str]:
    searchable = f" {remove_cv_header_and_contact(raw_text).lower()} "
    tools: list[str] = []
    for canonical, aliases in _TOOL_ALIASES.items():
        for alias in aliases:
            if alias.strip() == "git":
                pattern = r"(?<![a-z0-9])git(?![a-z0-9])"
            else:
                pattern = rf"(?<![a-z0-9]){re.escape(alias.strip().lower())}(?![a-z0-9])"
            if re.search(pattern, searchable):
                tools.append(canonical)
                break
    for skill in extracted_skills or []:
        key = _safe_lower(skill)
        if key in _TOOL_CANONICALS:
            tools.append(key)
    return _dedupe_keep_order(tools, limit=12)


def split_skills_and_tools(raw_text: str, extracted_skills: list[str]) -> tuple[list[str], list[str]]:
    tools = extract_tools(raw_text, extracted_skills)
    skills: list[str] = []
    for skill in extracted_skills or []:
        key = _safe_lower(skill)
        if not key or key in _TOOL_CANONICALS:
            continue
        skills.append(key)
    if _VISUALIZATION_TOOLS.intersection(tools):
        skills.append("data visualization")
    if _DATA_TOOLS.intersection(tools):
        skills.append("data analysis")
    if {"tensorflow", "pytorch", "scikit-learn", "opencv"}.intersection(tools):
        skills.append("machine learning")
    return _dedupe_keep_order(skills, limit=12), _dedupe_keep_order(tools, limit=12)

def formalize_experience(raw_text: str) -> str:
    if not raw_text or not raw_text.strip():
        return ""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", raw_text.strip())
    bullets = []
    for sentence in sentences:
        cleaned = sentence.strip(" -\\u2022\\t")
        if len(cleaned) >= 15:
            bullets.append(f"- {cleaned}")
        if len(bullets) >= 5:
            break
    return "\n".join(bullets)


def extract_evidence_signals(text: str) -> dict[str, Any]:
    lower = (text or "").lower()
    metrics = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|persen|jam|menit|hari|bulan|tahun|x)?", lower)
    impact_keywords = re.findall(
        r"\b(meningkat\w*|mengurangi|mempercepat|membantu|efisiensi|akurasi|hemat|optimal|otomatisasi|dashboard|rekomendasi|menurunkan|mengurangi|memperbaiki)\b",
        lower,
    )
    contribution_keywords = re.findall(
        r"\b(saya|membuat|mengembangkan|menganalisis|bertanggung jawab|memimpin|mengelola|mendesain|membangun|mengimplementasikan)\b",
        lower,
    )
    project_keywords = re.findall(
        r"\b(project|proyek|magang|organisasi|freelance|lomba|bootcamp|tugas akhir|skripsi|client|tim|perusahaan|kampus)\b",
        lower,
    )
    return {
        "has_metric": bool(metrics),
        "metrics": metrics[:10],
        "impact_keywords": list(dict.fromkeys(impact_keywords))[:10],
        "contribution_keywords": list(dict.fromkeys(contribution_keywords))[:10],
        "project_keywords": list(dict.fromkeys(project_keywords))[:10],
    }


def calculate_initial_evidence_score(text: str, skills: list[str]) -> int:
    signals = extract_evidence_signals(text)
    score = 20
    score += min(len(skills or []) * 5, 25)
    if signals["project_keywords"]:
        score += 15
    if signals["contribution_keywords"]:
        score += 15
    if signals["impact_keywords"]:
        score += 15
    if signals["has_metric"]:
        score += 10
    return max(0, min(100, score))


def extract_experience_summary(text: str, max_items: int = 5) -> list[str]:
    if not text:
        return []
    keywords = (
        "project", "proyek", "magang", "organisasi", "freelance", "lomba",
        "membuat", "mengembangkan", "menganalisis", "dashboard", "model",
        "aplikasi", "sistem", "website", "api", "data", "cloud", "testing",
        "automation", "security", "mobile",
    )
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    selected: list[str] = []
    for sentence in sentences:
        clean = sentence.strip(" -\\u2022\\t")
        if len(clean) < 15:
            continue
        if any(k in clean.lower() for k in keywords):
            selected.append(clean)
        if len(selected) >= max_items:
            break
    return selected


def extract_from_short_profile(profile_text: str, target_role: str = "posisi yang dipilih") -> dict[str, Any]:
    if not isinstance(profile_text, str) or not profile_text.strip():
        raise ValueError("Profil singkat tidak boleh kosong.")
    raw_text = profile_text.strip()
    extracted_skills = normalize_skills(extract_skills(raw_text))
    skills, tools = split_skills_and_tools(raw_text, extracted_skills)
    evidence_signals = extract_evidence_signals(raw_text)
    experience_summary = extract_experience_summary(raw_text)
    profile_summary = raw_text if len(raw_text) <= 800 else raw_text[:800].rsplit(" ", 1)[0] + "..."
    return {
        "source": "short_profile",
        "target_role": target_role,
        "raw_text": raw_text,
        "cleaned_text": clean_text(raw_text),
        "skills": skills,
        "tools": tools,
        "experience_summary": experience_summary,
        "profile_summary": profile_summary,
        "evidence_signals": evidence_signals,
        "initial_evidence_score": calculate_initial_evidence_score(raw_text, skills + tools),
    }

if __name__ == "__main__":
    sample = "Saya membuat dashboard penjualan menggunakan Python, SQL, Pandas, dan Power BI. Dashboard ini meningkatkan efisiensi laporan 30%."
    print("Taxonomy source:", _taxonomy_source)
    print(extract_from_short_profile(sample, target_role="Data Analyst"))









