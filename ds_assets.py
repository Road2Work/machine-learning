"""
ds_assets.py — Data Science resource loader for Road2Work.id AI service.

v2.3 — aligned with Road2Work/data-science repo layout
- Train/validation/test answer-quality datasets are loaded from:
  data_science_resources/data/03_processed/train_df.csv
  data_science_resources/data/03_processed/val_df.csv
  data_science_resources/data/03_processed/test_df.csv
- Raw guardrail assets are loaded from:
  data_science_resources/data/01_raw/
- Legacy flat layout is still supported for backward compatibility.
- Environment variables can override every asset path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

PROJECT_ROOT = Path(__file__).resolve().parent
DS_RESOURCES_DIR = Path(os.getenv("DS_RESOURCES_DIR", PROJECT_ROOT / "data_science_resources")).expanduser().resolve()
DS_RAW_DIR = Path(os.getenv("DS_RAW_DIR", DS_RESOURCES_DIR / "data" / "01_raw")).expanduser().resolve()
DS_PROCESSED_DIR = Path(os.getenv("DS_PROCESSED_DIR", DS_RESOURCES_DIR / "data" / "03_processed")).expanduser().resolve()
DS_INTERIM_DIR = Path(os.getenv("DS_INTERIM_DIR", DS_RESOURCES_DIR / "data" / "02_interim")).expanduser().resolve()
DS_ASSET_AUTO_RELOAD = os.getenv("DS_ASSET_AUTO_RELOAD", "true").lower() in {"1", "true", "yes", "y"}

# Canonical asset keys used by the AI service.
# "filename" is only a label; asset_path() resolves multiple candidate paths.
ASSET_FILENAMES: dict[str, str] = {
    "role_tree_dropdown": "role_tree_dropdown.json",
    "role_tree_dropdown_csv": "role_tree_dropdown.csv",
    "role_skill_matrix": "role_skill_matrix.json",
    "role_skill_matrix_csv": "role_skill_matrix.csv",
    "skill_taxonomy": "skill_taxonomy.json",
    "competency_map": "competency_map.json",
    "question_seed": "question_seed.json",
    "weakness_taxonomy": "weakness_taxonomy.json",
    "scoring_rubric": "scoring_rubric.json",
    "evidence_ladder_mapping": "evidence_ladder_mapping.json",
    # raw, unsplit answer dataset; kept only for fallback/EDA compatibility
    "answer_quality_dataset": "answer_dataset.csv",
    # official model splits from DS repo v2 layout
    "answer_quality_train": "train_df.csv",
    "answer_quality_val": "val_df.csv",
    "answer_quality_test": "test_df.csv",
    "answer_quality_manual_test": "answer_quality_manual_test.csv",
}

_ENV_BY_ASSET: dict[str, str] = {
    "role_tree_dropdown": "ROLE_TREE_DROPDOWN_PATH",
    "role_tree_dropdown_csv": "ROLE_TREE_DROPDOWN_CSV_PATH",
    "role_skill_matrix": "ROLE_SKILL_MATRIX_PATH",
    "role_skill_matrix_csv": "ROLE_SKILL_MATRIX_CSV_PATH",
    "skill_taxonomy": "SKILL_TAXONOMY_PATH",
    "competency_map": "COMPETENCY_MAP_PATH",
    "question_seed": "QUESTION_SEED_PATH",
    "weakness_taxonomy": "WEAKNESS_TAXONOMY_PATH",
    "scoring_rubric": "SCORING_RUBRIC_PATH",
    "evidence_ladder_mapping": "EVIDENCE_LADDER_MAPPING_PATH",
    "answer_quality_dataset": "ANSWER_QUALITY_DATASET_PATH",
    "answer_quality_train": "ANSWER_QUALITY_TRAIN_PATH",
    "answer_quality_val": "ANSWER_QUALITY_VAL_PATH",
    "answer_quality_test": "ANSWER_QUALITY_TEST_PATH",
    "answer_quality_manual_test": "ANSWER_QUALITY_MANUAL_TEST_PATH",
}

# Candidate paths are relative to DS_RESOURCES_DIR unless absolute.
# Order matters: official Road2Work/data-science layout first, old flat layout last.
_CANDIDATE_RELATIVE_PATHS: dict[str, list[str]] = {
    # Raw guardrail assets. In the DS repo, these are expected in data/01_raw.
    "role_tree_dropdown": ["data/01_raw/role_tree_dropdown.json", "01_raw/role_tree_dropdown.json", "role_tree_dropdown.json"],
    "role_tree_dropdown_csv": ["data/01_raw/role_tree_dropdown.csv", "01_raw/role_tree_dropdown.csv", "role_tree_dropdown.csv"],
    "role_skill_matrix": ["data/01_raw/role_skill_matrix.json", "01_raw/role_skill_matrix.json", "role_skill_matrix.json"],
    "role_skill_matrix_csv": ["data/01_raw/role_skill_matrix.csv", "01_raw/role_skill_matrix.csv", "role_skill_matrix.csv"],
    "skill_taxonomy": ["data/01_raw/skill_taxonomy.json", "01_raw/skill_taxonomy.json", "skill_taxonomy.json"],
    "competency_map": ["data/01_raw/competency_map.json", "01_raw/competency_map.json", "competency_map.json"],
    "question_seed": ["data/01_raw/question_seed.json", "01_raw/question_seed.json", "question_seed.json"],
    "weakness_taxonomy": ["data/01_raw/weakness_taxonomy.json", "01_raw/weakness_taxonomy.json", "weakness_taxonomy.json"],
    "scoring_rubric": ["data/01_raw/scoring_rubric.json", "01_raw/scoring_rubric.json", "scoring_rubric.json"],
    "evidence_ladder_mapping": ["data/01_raw/evidence_ladder_mapping.json", "01_raw/evidence_ladder_mapping.json", "evidence_ladder_mapping.json"],
    # Raw answer dataset is not used for model training once train/val/test exist.
    "answer_quality_dataset": [
        "data/01_raw/answer_dataset.csv",
        "01_raw/answer_dataset.csv",
        "answer_dataset.csv",
        "answer_quality_dataset_synthetic.csv",
    ],
    # Official processed split files.
    "answer_quality_train": [
        "data/03_processed/train_df.csv",
        "03_processed/train_df.csv",
        "data/03_processed/dataset_train.csv",
        "dataset_train.csv",
        "train_df.csv",
    ],
    "answer_quality_val": [
        "data/03_processed/val_df.csv",
        "03_processed/val_df.csv",
        "data/03_processed/dataset_val.csv",
        "dataset_val.csv",
        "val_df.csv",
    ],
    "answer_quality_test": [
        "data/03_processed/test_df.csv",
        "03_processed/test_df.csv",
        "data/03_processed/dataset_test.csv",
        "dataset_test.csv",
        "test_df.csv",
    ],
    "answer_quality_manual_test": [
        "data/03_processed/manual_test_df.csv",
        "data/03_processed/answer_quality_manual_test.csv",
        "03_processed/manual_test_df.csv",
        "answer_quality_manual_test.csv",
    ],
}

_cache: dict[str, Any] = {}
_cache_mtime: dict[str, float | None] = {}
_cache_path: dict[str, Path] = {}


def _env_path(asset_key: str) -> Path | None:
    env_name = _ENV_BY_ASSET.get(asset_key)
    value = os.getenv(env_name or "", "").strip() if env_name else ""
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _candidate_paths(asset_key: str) -> list[Path]:
    paths: list[Path] = []

    env_candidate = _env_path(asset_key)
    if env_candidate is not None:
        paths.append(env_candidate)

    for rel in _CANDIDATE_RELATIVE_PATHS.get(asset_key, [ASSET_FILENAMES[asset_key]]):
        rel_path = Path(rel)
        paths.append(rel_path if rel_path.is_absolute() else DS_RESOURCES_DIR / rel_path)

    # Extra explicit folder hints in case only DS_RAW_DIR/DS_PROCESSED_DIR are customized.
    filename = ASSET_FILENAMES[asset_key]
    if asset_key.startswith("answer_quality_train"):
        paths.append(DS_PROCESSED_DIR / "train_df.csv")
    elif asset_key.startswith("answer_quality_val"):
        paths.append(DS_PROCESSED_DIR / "val_df.csv")
    elif asset_key.startswith("answer_quality_test"):
        paths.append(DS_PROCESSED_DIR / "test_df.csv")
    elif asset_key == "answer_quality_dataset":
        paths.append(DS_RAW_DIR / "answer_dataset.csv")
    else:
        paths.append(DS_RAW_DIR / filename)

    # Legacy development fallback: asset next to source code.
    paths.append(PROJECT_ROOT / filename)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())
    return unique


def asset_path(asset_key: str) -> Path:
    """Resolve path for an asset.

    Priority:
    1. explicit environment variable path,
    2. official data-science repo layout,
    3. legacy flat layout,
    4. project-root fallback.
    """
    if asset_key not in ASSET_FILENAMES:
        raise KeyError(f"Unknown DS asset key: {asset_key}")

    candidates = _candidate_paths(asset_key)
    for path in candidates:
        if path.exists():
            return path
    # Return first candidate to make status/error messages actionable.
    return candidates[0]


def asset_candidates(asset_key: str) -> list[str]:
    """Return all candidate paths for debugging/admin status."""
    if asset_key not in ASSET_FILENAMES:
        raise KeyError(f"Unknown DS asset key: {asset_key}")
    return [str(path) for path in _candidate_paths(asset_key)]


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _should_reload(asset_key: str, path: Path) -> bool:
    if asset_key not in _cache:
        return True
    if not DS_ASSET_AUTO_RELOAD:
        return False
    return _cache_mtime.get(asset_key) != _mtime(path) or _cache_path.get(asset_key) != path


def load_json_asset(asset_key: str, fallback: Any | None = None) -> Any:
    path = asset_path(asset_key)
    if not _should_reload(asset_key, path):
        return _cache[asset_key]

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _cache[asset_key] = data
        _cache_mtime[asset_key] = _mtime(path)
        _cache_path[asset_key] = path
        print(f"[ds_assets] ✅ Loaded {asset_key}: {path}")
        return data
    except Exception as exc:
        print(f"[ds_assets] ⚠️ Failed loading {asset_key} from {path}: {exc}")
        data = fallback if fallback is not None else {}
        _cache[asset_key] = data
        _cache_mtime[asset_key] = _mtime(path)
        _cache_path[asset_key] = path
        return data


def load_csv_asset(asset_key: str):
    path = asset_path(asset_key)
    if not _should_reload(asset_key, path):
        return _cache[asset_key]

    if pd is None:
        raise RuntimeError("pandas belum terinstall. Jalankan: pip install pandas")

    try:
        df = pd.read_csv(path)
        _cache[asset_key] = df
        _cache_mtime[asset_key] = _mtime(path)
        _cache_path[asset_key] = path
        print(f"[ds_assets] ✅ Loaded {asset_key}: {path} ({df.shape[0]} rows)")
        return df
    except Exception as exc:
        print(f"[ds_assets] ⚠️ Failed loading {asset_key} from {path}: {exc}")
        df = pd.DataFrame() if pd is not None else None
        _cache[asset_key] = df
        _cache_mtime[asset_key] = _mtime(path)
        _cache_path[asset_key] = path
        return df


def reload_all() -> dict[str, Any]:
    """Clear cache lalu load ulang semua asset. Cocok dipanggil dari endpoint admin."""
    _cache.clear()
    _cache_mtime.clear()
    _cache_path.clear()
    status: dict[str, Any] = {}
    for key, filename in ASSET_FILENAMES.items():
        path = asset_path(key)
        if filename.endswith(".json"):
            data = load_json_asset(key, fallback={})
            status[key] = {
                "path": str(path),
                "exists": path.exists(),
                "type": type(data).__name__,
                "size": len(data) if hasattr(data, "__len__") else None,
                "candidates": asset_candidates(key),
            }
        else:
            try:
                df = load_csv_asset(key)
                status[key] = {
                    "path": str(path),
                    "exists": path.exists(),
                    "rows": int(getattr(df, "shape", [0])[0]),
                    "candidates": asset_candidates(key),
                }
            except Exception as exc:
                status[key] = {"path": str(path), "exists": path.exists(), "error": str(exc), "candidates": asset_candidates(key)}
    return status


def asset_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "resources_dir": str(DS_RESOURCES_DIR),
        "raw_dir": str(DS_RAW_DIR),
        "processed_dir": str(DS_PROCESSED_DIR),
        "interim_dir": str(DS_INTERIM_DIR),
        "auto_reload": DS_ASSET_AUTO_RELOAD,
        "expected_layout": {
            "raw_assets": "data_science_resources/data/01_raw/",
            "processed_splits": "data_science_resources/data/03_processed/train_df.csv, val_df.csv, test_df.csv",
        },
        "assets": {},
    }
    for key, filename in ASSET_FILENAMES.items():
        path = asset_path(key)
        status["assets"][key] = {
            "filename": filename,
            "path": str(path),
            "exists": path.exists(),
            "mtime": _mtime(path),
            "candidates": asset_candidates(key),
        }
    return status


# --------------------------------------------------------------------------- #
# Domain / role helpers
# --------------------------------------------------------------------------- #
def get_role_tree() -> dict[str, Any]:
    return load_json_asset("role_tree_dropdown", fallback={"domains": []})


def get_target_roles() -> list[str]:
    roles: list[str] = []
    for domain in get_role_tree().get("domains", []):
        for family in domain.get("role_families", []):
            for role in family.get("target_roles", []):
                name = role.get("name")
                if name:
                    roles.append(str(name))
    if roles:
        return roles
    return list(get_role_skill_matrix().keys())


def get_role_family_for_role(target_role: str) -> dict[str, str] | None:
    for domain in get_role_tree().get("domains", []):
        for family in domain.get("role_families", []):
            for role in family.get("target_roles", []):
                if str(role.get("name", "")).lower() == target_role.lower():
                    return {
                        "domain": str(domain.get("name", "")),
                        "role_family": str(family.get("name", "")),
                        "target_role": str(role.get("name", "")),
                    }
    detail = get_role_skill_detail().get(target_role)
    if isinstance(detail, dict):
        return {"domain": str(detail.get("domain", "")), "role_family": str(detail.get("role_family", "")), "target_role": target_role}
    return None


# --------------------------------------------------------------------------- #
# Role skill matrix helpers
# --------------------------------------------------------------------------- #
def get_role_skill_detail() -> dict[str, dict[str, Any]]:
    raw = load_json_asset("role_skill_matrix", fallback={})
    if not isinstance(raw, dict):
        return {}
    return raw


def _skills_from_role_entry(entry: Any) -> list[str]:
    if isinstance(entry, list):
        result = []
        for item in entry:
            if isinstance(item, str):
                result.append(item.lower())
            elif isinstance(item, dict) and item.get("skill"):
                result.append(str(item["skill"]).lower())
        return result
    if isinstance(entry, dict):
        skills = entry.get("required_skills", [])
        return _skills_from_role_entry(skills)
    return []


def get_role_skill_matrix() -> dict[str, list[str]]:
    raw = get_role_skill_detail()
    return {str(role): _skills_from_role_entry(entry) for role, entry in raw.items()}


def get_role_skill_weights(role: str) -> dict[str, float]:
    entry = get_role_skill_detail().get(role, {})
    result: dict[str, float] = {}
    skills = entry.get("required_skills", []) if isinstance(entry, dict) else []
    for item in skills:
        if isinstance(item, dict) and item.get("skill"):
            try:
                result[str(item["skill"]).lower()] = float(item.get("weight", 1.0))
            except (TypeError, ValueError):
                result[str(item["skill"]).lower()] = 1.0
        elif isinstance(item, str):
            result[item.lower()] = 1.0
    return result


# --------------------------------------------------------------------------- #
# Interview guardrail helpers
# --------------------------------------------------------------------------- #
def get_competency_map() -> dict[str, Any]:
    return load_json_asset("competency_map", fallback={})


def get_competencies(role: str) -> list[dict[str, Any]]:
    data = get_competency_map().get(role, {})
    if isinstance(data, dict):
        comps = data.get("competencies", [])
        return comps if isinstance(comps, list) else []
    if isinstance(data, list):
        return data
    return []


def get_question_seed() -> dict[str, Any]:
    return load_json_asset("question_seed", fallback={})


def get_question_seeds(role: str, question_type: str | None = None) -> list[dict[str, Any]]:
    seeds = get_question_seed().get(role, [])
    if not isinstance(seeds, list):
        return []
    if question_type:
        return [s for s in seeds if isinstance(s, dict) and s.get("question_type") == question_type]
    return [s for s in seeds if isinstance(s, dict)]


def get_weakness_taxonomy() -> dict[str, Any]:
    return load_json_asset("weakness_taxonomy", fallback={})


def get_scoring_rubric() -> dict[str, Any]:
    return load_json_asset("scoring_rubric", fallback={})


def get_scoring_weights() -> dict[str, float]:
    components = get_scoring_rubric().get("components", {})
    weights: dict[str, float] = {}
    for key, value in components.items():
        if isinstance(value, dict):
            try:
                weights[key] = float(value.get("weight", 0))
            except (TypeError, ValueError):
                weights[key] = 0.0
    return weights


def get_need_clarification_rule() -> dict[str, Any]:
    return get_scoring_rubric().get("need_clarification_rule", {})


def get_evidence_ladder_mapping() -> dict[str, Any]:
    return load_json_asset("evidence_ladder_mapping", fallback={"levels": []})


def get_evidence_levels() -> list[dict[str, Any]]:
    levels = get_evidence_ladder_mapping().get("levels", [])
    return levels if isinstance(levels, list) else []


def get_skill_taxonomy() -> dict[str, Any]:
    return load_json_asset("skill_taxonomy", fallback={})


def answer_quality_dataset_path() -> Path:
    """Backward-compatible raw dataset path. Prefer train/val/test for model training."""
    train_path = asset_path("answer_quality_train")
    return train_path if train_path.exists() else asset_path("answer_quality_dataset")


def answer_quality_train_path() -> Path:
    return asset_path("answer_quality_train")


def answer_quality_val_path() -> Path:
    return asset_path("answer_quality_val")


def answer_quality_test_path() -> Path:
    return asset_path("answer_quality_test")


def answer_quality_manual_test_path() -> Path:
    """Dataset manual/realistic untuk external validation, tidak dipakai training utama."""
    return asset_path("answer_quality_manual_test")


def load_answer_quality_dataset():
    return load_csv_asset("answer_quality_dataset")


def load_answer_quality_train():
    return load_csv_asset("answer_quality_train")


def load_answer_quality_val():
    return load_csv_asset("answer_quality_val")


def load_answer_quality_test():
    return load_csv_asset("answer_quality_test")


def load_answer_quality_manual_test():
    return load_csv_asset("answer_quality_manual_test")
