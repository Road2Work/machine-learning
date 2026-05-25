"""Smoke test API Contract v2.3 Adaptive Session with FastAPI TestClient."""
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main

client = TestClient(main.app)

print("1) profile/extract-manual")
r = client.post("/v1/profile/extract-manual", json={
    "target_role": {"id": "role_data_analyst", "name": "Data Analyst"},
    "most_relevant_experience": "Saya pernah membuat dashboard penjualan untuk project kampus.",
    "skills_and_tools": "Excel, Python, SQL, data visualization",
    "project_experience": "Saya membersihkan data penjualan dan membuat visualisasi produk terlaris.",
    "achievement_or_impact": "Dashboard membantu tim memahami performa penjualan.",
})
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["source"] == "manual"
profile = {"skills": r.json()["skills"], "tools": r.json()["tools"], "professional_summary": r.json()["professional_summary"]}

print("2) role-fit/calculate-score")
r = client.post("/v1/role-fit/calculate-score", json={
    "profile": profile,
    "selected_role": {"id": "role_data_analyst", "name": "Data Analyst"},
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "fit_score" in r.json()

adaptive_memory = {
    "enabled": True,
    "previousSessionIds": ["session_old_001"],
    "previousInterviewSummary": "User cukup jelas, tetapi evidence dan impact masih lemah.",
    "previousDetectedWeaknesses": ["weak_evidence", "missing_impact", "weak_star_structure"],
    "previousEvidenceLevels": [2, 3],
    "askedQuestionHistory": [
        {
            "questionId": "q_old_001",
            "questionText": "Ceritakan pengalamanmu membuat dashboard.",
            "questionType": "main",
            "competencyTarget": "role_relevance_and_evidence",
            "askedAt": "2026-05-25T10:00:00Z",
        }
    ],
    "latestInterviewFeedback": "Tambahkan tools, kontribusi pribadi, dan hasil terukur.",
    "nextBestActions": [{"id": "nba_001", "title": "Perkuat evidence", "description": "Tambahkan impact."}],
    "improvementFocus": ["evidence_specificity", "star_structure"],
    "avoidRepeatedQuestions": True,
    "retryMode": False,
}

print("3) interview/build-context with adaptive memory")
r = client.post("/v1/interview/build-context", json={
    "profile_id": "profile_001",
    "profile": profile,
    "selected_role": {"id": "role_data_analyst", "name": "Data Analyst"},
    "role_fit": {"fit_score": 70, "gaps": ["Tableau"]},
    "practiceMode": "adaptive_from_history",
    "adaptivePracticeMemory": adaptive_memory,
})
print(r.status_code, r.json())
assert r.status_code == 200
interview_context = r.json()["interview_context"]
assert r.json()["practice_mode"] == "adaptive_from_history"

print("4) interview/generate-question self introduction")
r = client.post("/v1/interview/generate-question", json={
    "session_id": "session_001",
    "selected_role": {"id": "role_data_analyst", "name": "Data Analyst"},
    "interview_context": interview_context,
    "session_state": {"current_question_index": 1, "question_count": 5, "asked_questions": [], "first_question_required": True},
    "adaptivePracticeMemory": adaptive_memory,
    "practiceMode": "adaptive_from_history",
})
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["competency_target"] == "self_introduction"
assert r.json()["recording_policy"]["answerLimitSeconds"] == 90
assert r.json()["recording_policy"]["silenceAutoStopEnabled"] is False
question_text = r.json()["question_text"]

print("5) interview/generate-question adaptive from history")
r = client.post("/v1/interview/generate-question", json={
    "session_id": "session_001",
    "selected_role": {"id": "role_data_analyst", "name": "Data Analyst"},
    "interview_context": interview_context,
    "session_state": {"current_question_index": 2, "question_count": 5, "asked_questions": [question_text], "first_question_required": True},
    "adaptivePracticeMemory": adaptive_memory,
    "practiceMode": "adaptive_from_history",
})
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["generated_from"] in {"weakness_history", "next_best_action", "role_context", "retry_focus"}
assert r.json()["question_text"].strip().lower() != "ceritakan pengalamanmu membuat dashboard."

print("6) evaluate-answer")
r = client.post("/v1/interview/evaluate-answer", json={
    "session_id": "session_001",
    "question": {"id": "q1", "question_text": question_text, "question_type": "main", "competency_target": "self_introduction"},
    "answer": {
        "transcript_text": "Saya pernah membuat dashboard penjualan menggunakan Excel dan Python.",
        "stt_confidence": 0.91,
        "voice_metadata": {"duration_seconds": 35, "stopped_by": "user_mic_off", "max_duration_seconds": 90}
    },
    "profile": profile,
    "selected_role": {"id": "role_data_analyst", "name": "Data Analyst"},
    "session_state": {"clarification_count": 0, "max_clarification": 3}
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "answer_score" in r.json()
assert "created_at" in r.json()

print("7) interview/generate-clarification")
r = client.post("/v1/interview/generate-clarification", json={
    "question_text": question_text,
    "answer_text": "Saya pernah membuat dashboard penjualan.",
    "detected_weaknesses": ["missing_tools", "missing_impact", "weak_evidence"],
    "clarification_type": "weak_evidence",
    "selected_role": "Data Analyst"
})
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["question_type"] == "clarification"
assert r.json()["hrd_state"] == "clarifying"

print("8) predict-answer-quality")
r = client.post("/v1/model/predict-answer-quality", json={
    "transcript_text": "Saya menggunakan SQL dan Excel untuk membuat funnel analysis.",
    "features": {"word_count": 11, "has_metric": False, "has_tool": True, "evidence_level": 3}
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "predicted_quality" in r.json()

print("9) generate-result with adaptive session suggestion")
r = client.post("/v1/interview/generate-result", json={
    "session_id": "session_001",
    "selected_role": "Data Analyst",
    "answers": [{
        "question_text": question_text,
        "answer_text": "Saya pernah membuat dashboard penjualan menggunakan Excel dan Python.",
        "answer_score": 70,
        "evidence_level": 3,
        "detected_weaknesses": ["missing_impact"],
        "score_breakdown": {"role_relevance": 75, "star_structure": 55, "evidence_specificity": 50, "technical_accuracy": 70, "communication_clarity": 80, "self_awareness": 55}
    }]
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "adaptive_session_suggestion" in r.json()

print("10) dashboard/generate-summary locked and unlocked")
r = client.post("/v1/dashboard/generate-summary", json={"careerReadinessScore": 89, "dashboard": {}})
print("locked", r.status_code, r.json())
assert r.status_code == 403
r = client.post("/v1/dashboard/generate-summary", json={"careerReadinessScore": 90, "user": {"name": "Rizky"}, "selectedRole": {"name": "Data Analyst"}, "dashboard": {"strengths": ["Komunikasi"], "gaps": ["Evidence"]}})
print("unlocked", r.status_code, r.json())
assert r.status_code == 200
assert r.json()["download_ready"] is True

print("✅ Smoke test contract v2.3 Adaptive Session selesai.")
