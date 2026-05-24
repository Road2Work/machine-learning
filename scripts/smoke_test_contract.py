"""Smoke test API contract with FastAPI TestClient."""
from fastapi.testclient import TestClient
import main

client = TestClient(main.app)

profile_payload = {
    "target_role_id": "role_data_analyst",
    "target_role_name": "Data Analyst",
    "most_relevant_experience": "Saya pernah membuat dashboard penjualan untuk project kampus.",
    "skills_and_tools": "Excel, Python, data visualization, data cleaning",
    "project_experience": "Saya membersihkan data penjualan, membuat visualisasi produk terlaris, dan menyusun insight.",
    "achievement_or_impact": "Dashboard membantu tim memahami performa penjualan dengan lebih mudah.",
}

print("1) extract-profile")
r = client.post("/v1/context/extract-profile", json=profile_payload)
print(r.status_code, r.json())
assert r.status_code == 200
assert "profile_summary" in r.json()

next_question_payload = {
    "session_id": "session_001",
    "target_role": {"id": "role_data_analyst", "role_name": "Data Analyst", "role_family": "Data & AI"},
    "interview_context": {
        "skills": ["Data Cleaning", "Data Visualization"],
        "tools": ["Excel", "Python"],
        "experience_summary": "Created sales dashboard for academic project.",
        "evidence_items": ["Created sales dashboard", "Used Excel and Python"],
    },
    "session_state": {"question_index": 1, "total_main_questions": 5, "asked_questions": [], "clarification_count": 0, "detected_weaknesses": []},
    "question_seed": [],
    "competency_map": [],
}
print("2) next-question")
r = client.post("/v1/interview/next-question", json=next_question_payload)
print(r.status_code, r.json())
assert r.status_code == 200
assert "question_text" in r.json()

question_text = r.json()["question_text"]
eval_payload = {
    "session_id": "session_001",
    "question": {"id": "question_001", "question_text": question_text, "question_type": "main", "competency_target": "role_relevance_and_experience"},
    "answer": {"transcript_text": "Saya pernah membuat dashboard penjualan.", "stt_confidence": 0.91},
    "target_role": {"id": "role_data_analyst", "role_name": "Data Analyst"},
    "interview_context": next_question_payload["interview_context"],
    "score_history": [],
    "clarification_count": 0,
}
print("3) evaluate-answer")
r = client.post("/v1/interview/evaluate-answer", json=eval_payload)
print(r.status_code, r.json())
assert r.status_code == 200
assert "answer_score" in r.json()

print("4) clarifying-question")
r = client.post("/v1/interview/clarifying-question", json={
    "target_role": "Data Analyst",
    "question_text": question_text,
    "answer_text": "Saya pernah membuat dashboard penjualan.",
    "detected_weaknesses": ["missing_tools", "missing_impact"],
    "clarification_type": "weak_evidence",
})
print(r.status_code, r.json())
assert r.status_code == 200
assert r.json()["question_type"] == "clarification"

print("5) predict-answer-quality")
r = client.post("/v1/model/predict-answer-quality", json={
    "answer_text": "Saya pernah membuat dashboard penjualan menggunakan Excel dan Python.",
    "features": {"word_count": 9, "has_tools": True, "has_impact": False, "has_metric": False, "has_personal_contribution": False, "evidence_level": 2},
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "predicted_quality" in r.json()

print("6) generate-result")
r = client.post("/v1/interview/generate-result", json={
    "session_id": "session_001",
    "target_role": "Data Analyst",
    "answers": [{
        "question_text": question_text,
        "answer_text": "Saya pernah membuat dashboard penjualan.",
        "score_breakdown": {"role_relevance": 75, "star_structure": 35, "evidence_specificity": 30, "technical_accuracy": 45, "communication_clarity": 80, "self_awareness": 50},
        "evidence_level": 3,
        "detected_weaknesses": ["weak_evidence"],
    }],
})
print(r.status_code, r.json())
assert r.status_code == 200
assert "final_score" in r.json()

print("All contract smoke tests passed.")
