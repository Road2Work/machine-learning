# Road2Work AI Service — Data Science Integration

Package ini sudah dihubungkan dengan file JSON/CSV dari tim Data Science melalui folder `data_science_resources/`.

## Asset yang dibaca AI service

- `role_tree_dropdown.json` → cascading dropdown Domain → Role Family → Target Role.
- `role_skill_matrix.json` → role-fit helper dan guardrail skill per role.
- `skill_taxonomy.json` → ekstraksi dan normalisasi skill dari CV/profil.
- `competency_map.json` → target competency per role untuk adaptive question.
- `question_seed.json` → seed pertanyaan yang diparafrase oleh GenAI.
- `weakness_taxonomy.json` → weakness tag dan template clarification.
- `scoring_rubric.json` → bobot scoring dan rule clarification.
- `evidence_ladder_mapping.json` → definisi level Evidence Ladder.
- `answer_quality_dataset_synthetic.csv` → training dataset TensorFlow answer quality model.

## Cara terbaik menghubungkan repo Data Science

Di repo AI/machine-learning, jadikan repo Data Science sebagai submodule:

```bash
git submodule add https://github.com/Road2Work/data-science.git data_science_resources
git submodule update --init --recursive
```

Lalu copy `.env.example` menjadi `.env`:

```bash
cp .env.example .env
```

Pastikan:

```env
DS_RESOURCES_DIR=./data_science_resources
```

Jalankan FastAPI:

```bash
uvicorn main:app --reload
```

Cek asset:

```bash
curl http://localhost:8000/v1/admin/ds-assets/status
curl http://localhost:8000/v1/roles/tree
```

## Ketika Data Science update data

Cara manual:

```bash
git submodule update --remote --merge data_science_resources
curl -X POST http://localhost:8000/v1/admin/reload-ds-assets
```

Atau pakai script:

```bash
./scripts/update_ds_resources.sh
```

Jika deployment memakai multi-worker atau Docker production, tetap lebih aman restart service setelah update submodule.

## Kenapa pakai submodule?

Submodule cocok untuk kondisi capstone karena:

1. Versi data Data Science terkunci sebagai commit tertentu.
2. AI Engineer bisa update data tanpa copy-paste manual.
3. Riwayat perubahan data tetap terpisah di repo Data Science.
4. Kalau ada error, commit data bisa dirollback.

Alternatif yang lebih scalable untuk production adalah membuat package Python khusus `road2work-ds-assets` atau menyimpan asset di object storage/database. Tapi untuk project tim dan GitHub organization, submodule adalah opsi paling rapi dan mudah.

## Training model TensorFlow

```bash
python scripts/train_answer_quality_model.py
```

Model akan membaca dataset dari:

```text
data_science_resources/answer_quality_dataset_synthetic.csv
```

Output model:

```text
models/answer_quality_model.keras
models/answer_quality_tokenizer.json
models/answer_quality_meta.json
```

Model punya dua output:

- `answer_quality` → klasifikasi Weak / Average / Strong untuk Accuracy.
- `readiness_score` → skor 0.0–1.0 untuk MAE.

## Smoke test FastAPI tanpa uvicorn

```bash
python scripts/smoke_test_api.py
```

## Update v1.2 — Evidence Level, Manual Test, Confusion Matrix

Perbaikan tambahan:

1. **Evidence level inference diperjelas**
   - Input model tetap memakai `evidence_level` dalam bentuk normalisasi 0.2–1.0.
   - Response API sekarang menampilkan `features.evidence_level` sebagai level asli 1–5.
   - Response juga menampilkan `features.evidence_level_norm` agar tetap transparan untuk debugging.

2. **Manual realistic test set**
   - File baru: `data_science_resources/answer_quality_manual_test.csv`.
   - Isi: 54 data manual, balanced 18 Weak / 18 Average / 18 Strong.
   - Dataset ini tidak dipakai untuk training utama, tetapi dipakai sebagai external validation agar evaluasi tidak hanya synthetic.

3. **Confusion matrix & classification report**
   - Function baru: `evaluate_saved_model_detailed()` di `model_builder.py`.
   - Script baru: `scripts/evaluate_realistic_validation.py`.
   - Endpoint baru: `GET /v1/model/evaluation-report?dataset=manual`.

4. **Validasi lebih realistis**
   - Jalankan setelah training:

```bash
python scripts/evaluate_realistic_validation.py
```

Atau lewat FastAPI:

```bash
curl "http://localhost:8000/v1/model/evaluation-report?dataset=manual"
```

Untuk melihat prediksi tiap baris:

```bash
curl "http://localhost:8000/v1/model/evaluation-report?dataset=manual&include_predictions=true"
```

Catatan: jika hasil synthetic bagus tetapi manual test rendah, berarti model masih terlalu bergantung pada pola synthetic dataset. Tambahkan data manual/real user answer ke `answer_quality_manual_test.csv` atau dataset training utama.
