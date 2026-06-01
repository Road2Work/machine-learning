# Road2Work.id — AI Engineer Service

AI Engineer service untuk project **Road2Work.id**, yaitu AI Career Readiness Platform yang menangani:

- Professional profile extraction dari CV atau profil manual
- Role fit ranking dan role fit score
- Adaptive voice interview
- Speech-to-text 90 detik
- Answer evaluation dan Evidence Ladder
- Clarifying question
- TensorFlow answer quality model
- Career readiness result / dashboard summary
- Adaptive interview antar session berdasarkan practice memory

Service ini dibangun menggunakan **FastAPI**, **TensorFlow**, **Google Gemini / GenAI**, dan **faster-whisper**.

---

## 1. Tech Stack

| Kebutuhan | Teknologi |
|---|---|
| API Service | FastAPI |
| Server Runner | Uvicorn |
| Deep Learning Model | TensorFlow / Keras Functional API |
| Custom Training Loop | `tf.GradientTape` |
| Model Monitoring | TensorBoard |
| Generative AI | Google Gemini API |
| Speech-to-Text | faster-whisper |
| NLP Utility | NLTK, Sastrawi, regex-based extraction |
| Dataset / Asset Loader | JSON / CSV dari `data_science_resources` |
| Containerization | Docker, Docker Compose |

---

## 2. Struktur Project

```txt
road2work-ai/
├── main.py
├── model_builder.py
├── genai_helper.py
├── ds_assets.py
├── nlp_utils.py
├── stt_utils.py
├── readiness_engine.py
├── requirements.txt
├── .env.example
├── notebook.ipynb
├── Dockerfile
├── docker-compose.yml
├── scripts/
│   ├── smoke_test_contract.py
│   ├── train_answer_quality_model.py
│   ├── train_answer_quality_custom_loop.py
│   ├── evaluate_split_datasets.py
│   └── update_ds_resources.sh
├── models/
│   ├── answer_quality_model.keras
│   ├── answer_quality_tokenizer.json
│   ├── answer_quality_meta.json
│   └── logs/
└── data_science_resources/
    ├── dataset_train.csv
    ├── dataset_val.csv
    ├── dataset_test.csv
    ├── role_tree_dropdown.json
    ├── role_skill_matrix.json
    ├── skill_taxonomy.json
    ├── competency_map.json
    ├── question_seed.json
    ├── weakness_taxonomy.json
    ├── scoring_rubric.json
    └── evidence_ladder_mapping.json
```

---

## 3. Clone Project

```bash
git clone <https://github.com/Road2Work/machine-learning.git>
cd <NAMA_FOLDER_PROJECT>
```

Jika project menggunakan submodule untuk repo Data Science:

```bash
git submodule update --init --recursive
```

Jika belum ada submodule Data Science, tambahkan:

```bash
git submodule add https://github.com/Road2Work/data-science.git data_science_resources
git submodule update --init --recursive
```

---

## 4. Setup Environment Python

Disarankan memakai Python **3.10**.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Setup Environment Variable

Copy file `.env.example` menjadi `.env`.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Isi minimal `.env`:

```env
GEMINI_API_KEY=isi_api_key_kamu
GEMINI_MODEL=gemini-2.5-flash

DS_RESOURCES_DIR=./data_science_resources

MODEL_DIR=./models
ANSWER_QUALITY_MODEL_PATH=./models/answer_quality_model.keras
ANSWER_QUALITY_TOKENIZER_PATH=./models/answer_quality_tokenizer.json
ANSWER_QUALITY_META_PATH=./models/answer_quality_meta.json

MAX_MAIN_QUESTIONS=5
MAX_AUDIO_DURATION_SECONDS=90
STT_LOW_CONFIDENCE_THRESHOLD=0.60
```

Jika `GEMINI_API_KEY` kosong, beberapa fitur GenAI akan memakai fallback lokal. Namun untuk hasil terbaik, gunakan API key Gemini.

---

## 6. Data Science Resources

AI service membaca asset dari folder:

```txt
data_science_resources/
```

File yang dibutuhkan:

| File | Fungsi |
|---|---|
| `dataset_train.csv` | Dataset training answer quality model |
| `dataset_val.csv` | Dataset validation |
| `dataset_test.csv` | Dataset testing |
| `role_tree_dropdown.json` | Data dropdown Domain → Role Family → Target Role |
| `role_skill_matrix.json` | Matrix role dan skill |
| `skill_taxonomy.json` | Normalisasi skill dan tools |
| `competency_map.json` | Competency per role |
| `question_seed.json` | Seed pertanyaan interview |
| `weakness_taxonomy.json` | Weakness dan clarification mapping |
| `scoring_rubric.json` | Bobot scoring answer evaluation |
| `evidence_ladder_mapping.json` | Definisi Evidence Ladder level 1–5 |

Untuk update data dari repo Data Science:

```bash
git submodule update --remote --merge data_science_resources
```

Setelah update, jalankan ulang training jika dataset berubah.

---

## 7. Train Model TensorFlow

Model yang digunakan adalah **Answer Quality Model** dengan dua output:

| Output | Fungsi |
|---|---|
| `answer_quality` | Klasifikasi `Weak`, `Average`, `Strong` |
| `readiness_score` | Skor numerik 0.0–1.0 untuk MAE |

### Training biasa dengan `model.fit()`

```bash
python scripts/train_answer_quality_model.py
```

### Training custom loop dengan `tf.GradientTape`

```bash
python scripts/train_answer_quality_custom_loop.py
```

Custom loop ini digunakan untuk memenuhi side quest AI. Di dalamnya terdapat:

- Manual forward pass
- Manual loss calculation
- `tf.GradientTape`
- `optimizer.apply_gradients()`
- Manual accuracy tracking
- Manual MAE tracking
- TensorBoard logging
- Export model `.keras`

---

## 8. Evaluasi Model

Jalankan evaluasi pada train, validation, dan test set:

```bash
python scripts/evaluate_split_datasets.py
```

Target performa minimum:

```txt
Accuracy >= 0.85
MAE <= 0.02
```

Contoh hasil yang diharapkan:

```txt
=== TEST ===
Accuracy : 0.9783 | Target >= 0.85 | PASS
MAE      : 0.0034 | Target <= 0.02 | PASS
```

Gunakan hasil **test set** sebagai bukti utama performa model, bukan training accuracy.

---

## 9. Jalankan FastAPI Service

```bash
uvicorn main:app --reload
```

Service berjalan di:

```txt
http://localhost:8000
```

Swagger API documentation:

```txt
http://localhost:8000/docs
```

---

## 10. Smoke Test API Contract

Untuk memastikan endpoint utama berjalan:

```bash
python scripts/smoke_test_contract.py
```

Smoke test ini mengecek flow utama seperti:

- Extract manual profile
- Role fit score
- Build interview context
- Generate question
- Evaluate answer
- Generate clarification
- Predict answer quality
- Generate result / dashboard summary

---

## 11. Endpoint Utama FastAPI AI Service

Endpoint canonical v2.3:

| Method | Endpoint | Fungsi |
|---|---|---|
| POST | `/v1/profile/extract-cv` | Ekstraksi CV |
| POST | `/v1/profile/extract-manual` | Ekstraksi profil manual |
| POST | `/v1/role-fit/generate-ranking` | Generate role fit ranking untuk jalur CV |
| POST | `/v1/role-fit/calculate-score` | Hitung role fit score terhadap target role |
| POST | `/v1/interview/build-context` | Build personalized interview context |
| POST | `/v1/interview/generate-question` | Generate pertanyaan interview adaptif |
| POST | `/v1/stt/transcribe` | Speech-to-text dari audio |
| POST | `/v1/interview/evaluate-answer` | Evaluasi jawaban interview |
| POST | `/v1/interview/generate-clarification` | Generate clarifying question |
| POST | `/v1/model/predict-answer-quality` | Inference TensorFlow answer quality |
| POST | `/v1/interview/generate-result` | Generate interview result |
| POST | `/v1/dashboard/generate-summary` | Generate dashboard summary |

---

## 12. Contoh Request Inference Model

Endpoint:

```txt
POST /v1/model/predict-answer-quality
```

Payload:

```json
{
  "answer_text": "Saya membuat model klasifikasi teks menggunakan Python dan TensorFlow untuk project kampus. Saya bertanggung jawab membersihkan data, melatih model, dan mengevaluasi akurasi sampai 88%.",
  "features": {}
}
```

Contoh response:

```json
{
  "predicted_quality": "Strong",
  "confidence": 0.995,
  "supporting_score": 86
}
```

---

## 13. Speech-to-Text 90 Detik

STT mengikuti business rule:

- Mic otomatis aktif setelah HRD selesai bertanya
- Timer jawaban maksimal 90 detik
- Silence tidak menghentikan recording
- User boleh stop manual sebelum 90 detik
- Audio diproses setelah recording selesai

Endpoint:

```txt
POST /v1/stt/transcribe
```

Form-data:

| Field | Type | Required | Keterangan |
|---|---|---|---|
| `audioFile` | File | Yes | File audio user |
| `language` | String | No | Default `id` |
| `maxDurationSec` | Number | No | Default 90 |
| `silenceAutoStopEnabled` | Boolean | No | Harus `false` |
| `audioFormat` | String | No | `webm`, `wav`, atau `mp3` |

---

## 14. TensorBoard

TensorBoard digunakan untuk monitoring training.

Jalankan:

```bash
tensorboard --logdir models/logs --port 6007
```

Buka di browser:

```txt
http://localhost:6007
```

Folder log:

```txt
models/logs/
├── gradient_tape/
└── model_fit/
```

Jika TensorBoard tidak terbuka di VS Code notebook, jalankan lewat terminal dan buka manual di browser.

---

## 15. Jalankan dengan Docker

Pastikan Docker Desktop sudah berjalan.

### Build image

```bash
docker compose build
```

### Run service

```bash
docker compose up
```

Atau background mode:

```bash
docker compose up -d
```

FastAPI akan berjalan di:

```txt
http://localhost:8000
```

Swagger:

```txt
http://localhost:8000/docs
```

### Stop service

```bash
docker compose down
```

### Lihat logs

```bash
docker compose logs -f road2work-ai
```

### Jalankan training di container

```bash
docker compose run --rm road2work-ai python scripts/train_answer_quality_custom_loop.py
```

### Jalankan evaluasi di container

```bash
docker compose run --rm road2work-ai python scripts/evaluate_split_datasets.py
```

---

## 16. Docker Compose Recommended Setup

Contoh `docker-compose.yml`:

```yaml
services:
  road2work-ai:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: road2work-ai-service
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./models:/app/models
      - ./data_science_resources:/app/data_science_resources
    restart: unless-stopped

  tensorboard:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: road2work-tensorboard
    command: tensorboard --logdir /app/models/logs --host 0.0.0.0 --port 6006
    ports:
      - "6006:6006"
    volumes:
      - ./models:/app/models
    depends_on:
      - road2work-ai
```

Buka TensorBoard Docker:

```txt
http://localhost:6006
```

---

## 17. Workflow Development

Jika baru clone project:

```bash
git clone <URL_REPOSITORY_AI_ENGINEER>
cd <NAMA_FOLDER_PROJECT>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
uvicorn main:app --reload
```

Jika dataset dari DS berubah:

```bash
git submodule update --remote --merge data_science_resources
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
```

Jika hanya menjalankan API:

```bash
uvicorn main:app --reload
```

Jika menggunakan Docker:

```bash
docker compose up --build
```

---

## 18. Troubleshooting

### 1. `ModuleNotFoundError`

Solusi:

```bash
pip install -r requirements.txt
```

Pastikan virtual environment aktif.

---

### 2. TensorFlow tidak terinstall

Solusi:

```bash
pip install tensorflow
```

Atau install ulang semua dependency:

```bash
pip install -r requirements.txt
```

---

### 3. Model `.keras` tidak ditemukan

Jalankan training:

```bash
python scripts/train_answer_quality_custom_loop.py
```

Pastikan file ini muncul:

```txt
models/answer_quality_model.keras
```

---

### 4. Dataset tidak ditemukan

Pastikan folder ini ada:

```txt
data_science_resources/
```

Dan minimal berisi:

```txt
dataset_train.csv
dataset_val.csv
dataset_test.csv
```

Jika menggunakan submodule:

```bash
git submodule update --init --recursive
```

---

### 5. Gemini API error

Pastikan `.env` berisi:

```env
GEMINI_API_KEY=isi_api_key_kamu
```

Jika API key kosong, service tetap bisa berjalan dengan fallback, tetapi hasil pertanyaan dan feedback tidak sebaik GenAI asli.

---

### 6. STT error karena `ffmpeg`

Jika STT gagal memproses audio, install `ffmpeg`.

Windows:

```bash
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

Linux:

```bash
sudo apt-get install ffmpeg
```

Docker sudah menginstall `ffmpeg` melalui `Dockerfile`.

---

### 7. TensorBoard tidak terbuka

Matikan proses lama:

Windows PowerShell:

```powershell
taskkill /F /IM tensorboard.exe
```

Jalankan ulang:

```bash
tensorboard --logdir models/logs --port 6007
```

Buka:

```txt
http://localhost:6007
```

---

## 19. Bukti Checklist AI Quest

| Quest | Bukti |
|---|---|
| TensorFlow Functional API | `model_builder.py`, `build_answer_quality_model()` |
| Custom Component | `WeightedRubricScoreLayer`, `TargetPerformanceCallback` |
| Export Model | `models/answer_quality_model.keras` |
| Inference Code | `predict_answer_quality()` |
| REST API FastAPI | `main.py`, `/v1/model/predict-answer-quality` |
| GradientTape | `train_answer_quality_model_custom_loop()` |
| GenAI API | `genai_helper.py` |
| TensorBoard | `models/logs/` |
| Accuracy & MAE | `scripts/evaluate_split_datasets.py` |

---

## 20. Catatan Integrasi Fullstack

Frontend tidak memanggil FastAPI secara langsung. Alur integrasi:

```txt
Next.js Frontend
↓
Express.js Backend API Gateway
↓
FastAPI AI Service
```

Backend Express bertugas menyimpan user, profile, role, interview session, answers, result, dashboard, quota, dan admin data.

FastAPI hanya menerima payload terkontrol dari backend dan mengembalikan output AI dalam format JSON sesuai API Contract v2.3.
