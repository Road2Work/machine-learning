# Road2Work.id â€” AI Engineer Service

AI Engineer service untuk **Road2Work.id**, AI Career Readiness Platform yang menangani:

- Professional profile extraction dari CV atau profil manual
- Role fit ranking dan role fit score
- Adaptive voice interview
- Speech-to-text 90 detik
- Answer evaluation, Evidence Ladder, dan weakness detection
- Clarifying question
- TensorFlow Answer Quality Model
- Career readiness result / dashboard summary
- Adaptive interview antar session berdasarkan practice memory

Service ini dibangun menggunakan **FastAPI**, **TensorFlow**, **OpenAI Responses API / GenAI**, **faster-whisper**, **Docker**, dan **DockerHub**.

---

## 1. Tech Stack

| Kebutuhan | Teknologi |
|---|---|
| API Service | FastAPI |
| Server Runner | Uvicorn |
| Deep Learning Model | TensorFlow / Keras Functional API |
| Custom Component | Custom Layer, Custom Callback |
| Custom Training Loop | `tf.GradientTape` |
| Model Monitoring | TensorBoard |
| Generative AI | OpenAI Responses API |
| Speech-to-Text | faster-whisper |
| NLP Utility | NLTK, Sastrawi, regex-based extraction |
| Dataset / Asset Loader | JSON / CSV dari repo `data-science` |
| Containerization | Docker, Docker Compose |
| Image Registry | DockerHub |

---

## 2. Struktur Project Terbaru

Struktur repo **machine-learning**:

```txt
machine-learning/
â”œâ”€â”€ main.py
â”œâ”€â”€ model_builder.py
â”œâ”€â”€ genai_helper.py
â”œâ”€â”€ ds_assets.py
â”œâ”€â”€ nlp_utils.py
â”œâ”€â”€ stt_utils.py
â”œâ”€â”€ readiness_engine.py
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ .env.example
â”œâ”€â”€ notebook.ipynb
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ docker-compose.dockerhub.yml
â”œâ”€â”€ .dockerignore
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ smoke_test_contract.py
â”‚   â”œâ”€â”€ train_answer_quality_model.py
â”‚   â”œâ”€â”€ train_answer_quality_custom_loop.py
â”‚   â”œâ”€â”€ evaluate_split_datasets.py
â”‚   â”œâ”€â”€ update_ds_resources.sh
â”‚   â””â”€â”€ update_ds_resources.ps1
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ answer_quality_model.keras
â”‚   â”œâ”€â”€ answer_quality_tokenizer.json
â”‚   â”œâ”€â”€ answer_quality_meta.json
â”‚   â””â”€â”€ logs/
â””â”€â”€ data_science_resources/
    â””â”€â”€ data/
        â”œâ”€â”€ 01_raw/
        â”‚   â”œâ”€â”€ answer_dataset.csv
        â”‚   â”œâ”€â”€ role_tree_dropdown.json
        â”‚   â”œâ”€â”€ role_skill_matrix.json
        â”‚   â”œâ”€â”€ skill_taxonomy.json
        â”‚   â”œâ”€â”€ competency_map.json
        â”‚   â”œâ”€â”€ question_seed.json
        â”‚   â”œâ”€â”€ weakness_taxonomy.json
        â”‚   â”œâ”€â”€ scoring_rubric.json
        â”‚   â””â”€â”€ evidence_ladder_mapping.json
        â”œâ”€â”€ 02_interim/
        â”œâ”€â”€ 03_processed/
        â”‚   â”œâ”€â”€ train_df.csv
        â”‚   â”œâ”€â”€ val_df.csv
        â”‚   â””â”€â”€ test_df.csv
        â””â”€â”€ 99_archive/
```

> Catatan penting: dataset utama model **tidak lagi dibaca dari root `data_science_resources/`**, tetapi dari `data_science_resources/data/03_processed/`.

---

## 3. Clone Project

```bash
git clone https://github.com/Road2Work/machine-learning.git
cd machine-learning
```

Jika repo Data Science sudah dipasang sebagai submodule:

```bash
git submodule update --init --recursive
```

Jika belum ada submodule Data Science:

```bash
git submodule add https://github.com/Road2Work/data-science.git data_science_resources
git submodule update --init --recursive
```

Cek struktur Data Science:

```bash
ls data_science_resources/data/03_processed
```

Harus ada:

```txt
train_df.csv
val_df.csv
test_df.csv
```

---

## 4. Data Science Resources Layout

AI service membaca data dari repo `data-science` dengan struktur berikut:

| Folder | Fungsi |
|---|---|
| `data/01_raw/` | Asset mentah dan guardrail AI |
| `data/02_interim/` | Data sementara dari pipeline DS |
| `data/03_processed/` | Dataset final train/validation/test untuk model |
| `data/99_archive/` | Arsip dataset lama |

### 4.1 File dari `data/03_processed`

File utama untuk training dan evaluasi:

| File | Fungsi |
|---|---|
| `train_df.csv` | Dataset training answer quality model |
| `val_df.csv` | Dataset validation |
| `test_df.csv` | Dataset testing final |

### 4.2 File dari `data/01_raw`

File yang digunakan sebagai guardrail AI:

| File | Fungsi |
|---|---|
| `answer_dataset.csv` | Dataset mentah sebelum split |
| `role_tree_dropdown.json` | Data dropdown Domain â†’ Role Family â†’ Target Role |
| `role_skill_matrix.json` | Matrix role dan skill |
| `skill_taxonomy.json` | Normalisasi skill dan tools |
| `competency_map.json` | Competency per role |
| `question_seed.json` | Seed pertanyaan interview |
| `weakness_taxonomy.json` | Weakness dan clarification mapping |
| `scoring_rubric.json` | Bobot scoring answer evaluation |
| `evidence_ladder_mapping.json` | Definisi Evidence Ladder level 1â€“5 |

---

## 5. Setup Environment Python

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

## 6. Setup Environment Variable

Copy file `.env.example` menjadi `.env`.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### macOS / Linux

```bash
cp .env.example .env
```

Isi minimal `.env` untuk local development:

```env
OPENAI_API_KEY=isi_api_key_kamu
OPENAI_MODEL=gpt-5.4-mini

DS_RESOURCES_DIR=./data_science_resources
DS_RAW_DIR=./data_science_resources/data/01_raw
DS_PROCESSED_DIR=./data_science_resources/data/03_processed

ANSWER_QUALITY_TRAIN_PATH=./data_science_resources/data/03_processed/train_df.csv
ANSWER_QUALITY_VAL_PATH=./data_science_resources/data/03_processed/val_df.csv
ANSWER_QUALITY_TEST_PATH=./data_science_resources/data/03_processed/test_df.csv

MODEL_DIR=./models
ANSWER_QUALITY_MODEL_PATH=./models/answer_quality_model.keras
ANSWER_QUALITY_TOKENIZER_PATH=./models/answer_quality_tokenizer.json
ANSWER_QUALITY_META_PATH=./models/answer_quality_meta.json

MAX_MAIN_QUESTIONS=5
MAX_AUDIO_DURATION_SECONDS=90
STT_LOW_CONFIDENCE_THRESHOLD=0.60
DS_ASSET_AUTO_RELOAD=true
```

Jika menggunakan Docker, path di dalam `.env` dapat diarahkan ke `/app`:

```env
DS_RESOURCES_DIR=/app/data_science_resources
DS_RAW_DIR=/app/data_science_resources/data/01_raw
DS_PROCESSED_DIR=/app/data_science_resources/data/03_processed

ANSWER_QUALITY_TRAIN_PATH=/app/data_science_resources/data/03_processed/train_df.csv
ANSWER_QUALITY_VAL_PATH=/app/data_science_resources/data/03_processed/val_df.csv
ANSWER_QUALITY_TEST_PATH=/app/data_science_resources/data/03_processed/test_df.csv

MODEL_DIR=/app/models
ANSWER_QUALITY_MODEL_PATH=/app/models/answer_quality_model.keras
ANSWER_QUALITY_TOKENIZER_PATH=/app/models/answer_quality_tokenizer.json
ANSWER_QUALITY_META_PATH=/app/models/answer_quality_meta.json
```

> Jangan push `.env` ke GitHub. Push hanya `.env.example`.

---

## 7. Update Data dari Repo Data Science

Ketika tim Data Science mengubah data di repo `data-science`, update submodule:

```bash
git submodule update --remote --merge data_science_resources
```

Atau pakai script:

### Windows PowerShell

```powershell
.\scripts\update_ds_resources.ps1
```

### macOS / Linux

```bash
./scripts/update_ds_resources.sh
```

Setelah data berubah, jalankan ulang training dan evaluasi:

```bash
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
```

Jika FastAPI sedang berjalan, reload asset:

```bash
curl -X POST http://localhost:8000/v1/admin/reload-ds-assets
```

Untuk Docker production atau multi-worker, lebih aman restart container:

```bash
docker compose restart road2work-ai
```

---

## 8. Train Model TensorFlow

Model yang digunakan adalah **Answer Quality Model** dengan dua output:

| Output | Fungsi |
|---|---|
| `answer_quality` | Klasifikasi `Weak`, `Average`, `Strong` |
| `readiness_score` | Skor numerik 0.0â€“1.0 untuk MAE |

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

## 9. Evaluasi Model

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

Gunakan hasil **test set** sebagai bukti utama performa model.

---

## 10. Jalankan FastAPI Service secara Lokal

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

Cek status Data Science asset:

```bash
curl http://localhost:8000/v1/admin/ds-assets/status
```

---

## 11. Smoke Test API Contract

Untuk memastikan endpoint utama berjalan:

```bash
python scripts/smoke_test_contract.py
```

Smoke test ini mengecek flow:

- Extract manual profile
- Role fit score
- Build interview context
- Generate question
- Evaluate answer
- Generate clarification
- Predict answer quality
- Generate result / dashboard summary

---

## 12. Endpoint Utama FastAPI AI Service

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

## 13. Contoh Request Inference Model

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

## 14. Speech-to-Text 90 Detik

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

## 15. TensorBoard

TensorBoard digunakan untuk monitoring training.

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
â”œâ”€â”€ gradient_tape/
â””â”€â”€ model_fit/
```

Jika TensorBoard tidak terbuka di VS Code notebook, jalankan lewat terminal dan buka manual di browser.

---

## 16. DockerHub Workflow

Untuk kolaborasi, image utama disimpan di DockerHub:

```txt
arteris/road2work-ai:v2.3
arteris/road2work-ai:latest
```

### 16.1 Pull image dari DockerHub

```bash
docker pull arteris/road2work-ai:v2.3
```

### 16.2 Jalankan image langsung

#### Windows PowerShell

```powershell
docker run --env-file .env `
  -p 8000:8000 `
  -v ${PWD}\models:/app/models `
  -v ${PWD}\data_science_resources:/app/data_science_resources `
  arteris/road2work-ai:v2.3
```

#### macOS / Linux

```bash
docker run --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data_science_resources:/app/data_science_resources \
  arteris/road2work-ai:v2.3
```

Buka:

```txt
http://localhost:8000/docs
```

---

## 17. Docker Compose dengan DockerHub Image

Gunakan file `docker-compose.dockerhub.yml` untuk menjalankan image dari DockerHub tanpa build ulang.

Contoh isi:

```yaml
services:
  road2work-ai:
    image: arteris/road2work-ai:v2.3
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
    image: arteris/road2work-ai:v2.3
    container_name: road2work-tensorboard
    command: tensorboard --logdir /app/models/logs --host 0.0.0.0 --port 6006
    ports:
      - "6006:6006"
    volumes:
      - ./models:/app/models
```

Jalankan:

```bash
docker compose -f docker-compose.dockerhub.yml up -d
```

Cek logs:

```bash
docker compose -f docker-compose.dockerhub.yml logs -f road2work-ai
```

Stop service:

```bash
docker compose -f docker-compose.dockerhub.yml down
```

---

## 18. Docker Build Lokal

Gunakan ini jika kamu sedang mengubah source code dan ingin build image sendiri.

```bash
docker compose up --build
```

Atau build manual:

```bash
docker build -t road2work-ai:local .
```

Run local image:

```bash
docker run --env-file .env -p 8000:8000 road2work-ai:local
```

---

## 19. Push Image ke DockerHub

Login DockerHub:

```bash
docker login
```

Build image:

```bash
docker build -t road2work-ai:latest .
```

Tag image:

```bash
docker tag road2work-ai:latest arteris/road2work-ai:v2.3
docker tag road2work-ai:latest arteris/road2work-ai:latest
```

Push image:

```bash
docker push arteris/road2work-ai:v2.3
docker push arteris/road2work-ai:latest
```

Rekomendasi tag:

| Tag | Fungsi |
|---|---|
| `v2.3` | Versi sesuai API Contract v2.3 |
| `latest` | Versi terbaru |
| `stable` | Versi aman untuk demo |
| `dev` | Versi eksperimen |

---

## 20. Training dan Evaluasi di Docker

Training custom loop:

```bash
docker compose run --rm road2work-ai python scripts/train_answer_quality_custom_loop.py
```

Evaluasi:

```bash
docker compose run --rm road2work-ai python scripts/evaluate_split_datasets.py
```

Jika memakai DockerHub compose:

```bash
docker compose -f docker-compose.dockerhub.yml run --rm road2work-ai python scripts/train_answer_quality_custom_loop.py
docker compose -f docker-compose.dockerhub.yml run --rm road2work-ai python scripts/evaluate_split_datasets.py
```

Karena folder `models` di-mount sebagai volume, hasil training tetap tersimpan ke folder lokal:

```txt
models/answer_quality_model.keras
models/answer_quality_tokenizer.json
models/answer_quality_meta.json
```

---

## 21. Workflow Harian

### Jika baru clone project

```bash
git clone https://github.com/Road2Work/machine-learning.git
cd machine-learning
git submodule update --init --recursive
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
uvicorn main:app --reload
```

Untuk Windows, ganti aktivasi venv:

```powershell
.\.venv\Scripts\activate
```

### Jika hanya ingin menjalankan API via DockerHub

```bash
git clone https://github.com/Road2Work/machine-learning.git
cd machine-learning
git submodule update --init --recursive
cp .env.example .env
docker pull arteris/road2work-ai:v2.3
docker compose -f docker-compose.dockerhub.yml up -d
```

### Jika dataset dari DS berubah

```bash
git submodule update --remote --merge data_science_resources
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
docker compose restart road2work-ai
```

### Jika source code AI berubah

```bash
docker build -t road2work-ai:latest .
docker tag road2work-ai:latest arteris/road2work-ai:v2.3
docker push arteris/road2work-ai:v2.3
```

Tim lain cukup menjalankan:

```bash
docker compose -f docker-compose.dockerhub.yml pull
docker compose -f docker-compose.dockerhub.yml up -d
```

---

## 22. Troubleshooting

### 22.1 `ModuleNotFoundError`

```bash
pip install -r requirements.txt
```

Pastikan virtual environment aktif.

---

### 22.2 Model `.keras` tidak ditemukan

Jalankan training:

```bash
python scripts/train_answer_quality_custom_loop.py
```

Pastikan file ini muncul:

```txt
models/answer_quality_model.keras
```

---

### 22.3 Dataset tidak ditemukan

Pastikan submodule sudah di-init:

```bash
git submodule update --init --recursive
```

Pastikan file ini ada:

```txt
data_science_resources/data/03_processed/train_df.csv
data_science_resources/data/03_processed/val_df.csv
data_science_resources/data/03_processed/test_df.csv
```

---

### 22.4 Data guardrail tidak ditemukan

Pastikan file JSON ada di:

```txt
data_science_resources/data/01_raw/
```

Cek status asset:

```bash
curl http://localhost:8000/v1/admin/ds-assets/status
```

---

### 22.5 OpenAI API error

Pastikan `.env` berisi:

```env
OPENAI_API_KEY=isi_api_key_kamu
```

Jika API key kosong, service tetap bisa berjalan dengan fallback, tetapi hasil question/feedback tidak sebaik GenAI asli.

---

### 22.6 STT error karena `ffmpeg`

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

### 22.7 TensorBoard tidak terbuka

Matikan proses lama:

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

### 22.8 Docker image lama masih kebaca

Cek image:

```bash
docker images
```

Hapus image lokal lama:

```bash
docker rmi road2work-ai:local
docker rmi capstoneproject-road2work-ai:latest
```

Pull ulang image DockerHub:

```bash
docker pull arteris/road2work-ai:v2.3
```

---

## 23. Bukti Checklist AI Quest

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

## 24. Catatan Integrasi Fullstack

Frontend tidak memanggil FastAPI secara langsung. Alur integrasi:

```txt
Next.js Frontend
â†“
Express.js Backend API Gateway
â†“
FastAPI AI Service
```

Backend Express bertugas menyimpan user, profile, role, interview session, answers, result, dashboard, quota, dan admin data.

FastAPI hanya menerima payload terkontrol dari backend dan mengembalikan output AI dalam format JSON sesuai API Contract v2.3.

---

## 25. Ringkasan Command Penting

### Local run

```bash
uvicorn main:app --reload
```

### Train

```bash
python scripts/train_answer_quality_custom_loop.py
```

### Evaluate

```bash
python scripts/evaluate_split_datasets.py
```

### DockerHub run

```bash
docker compose -f docker-compose.dockerhub.yml up -d
```

### DockerHub pull update

```bash
docker compose -f docker-compose.dockerhub.yml pull
docker compose -f docker-compose.dockerhub.yml up -d
```

### Update DS resources

```bash
git submodule update --remote --merge data_science_resources
```

### Check API docs

```txt
http://localhost:8000/docs
```

