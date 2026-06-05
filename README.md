# 🤖 Road2Work.id Machine Learning Service

Machine Learning Service Road2Work.id adalah service berbasis **FastAPI** yang menangani pemrosesan CV, speech-to-text, pembuatan pertanyaan interview adaptif, evaluasi jawaban interview, dan readiness insight.

Service ini dipanggil oleh **backend**, bukan langsung oleh frontend.

---

## 🧰 Tech Stack

* **Python**
* **FastAPI**
* **Uvicorn**
* **Pydantic**
* **OpenAI Responses API**
* **faster-whisper**
* **scikit-learn**
* **TensorFlow / Keras**
* **pandas**
* **numpy**
* **python-dotenv**
* **PyMuPDF / PDF Text Extraction**

---

## 📁 Struktur Folder

```txt
machine-learning/
├── main.py                         # FastAPI entry point
├── genai_helper.py                 # Helper GenAI untuk pertanyaan, evaluasi, feedback
├── readiness_engine.py             # Readiness scoring dan evaluasi jawaban
├── nlp_utils.py                    # NLP utility untuk ekstraksi profil dan skill
├── stt_utils.py                    # Speech-to-text utility
├── model_builder.py                # Training/building model utility
├── data/                           # Data kecil yang dibutuhkan service
├── models/                         # Model artifact lokal, sebagian di-ignore jika besar
├── scripts/                        # Script training/evaluation
├── requirements.txt                # Dependency Python
├── .env.example                    # Template environment
└── README_AI_ENGINEER.md
```

---

## ✨ Fungsi Utama

Service ini menangani beberapa proses utama pada Road2Work.id:

* CV extraction
* Short profile extraction
* Role fit context generation
* Adaptive interview question generation
* Speech-to-text transcription
* Answer quality evaluation
* Clarification decision
* Evidence ladder evaluation
* Interview result generation
* Readiness scoring support

---

## 🧭 Arsitektur Integrasi

Frontend tidak memanggil Machine Learning service secara langsung.

```txt
Frontend
→ Backend API
→ Machine Learning Service
→ Backend API
→ Frontend
```

### 📄 Contoh Flow Upload CV

```txt
Upload CV
→ Backend menerima file
→ Backend mengirim file ke ML service
→ ML service mengekstrak profil
→ Backend menyimpan hasil ke database
→ Frontend menampilkan profile review
```

### 🎙️ Contoh Flow Interview

```txt
User menjawab dengan suara
→ Frontend mengirim audio ke Backend
→ Backend mengirim audio ke ML service
→ ML service melakukan STT dan evaluasi jawaban
→ Backend menyimpan result
→ Frontend menampilkan next question atau result
```

---

## ✅ Prasyarat

Pastikan sudah terinstall:

* Python versi 3.10 atau lebih baru
* pip
* virtualenv
* FFmpeg untuk audio/STT
* Backend Road2Work.id sudah berjalan jika ingin mencoba flow penuh

Cek FFmpeg:

```bash
ffmpeg -version
```

Jika belum tersedia, install FFmpeg terlebih dahulu sesuai sistem operasi yang digunakan.

---

## ⚙️ Setup Local

Masuk ke folder `machine-learning`:

```bash
cd machine-learning
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment.

### Windows PowerShell

```bash
.venv\Scripts\Activate.ps1
```

### Windows CMD

```bash
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Buat file `.env` dari `.env.example`:

```bash
cp .env.example .env
```

Contoh konfigurasi `.env` local:

```env
OPENAI_API_KEY=isi_openai_api_key
OPENAI_MODEL=gpt-5.4-mini
OPENAI_MAX_OUTPUT_TOKENS=2048
GENAI_MAX_RETRIES=3

ROAD2WORK_DATA_DIR=data

STT_MODEL_SIZE=small
STT_LANGUAGE=id
```

Jalankan service:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Buka dokumentasi API:

```txt
http://127.0.0.1:8000/docs
```

---

## 🔐 Environment Variables

| Variable                   | Keterangan                                       |
| -------------------------- | ------------------------------------------------ |
| `OPENAI_API_KEY`           | API key untuk OpenAI Responses API               |
| `OPENAI_MODEL`             | Model yang dipakai untuk pertanyaan dan evaluasi |
| `OPENAI_MAX_OUTPUT_TOKENS` | Batas maksimal output token                      |
| `GENAI_MAX_RETRIES`        | Jumlah retry jika request AI gagal sementara     |
| `ROAD2WORK_DATA_DIR`       | Path folder data Machine Learning                |
| `STT_MODEL_SIZE`           | Ukuran model faster-whisper                      |
| `STT_LANGUAGE`             | Bahasa transkripsi, default `id`                 |

Contoh local:

```env
ROAD2WORK_DATA_DIR=data
```

Contoh jika data diarahkan ke folder tertentu:

```env
ROAD2WORK_DATA_DIR=E:\Kuliah\DBS Dicoding\2026\Capstone Project\Code\Final Apps\machine-learning\data
```

---

## 🌐 Endpoint Utama

### 🩺 Health Check

```http
GET /
```

Digunakan untuk memastikan service berjalan.

---

### 📄 Extract CV

```http
POST /v1/context/extract-cv
```

Input:

* Multipart file CV

Output:

* Professional summary
* Skills
* Tools
* Experience signals
* Achievements
* Evidence items

---

### 👤 Extract Short Profile

```http
POST /v1/context/extract-short-profile
```

Input:

* Profile summary
* Target role
* Skill/tools manual

Output:

* Structured profile context

---

### 🎧 Transcribe Audio

```http
POST /v1/stt/transcribe
```

Input:

* Audio file

Output:

* Transcript
* Confidence / metadata jika tersedia

---

### ❓ Generate Question

```http
POST /v1/interview/generate-question
```

Input:

* Target role
* Profile context
* Interview history
* Current state
* Improvement focus

Output:

* Adaptive question
* Question type
* Competency focus

---

### 🧪 Evaluate Answer

```http
POST /v1/interview/evaluate-answer
```

Input:

* Question
* Answer transcript
* Target role
* Profile context
* Previous answers

Output:

* Score breakdown
* Final score
* Evidence level
* Weakness
* Clarification decision
* Stronger answer
* Next recommendation

---

## 🎙️ Interview State

Service mendukung flow interview berikut:

```txt
IDLE
→ ASKING
→ LISTENING
→ THINKING
→ CLARIFYING / ASKING
→ COMPLETED
```

Fungsi ML/AI terutama aktif pada state:

```txt
THINKING
```

Pada state ini sistem akan:

* Memproses transcript
* Mengevaluasi jawaban
* Menentukan apakah perlu klarifikasi
* Menyiapkan pertanyaan berikutnya
* Menyusun feedback dan result

---

## 🪜 Evidence Ladder

Road2Work.id menggunakan konsep **Evidence Ladder** untuk menilai kekuatan jawaban user saat interview.

| Level | Nama              | Contoh                                                           |
| ----- | ----------------- | ---------------------------------------------------------------- |
| 1     | Claim             | “Saya bisa membuat dashboard.”                                   |
| 2     | Skill             | “Saya menggunakan Excel dan Python.”                             |
| 3     | Context           | “Saya membuat dashboard untuk data penjualan.”                   |
| 4     | Impact            | “Dashboard membantu tim memahami performa produk.”               |
| 5     | Measurable Result | “Waktu pembuatan laporan berkurang dari 2 jam menjadi 30 menit.” |

Semakin tinggi evidence level, semakin kuat jawaban interview user.

---

## 📊 Scoring

Evaluasi jawaban menggunakan beberapa dimensi:

* Role relevance
* STAR structure
* Evidence specificity
* Technical accuracy
* Communication clarity
* Self awareness

Bobot penilaian utama:

| Dimensi               | Bobot |
| --------------------- | ----: |
| Role Relevance        |   25% |
| STAR Structure        |   20% |
| Evidence Specificity  |   20% |
| Technical Accuracy    |   15% |
| Communication Clarity |   10% |
| Self Awareness        |   10% |

Final score akan dinormalisasi sebelum dikirim kembali ke backend.

---

## 🎧 STT Notes

Speech-to-text menggunakan **faster-whisper**.

Rekomendasi local:

```env
STT_MODEL_SIZE=small
STT_LANGUAGE=id
```

Jika transkripsi kurang akurat:

* Gunakan microphone yang lebih jelas
* Kurangi noise ruangan
* Gunakan model STT lebih besar jika resource cukup
* Pastikan audio dikirim dengan format yang didukung, seperti `webm`, `wav`, atau `mp3`

---

## 🧪 Script Testing Manual

Jalankan service:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Buka Swagger:

```txt
http://127.0.0.1:8000/docs
```

Tes endpoint berikut:

* Upload CV ke `/v1/context/extract-cv`
* Tes transkripsi audio ke `/v1/stt/transcribe`
* Tes generate question ke `/v1/interview/generate-question`
* Tes evaluate answer ke `/v1/interview/evaluate-answer`

---

## 🔗 Integrasi Dengan Backend

Backend harus mengarah ke service ini melalui environment variable:

```env
ML_SERVICE_URL=http://localhost:8000
```

Production:

```env
ML_SERVICE_URL=https://your-ml-service-url
```

Backend akan meneruskan request dari frontend ke Machine Learning service.

---

## 🚢 Deployment Notes

Sebelum deploy, pastikan:

* `.env` production sudah tersedia di server
* `OPENAI_API_KEY` valid
* `ROAD2WORK_DATA_DIR` mengarah ke folder data yang benar
* FFmpeg tersedia di environment deployment
* Dependency Python berhasil di-install
* Service dapat diakses oleh backend

Contoh command production:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Jika menggunakan process manager atau Docker, pastikan port `8000` terbuka untuk backend service.

---

## 🚫 Git Ignore Notes

Jangan commit file berikut:

```txt
.env
.venv/
__pycache__/
uploads/
tmp/
temp/
*.wav
*.mp3
*.m4a
*.webm
models/*.keras
models/*.h5
models/*.pb
models/*.savedmodel/
```

Data kecil yang dibutuhkan service boleh di-commit:

```txt
data/
```

Model besar sebaiknya disimpan di external storage atau Git LFS.

---

## 🛠️ Troubleshooting

### `OPENAI_API_KEY` tidak terbaca

Cek file `.env`:

```env
OPENAI_API_KEY=...
```

Restart service setelah mengubah `.env`.

---

### `422 Unprocessable Entity`

Biasanya payload request tidak sesuai schema endpoint.

Cek Swagger di:

```txt
http://127.0.0.1:8000/docs
```

Pastikan field wajib sudah dikirim.

---

### STT gagal membaca audio

Pastikan:

* FFmpeg sudah terinstall
* Format audio didukung
* File audio tidak kosong
* Microphone browser benar-benar aktif

---

### Port 8000 sudah dipakai

Cek proses yang memakai port.

Windows PowerShell:

```bash
netstat -ano | findstr :8000
```

Matikan proses:

```bash
taskkill /PID <PID> /F
```

Atau jalankan di port lain:

```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

---

### Hasil pertanyaan kurang adaptif

Pastikan request ke `/v1/interview/generate-question` mengirim context yang lengkap:

* Target role
* Profile context
* Previous answers
* Improvement focus
* Previous questions
* Current interview state

Tanpa context tersebut, pertanyaan akan cenderung generic.

---

## 📌 Status MVP

MVP Road2Work.id saat ini berfokus pada domain **Information Technology**.

Ekspansi domain lain dapat dilakukan dengan menambah taxonomy role, competency map, dan dataset pendukung.

---

## 👥 Author

**Road2Work.id**
Capstone Project CC26-PSU050
