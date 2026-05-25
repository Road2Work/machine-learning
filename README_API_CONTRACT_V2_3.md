# Road2Work AI Service — v2.3 Adaptive Session Alignment

Package ini sudah direvisi mengikuti **Road2Work Overview v2.3 Adaptive Session** dan **API Contract v2.3**.

## Update utama v2.3

1. **Adaptive interview antar session**
   - Backend dapat mengirim `adaptivePracticeMemory` ke FastAPI.
   - AI membaca `previousInterviewSummary`, `previousDetectedWeaknesses`, `askedQuestionHistory`, `latestInterviewFeedback`, `nextBestActions`, dan `improvementFocus`.
   - Pertanyaan baru menghindari pertanyaan yang sama persis jika `avoidRepeatedQuestions=true`.
   - Pertanyaan lama hanya boleh diulang jika `retryMode=true`.

2. **Endpoint canonical v2.3**
   - `POST /v1/profile/extract-cv`
   - `POST /v1/profile/extract-manual`
   - `POST /v1/role-fit/generate-ranking`
   - `POST /v1/role-fit/calculate-score`
   - `POST /v1/interview/build-context`
   - `POST /v1/interview/generate-question`
   - `POST /v1/stt/transcribe`
   - `POST /v1/interview/evaluate-answer`
   - `POST /v1/interview/generate-clarification`
   - `POST /v1/interview/generate-result`
   - `POST /v1/model/predict-answer-quality`
   - `POST /v1/dashboard/generate-summary`

3. **Backward compatibility**
   Endpoint v2.1 lama tetap hidup sebagai alias:
   - `/v1/context/extract-cv`
   - `/v1/context/extract-profile`
   - `/v1/role-fit/score`
   - `/v1/interview/next-question`
   - `/v1/interview/clarifying-question`

4. **Recording policy v2.3**
   Response question dan STT membawa `RecordingPolicy`:
   - `autoStartMic=true`
   - `autoStartTrigger=after_hrd_question_finished`
   - `answerLimitSeconds=90`
   - `silenceAutoStopEnabled=false`
   - `userCanStopBeforeLimit=true`
   - `stopReasons=["user_mic_off", "timer_timeout"]`
   - `audioFormat=webm|wav|mp3`

5. **HRD state v2.3**
   Endpoint question/evaluation/clarification memakai state utama:
   - `idle`
   - `asking`
   - `listening`
   - `thinking`
   - `clarifying`
   - `completed`
   - `error`

6. **Dashboard career summary**
   - `POST /v1/dashboard/generate-summary`
   - Jika `careerReadinessScore < 90`, service mengembalikan `403 CAREER_SUMMARY_LOCKED`.

## Dataset dan Model

Dataset TensorFlow tetap membaca split dari Data Science:

- `data_science_resources/dataset_train.csv`
- `data_science_resources/dataset_val.csv`
- `data_science_resources/dataset_test.csv`

Model tetap memenuhi arah quest:

- TensorFlow Functional API
- Custom Callback
- Export `.keras`
- Inference code
- FastAPI endpoint
- Custom GradientTape loop
- TensorBoard
- Evaluation accuracy & MAE

## Cara run

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn main:app --reload
```

## Smoke test v2.3

```bash
python scripts/smoke_test_contract.py
```

## Train model

```bash
python scripts/train_answer_quality_custom_loop.py
python scripts/evaluate_split_datasets.py
```

## TensorBoard

```bash
tensorboard --logdir models/logs
```

atau:

```bash
./scripts/start_tensorboard.sh
```

## Update Data Science resources

```bash
git submodule update --remote --merge data_science_resources
curl -X POST http://localhost:8000/v1/admin/reload-ds-assets
```

Untuk production multi-worker/Docker, restart service setelah update data tetap disarankan.
