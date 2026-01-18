# Sui Amor ❤️✨

An AI-powered service for generating personalized affirmations and evaluating quizzes using OpenAI. This project provides endpoints to generate affirmations, evaluate quizzes, and ingest alignment data for vector search.

---

## 🚀 Features

- ✅ **Generate Affirmations** — Create 12 personalized affirmations based on user quiz data and alignments.
- 🧠 **Quiz Evaluation** — Evaluate quizzes using OpenAI to return structured responses.
- 📂 **Alignment Upload** — Ingest alignment files to a vector DB for improved context and matching.
- ⚡ **Redis-backed Session Cache** — Optional Redis cache for storing session history.
- 🐳 **Docker + Uvicorn** — Easy to run locally or in containers.

---

## 📁 Project Structure

- `main.py` — FastAPI application entrypoint
- `app/` — Application code
  - `core/` — Configuration
  - `services/` — Business logic and API routes
  - `utils/` — Utilities (cache manager, etc.)
  - `vectordb/` — Ingestion and vector store helpers

---

## ⚙️ Prerequisites

- Python 3.10+ (recommended)
- `pip` for installing dependencies
- (Optional) Docker & Docker Compose for containerized runs
- OpenAI API key
- (Optional) Redis for caching session history

---

## 🧭 Quickstart (Local)

1. Create & activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Copy environment example and fill values

```bash
copy .env.example .env
# Edit .env and paste your OPENAI_API_KEY and other values
```

4. Run the app

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. Open the API docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 🐳 Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

The service exposes port `8000` by default.

---

## 🔌 API Endpoints

- GET `/` — Root health / welcome message
- GET `/health` — Health check
- POST `/generate_affirmations` — Generate affirmations (request body: quiz data) ✅
- POST `/quiz_evaluation` — Evaluate a quiz ✅
- POST `/alignments/upload` — Upload alignment file (multipart file upload) ✅

See `/docs` for request/response schemas.

---

## 🔐 Environment Variables

Create a `.env` file from `.env.example` and set secrets. Key variables include:

- `OPENAI_API_KEY` — Your OpenAI API key (required)
- `REDIS_URL` — Redis connection URL (optional)
- `REDIS_DB` — Redis DB index (optional, default: 0)
- `CACHE_TTL_HOURS` — Cache TTL in hours (optional)
- `PORT` — Port to run the app (defaults to 8000)

> Note: `.env` is ignored by git. Use `.env.example` to show expected keys without secrets.

---

## ✅ Testing

- Use the built-in Swagger UI to test endpoints quickly.
- For file uploads, use Postman or `curl` with `-F file=@yourfile` to POST to `/alignments/upload`.

---

## 🤝 Contributing

PRs are welcome. Please open issues for bugs or feature requests.

---

## 📄 License

Include your license here if applicable.

---

💡 Tip: For local development, copy `.env.example` to `.env` and add your `OPENAI_API_KEY` and optional Redis settings.
