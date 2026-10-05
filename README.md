# HR Chat Agent (Ollama + Qdrant RAG)

An HR assistant for authenticated employees. It answers policy questions from HR documents (RAG over Qdrant),
and answers personal / dynamic questions (leave balance, eligibility, working-day calculations, year-end lapse,
holidays, applying for leave) by calling tools backed by an HR database. All data is synthetic.

```
Browser (static/index.html)
   │  login → bearer token
   ▼
FastAPI (app/main.py) ── /api/chat ──► Agent loop (app/agent.py)
   │                                     │  1. privacy guard (other employees mentioned? → refuse)
   │                                     │  2. system prompt + today + calendar + trimmed chat history
   │                                     │  3. Ollama chat model decides: answer, or call tool(s)
   │                                     │  4. tool results fed back → repeat (max 6 steps)
   │                                     │  5. grounding nudge if it answered without any tool
   │                                     ▼
   │                         Tools (app/tools.py) — always scoped to the logged-in employee
   │                           ├─ search_hr_policy ─────────► Qdrant (all-minilm vectors)
   │                           ├─ get_leave_balance / get_leave_history / get_my_profile
   │                           ├─ check_leave_eligibility / calculate_working_days
   │                           ├─ project_year_end_balance / get_holidays ─► SQLite + leave_rules.py
   │                           └─ apply_leave (only after the user confirms)
   │
   └─ /api/documents (HR admin) ─► extract text → semantic chunking → embed → Qdrant
```

## Model choices

| Purpose | Default | Why |
|---|---|---|
| Chat + tool calling | `qwen2.5:7b` (set `CHAT_MODEL`) | Reliable tool calling. `qwen2.5:3b` works on low-RAM / CPU-only machines (tested: correct tools for balance, eligibility, RAG, lapse projection and apply; ~25-80 s per answer on a 12-core CPU). `qwen2.5:1.5b` was tested and is **not** reliable — it skips tools and repeats earlier answers. |
| Embeddings | `all-minilm` (22M params, 384-dim, ~46 MB) | Smallest Ollama embedding model; fast on CPU. HR policy chunks are short, so retrieval quality is good. |
| Semantic chunking | `all-minilm` (`CHUNK_EMBED_MODEL`) | Only needs to tell whether adjacent sentences are about the same topic. |

Upgrade path if retrieval quality matters more than footprint: `snowflake-arctic-embed:33m` (512-token context) or
`nomic-embed-text` (8k context, 768-dim). After changing `EMBED_MODEL`, run `python -m app.ingest --rebuild`.

### How semantic chunking works (`app/chunking.py`)
1. Markdown headings (also DOCX heading styles) are hard boundaries; each chunk records its heading path.
2. The section is split into sentences / list items.
3. Each sentence is embedded together with its neighbours (window = 1) using `all-minilm`.
4. The cosine distance between consecutive sentences is computed; a new chunk starts where the distance is above
   the 90th percentile for the document (a topic shift), while keeping chunks between 200 and 1000 characters
   (1000 chars ≈ all-minilm's 256-token limit, so nothing is silently truncated).

Tune with `CHUNK_BREAKPOINT_PERCENTILE`, `CHUNK_MIN_CHARS`, `CHUNK_MAX_CHARS`. Use the **Preview chunks** button (or
`POST /api/documents/preview`) to inspect the result without storing it.

## Setup

### 1. Ollama models
```bash
ollama pull qwen2.5:7b        # or qwen2.5:3b on machines with < 8 GB free RAM
ollama pull all-minilm
```

### 2. Qdrant
See [Qdrant setup](#qdrant-setup) below. Quick version: `docker compose up -d`.

### 3. Python app
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # adjust models / URLs if needed

python -m app.db --reset        # create SQLite HR DB with synthetic employees
python -m app.ingest            # chunk + embed data/policies/* into Qdrant
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 and sign in with a demo user (password `demo123`):

| User | Scenario |
|---|---|
| `asha` | Senior engineer, 5 years, has EL carry-forward, approved + pending leaves |
| `rahul` | Joined ~50 days ago — on probation, prorated CL/SL, EL not usable yet |
| `arjun` | Intern — CL/SL only |
| `meera` | HR Manager, `hr_admin` role — can upload policy documents |

Check service status at http://localhost:8000/api/health.

## Qdrant setup

**Option A — Docker Compose (recommended)**
```bash
docker compose up -d                       # uses docker-compose.yml in this folder
docker compose ps                          # hr-qdrant should be "running"
curl http://localhost:6333/healthz         # -> "healthz check passed"
```
Data persists in `./qdrant_storage`. Dashboard: http://localhost:6333/dashboard

**Option B — plain Docker**
```bash
docker pull qdrant/qdrant
docker run -d --name hr-qdrant -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

**Option C — no Docker (binary)**
Download the release for your OS from https://github.com/qdrant/qdrant/releases, extract it and run `./qdrant`.
It listens on port 6333 by default.

**Option D — Qdrant Cloud**
Create a free cluster at https://cloud.qdrant.io, then set in `.env`:
```
QDRANT_URL=https://<your-cluster>.cloud.qdrant.io:6333
QDRANT_API_KEY=<key>
```

**Option E — quick test without a server**: `QDRANT_URL=:memory:` (embedded, data lost on restart).

**Verify the index**
```bash
python -m app.ingest
curl http://localhost:6333/collections/hr_policies    # points_count > 0, vectors size 384
```
In the dashboard, open the `hr_policies` collection to browse chunks and their payloads
(`document`, `section`, `text`, `uploaded_by`, `indexed_at`).

## Document API (HR admin only)

```bash
TOKEN=$(curl -s localhost:8000/api/login -H 'Content-Type: application/json' \
  -d '{"username":"meera","password":"demo123"}' | python -c 'import sys,json;print(json.load(sys.stdin)["token"])')

# Upload + index (.md .txt .pdf .docx, max 10 MB). Re-uploading the same file name replaces its chunks.
curl -H "Authorization: Bearer $TOKEN" -F "file=@travel_policy.pdf" localhost:8000/api/documents

# Preview semantic chunks without storing
curl -H "Authorization: Bearer $TOKEN" -F "file=@travel_policy.pdf" "localhost:8000/api/documents/preview?limit=10"

# List indexed documents (any logged-in user)
curl -H "Authorization: Bearer $TOKEN" localhost:8000/api/documents
```
Uploaded files are also saved to `data/policies/` so `python -m app.ingest --rebuild` can recreate the index.

## Try these questions
- "What is my leave balance?"
- "Can I take earned leave from next Monday to Friday?" → eligibility tool (balance, notice, holidays)
- "How many working days between 2026-10-19 and 2026-10-23?" → excludes the 20 Oct holiday
- "How many leaves will lapse at year end?" → year-end projection with the 10-day carry-forward cap
- "I'm on probation, can I take EL?" (as `rahul`)
- "What's the medical certificate rule for sick leave?" → RAG
- "What is Rahul's leave balance?" (as `asha`) → blocked by the privacy guard
- "Apply EL for 2026-10-26 to 2026-10-27 for a family function" → asks to confirm, then submits

## Project layout
```
app/
  config.py       settings (env / .env)
  db.py           SQLite schema, synthetic seed data, queries, sessions
  leave_rules.py  leave policy as code (accrual, proration, working days, eligibility)
  tools.py        agent tools + JSON schemas for Ollama
  agent.py        system prompt, agent loop, guardrails, conversation memory
  chunking.py     semantic chunking
  ingest.py       text extraction (md/txt/pdf/docx), indexing into Qdrant
  rag.py          embeddings + Qdrant search
  main.py         FastAPI app (auth, chat, document APIs, UI)
static/index.html chat UI
data/policies/    sample HR policy documents (fictional)
```

## Notes / limitations
- Demo auth (username/password + server-side session tokens). Use your SSO (OIDC/SAML) in production.
- Chat history lives in memory per session; it is lost when the server restarts.
- The privacy guard matches other employees' names/IDs from the DB. It is deliberately strict, so mentioning a
  colleague's name in an unrelated question is also refused.
- The holiday table is seeded for 2026 only.
