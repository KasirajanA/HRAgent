"""FastAPI server: authentication, chat endpoint and the static UI.

Run: uvicorn app.main:app --reload
"""
import json
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, db
from .agent import conversations, run_agent, run_agent_stream, warm_up
from .ingest import IngestError, ingest_document, list_documents, preview_chunks
from .rag import ollama_client, qdrant_client

STATIC_DIR = config.BASE_DIR / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    if config.WARM_UP:  # in the background so the server is reachable immediately
        threading.Thread(target=warm_up, daemon=True).start()
    yield


app = FastAPI(title="HR Chat Agent", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


def current_session(authorization: str = Header(default="")) -> tuple[str, dict]:
    token = authorization.removeprefix("Bearer ").strip()
    employee = db.get_session_employee(token) if token else None
    if not employee:
        raise HTTPException(status_code=401, detail="Not authenticated or session expired")
    return token, employee


def require_hr_admin(session=Depends(current_session)) -> tuple[str, dict]:
    if session[1]["role"] != "hr_admin":
        raise HTTPException(status_code=403, detail="Only HR administrators can manage policy documents")
    return session


def public_profile(employee: dict) -> dict:
    return {k: employee[k] for k in ("id", "full_name", "department", "designation", "employment_type", "role")}


async def read_upload(file: UploadFile) -> tuple[str, bytes]:
    # Keep only the base name and safe characters so the file can't escape POLICY_DIR.
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file.filename or "").name).strip("._")
    if Path(filename).suffix.lower() not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Allowed file types: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")
    data = await file.read(config.UPLOAD_MAX_MB * 1024 * 1024 + 1)
    if len(data) > config.UPLOAD_MAX_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {config.UPLOAD_MAX_MB} MB")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return filename, data


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/login")
def login(body: LoginRequest):
    employee = db.get_employee_by_username(body.username.strip().lower())
    if not employee or not db.verify_password(body.password, employee["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": db.create_session(employee["id"]), "employee": public_profile(employee)}


@app.post("/api/logout")
def logout(session=Depends(current_session)):
    token, _ = session
    conversations.clear(token)
    db.delete_session(token)
    return {"ok": True}


@app.get("/api/me")
def me(session=Depends(current_session)):
    return public_profile(session[1])


@app.post("/api/chat")
def chat(body: ChatRequest, session=Depends(current_session)):
    token, employee = session
    return run_agent(token, employee, body.message.strip())


@app.post("/api/chat/stream")
def chat_stream(body: ChatRequest, session=Depends(current_session)):
    """Same as /api/chat but streams newline-delimited JSON events (status, step, token, done/error)."""
    token, employee = session
    events = (json.dumps(e, default=str) + "\n" for e in run_agent_stream(token, employee, body.message.strip()))
    return StreamingResponse(events, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/chat/reset")
def reset_chat(session=Depends(current_session)):
    conversations.clear(session[0])
    return {"ok": True}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...), session=Depends(require_hr_admin)):
    """Upload a policy document: semantic-chunk it, embed it and push it to Qdrant.
    Re-uploading a file with the same name replaces its previous chunks."""
    filename, data = await read_upload(file)
    try:
        stats = await run_in_threadpool(ingest_document, filename, data, session[1]["id"])
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # Ollama / Qdrant unreachable, model not pulled, ...
        raise HTTPException(status_code=503, detail=f"Indexing failed: {type(exc).__name__}: {exc}")
    config.POLICY_DIR.mkdir(parents=True, exist_ok=True)
    (config.POLICY_DIR / filename).write_bytes(data)  # keep the source so `python -m app.ingest --rebuild` works
    return stats


@app.post("/api/documents/preview")
async def preview_document(file: UploadFile = File(...), limit: int = Query(20, ge=1, le=200),
                           _=Depends(require_hr_admin)):
    """Show how a document would be chunked, without storing anything."""
    filename, data = await read_upload(file)
    try:
        chunks = await run_in_threadpool(preview_chunks, filename, data)
    except IngestError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Chunking failed: {type(exc).__name__}: {exc}")
    return {"document": filename, "total_chunks": len(chunks), "chunks": chunks[:limit]}


@app.get("/api/documents")
def documents(_=Depends(current_session)):
    return {"collection": config.QDRANT_COLLECTION, "documents": list_documents()}


@app.get("/api/health")
def health():
    status = {"chat_model": config.CHAT_MODEL, "embed_model": config.EMBED_MODEL,
              "chunk_model": config.CHUNK_EMBED_MODEL}
    try:
        models = {m.model for m in ollama_client().list().models}
        status["ollama"] = "ok"
        status["missing_models"] = [m for m in {config.CHAT_MODEL, config.EMBED_MODEL, config.CHUNK_EMBED_MODEL}
                                    if m not in models and f"{m}:latest" not in models]
    except Exception as exc:
        status["ollama"] = f"unreachable: {exc}"
    try:
        client = qdrant_client()
        status["qdrant"] = "ok" if client.collection_exists(config.QDRANT_COLLECTION) \
            else f"collection '{config.QDRANT_COLLECTION}' missing - run python -m app.ingest"
    except Exception as exc:
        status["qdrant"] = f"unreachable: {exc}"
    return status
