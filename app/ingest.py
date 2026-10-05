"""Index HR policy documents (.md, .txt, .pdf, .docx) into Qdrant using semantic chunking.

Used by both the CLI and the upload API.
Usage: python -m app.ingest             # index every file in POLICY_DIR (re-indexes changed documents)
       python -m app.ingest --rebuild   # drop the collection first (needed after changing EMBED_MODEL)
"""
import argparse
import io
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import models

from . import config
from .chunking import semantic_chunks
from .rag import embed, qdrant_client


class IngestError(Exception):
    pass


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(data))
        lines = []
        for para in doc.paragraphs:
            style = (para.style.name or "").lower()
            level = re.search(r"heading (\d)", style)
            lines.append(f"{'#' * int(level.group(1))} {para.text}" if level and para.text else para.text)
        return "\n".join(lines)
    if suffix in {".md", ".txt"}:
        return data.decode("utf-8", errors="replace")
    raise IngestError(f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}")


def ensure_collection(vector_size: int) -> None:
    client = qdrant_client()
    if client.collection_exists(config.QDRANT_COLLECTION):
        existing = client.get_collection(config.QDRANT_COLLECTION).config.params.vectors.size
        if existing != vector_size:
            raise IngestError(f"Collection '{config.QDRANT_COLLECTION}' has {existing}-dim vectors but "
                              f"'{config.EMBED_MODEL}' produces {vector_size}. Run: python -m app.ingest --rebuild")
        return
    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    if config.QDRANT_URL != ":memory:":  # payload indexes only exist on a Qdrant server
        client.create_payload_index(config.QDRANT_COLLECTION, "document", models.PayloadSchemaType.KEYWORD)


def ingest_document(filename: str, data: bytes, uploaded_by: str | None = None) -> dict:
    """Chunk, embed and store one document. Chunks from a previous version of the same file are replaced."""
    text = extract_text(filename, data)
    if not text.strip():
        raise IngestError("No extractable text found (scanned PDFs need OCR first).")

    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    chunks = semantic_chunks(text, title)
    vectors = []
    for i in range(0, len(chunks), 32):
        vectors += embed([c["text"] for c in chunks[i:i + 32]], "document")
    ensure_collection(len(vectors[0]))

    client = qdrant_client()
    same_doc = models.Filter(must=[models.FieldCondition(key="document", match=models.MatchValue(value=filename))])
    client.delete(config.QDRANT_COLLECTION, points_selector=models.FilterSelector(filter=same_doc))
    indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=[
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}|{i}")), vector=vector,
            payload={"document": filename, "section": c["section"], "chunk": i, "text": c["text"],
                     "sentences": c["sentences"], "indexed_at": indexed_at, "uploaded_by": uploaded_by,
                     "embed_model": config.EMBED_MODEL})
        for i, (c, vector) in enumerate(zip(chunks, vectors))])

    sizes = [len(c["text"]) for c in chunks]
    return {"document": filename, "chunks": len(chunks), "characters": len(text),
            "avg_chunk_chars": round(sum(sizes) / len(sizes)), "max_chunk_chars": max(sizes),
            "embed_model": config.EMBED_MODEL, "chunk_model": config.CHUNK_EMBED_MODEL,
            "vector_size": len(vectors[0])}


def list_documents() -> list[dict]:
    client = qdrant_client()
    if not client.collection_exists(config.QDRANT_COLLECTION):
        return []
    docs, offset = {}, None
    while True:
        points, offset = client.scroll(config.QDRANT_COLLECTION, limit=256, offset=offset,
                                       with_payload=["document", "indexed_at", "uploaded_by"], with_vectors=False)
        for p in points:
            doc = docs.setdefault(p.payload["document"], {"document": p.payload["document"], "chunks": 0,
                                                          "indexed_at": p.payload.get("indexed_at"),
                                                          "uploaded_by": p.payload.get("uploaded_by")})
            doc["chunks"] += 1
        if offset is None:
            break
    return sorted(docs.values(), key=lambda d: d["document"])


def preview_chunks(filename: str, data: bytes) -> list[dict]:
    """Chunk without storing - handy for tuning the chunking parameters."""
    title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    return semantic_chunks(extract_text(filename, data), title)


def ingest_directory(rebuild: bool = False) -> None:
    files = sorted(p for p in config.POLICY_DIR.iterdir() if p.suffix.lower() in config.ALLOWED_EXTENSIONS)
    if not files:
        raise SystemExit(f"No policy documents found in {config.POLICY_DIR}")
    client = qdrant_client()
    if rebuild and client.collection_exists(config.QDRANT_COLLECTION):
        client.delete_collection(config.QDRANT_COLLECTION)
        print(f"Dropped collection '{config.QDRANT_COLLECTION}'")
    print(f"Indexing {len(files)} file(s) from {config.POLICY_DIR} "
          f"(chunk model: {config.CHUNK_EMBED_MODEL}, embed model: {config.EMBED_MODEL})")
    for path in files:
        stats = ingest_document(path.name, path.read_bytes(), uploaded_by="cli")
        print(f"  {path.name}: {stats['chunks']} chunks (avg {stats['avg_chunk_chars']} chars)")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="drop and recreate the Qdrant collection")
    ingest_directory(parser.parse_args().rebuild)
