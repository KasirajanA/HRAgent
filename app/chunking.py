"""Semantic chunking.

Instead of cutting text every N characters, we:
  1. split each markdown section into sentences (headings are always hard boundaries),
  2. embed every sentence together with its neighbours (a small sliding window smooths noise),
  3. measure the cosine distance between consecutive sentence embeddings,
  4. start a new chunk where the distance is unusually large (above the Nth percentile for the document),
     i.e. where the topic shifts, while keeping chunks between CHUNK_MIN_CHARS and CHUNK_MAX_CHARS.

A tiny embedding model (all-minilm, 22M params) is enough for boundary detection: it only has to tell
whether two adjacent sentences talk about the same thing.
"""
import math
import re

from . import config
from .rag import embed

HEADING = re.compile(r"^(#{1,4})\s+(.*)$")
# Sentence ends at . ! ? followed by whitespace + an uppercase letter/digit/bullet, or at a line break
# before a list item / table row.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\-*•])|\n(?=\s*(?:[-*•]|\d+[.)]|\|))|\n{2,}")


def split_sections(text: str, title: str) -> list[tuple[str, str]]:
    """Split markdown into (heading path, body) pairs so each chunk knows which section it came from."""
    sections, path, buffer = [], [title], []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append((" > ".join(path), body))
        buffer.clear()

    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = path[:level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return sections


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text)]
    sentences = []
    for part in filter(None, parts):
        # Hard-wrap pathological "sentences" (e.g. PDF text without punctuation).
        while len(part) > config.CHUNK_MAX_CHARS:
            cut = part.rfind(" ", 0, config.CHUNK_MAX_CHARS)
            cut = cut if cut > 0 else config.CHUNK_MAX_CHARS
            sentences.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            sentences.append(part)
    return sentences


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1 - dot / norm if norm else 1.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def semantic_chunks(text: str, title: str) -> list[dict]:
    """Return [{'section', 'text', 'sentences'}] for one document."""
    sections = [(heading, split_sentences(body)) for heading, body in split_sections(text, title)]
    sections = [(h, s) for h, s in sections if s]
    flat = [s for _, sents in sections for s in sents]
    if not flat:
        return []

    # Embed each sentence with a window of neighbours, in one batched pass over the whole document.
    w = config.CHUNK_WINDOW
    windows = []
    for _, sents in sections:
        windows += [" ".join(sents[max(0, i - w):i + w + 1]) for i in range(len(sents))]
    vectors = []
    for i in range(0, len(windows), 64):
        vectors += embed(windows[i:i + 64], "document", model=config.CHUNK_EMBED_MODEL)

    # Distances between consecutive sentences within the same section -> document-level threshold.
    distances_per_section, offset = [], 0
    for _, sents in sections:
        vecs = vectors[offset:offset + len(sents)]
        distances_per_section.append([_cosine_distance(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)])
        offset += len(sents)
    threshold = _percentile([d for ds in distances_per_section for d in ds], config.CHUNK_BREAKPOINT_PERCENTILE)

    chunks = []
    for (heading, sents), distances in zip(sections, distances_per_section):
        current = [sents[0]]
        for sentence, distance in zip(sents[1:], distances):
            size = sum(len(s) + 1 for s in current)
            topic_shift = distance > threshold and size >= config.CHUNK_MIN_CHARS
            if topic_shift or size + len(sentence) > config.CHUNK_MAX_CHARS:
                chunks.append({"section": heading, "sentences": current})
                current = []
            current.append(sentence)
        # Merge a tiny trailing piece into the previous chunk of the same section when it fits.
        size = sum(len(s) + 1 for s in current)
        if (chunks and chunks[-1]["section"] == heading and size < config.CHUNK_MIN_CHARS
                and sum(len(s) + 1 for s in chunks[-1]["sentences"]) + size <= config.CHUNK_MAX_CHARS):
            chunks[-1]["sentences"] += current
        else:
            chunks.append({"section": heading, "sentences": current})

    for chunk in chunks:
        chunk["text"] = f"[{chunk['section']}]\n" + "\n".join(chunk["sentences"])
        chunk["sentences"] = len(chunk["sentences"])
    return chunks
