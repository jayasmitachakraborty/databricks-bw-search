"""Chunking utilities for RAG: paragraph-first merge, then overlap windows with soft breaks.

Tuned for web / HTML-parsed text. Default ``max_chars`` ≈ 600–800 tokens at ~4 chars/token;
set explicitly to match your embedding model limit and retrieval granularity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_MAX_CHARS = 2500
DEFAULT_OVERLAP = 300
# Attach very small tail pieces to the previous chunk when it fits (reduces sparse vectors).
MIN_CHUNK_CHARS = 80


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = DEFAULT_MAX_CHARS
    overlap: int = DEFAULT_OVERLAP
    min_soft_break: int = 120


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _soft_break_length(segment: str, max_len: int, min_keep: int) -> int:
    """Pick a split point before ``max_len``, preferring paragraph/sentence/word boundaries."""
    if len(segment) <= max_len:
        return len(segment)
    region = segment[:max_len]
    for sep in ("\n\n", "\n", ". ", "。", "? ", "! ", "; ", ", ", " "):
        pos = region.rfind(sep, min_keep, max_len)
        if pos != -1:
            return pos + len(sep)
    return max_len


def _window_chunk(text: str, max_chars: int, overlap: int, min_soft_break: int) -> list[str]:
    """Split a long string into overlapping chunks with soft boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    out: list[str] = []
    start = 0
    stride_min = max(1, max_chars - overlap)
    min_break = min(min_soft_break, max_chars // 2)

    while start < len(text):
        end = min(start + max_chars, len(text))
        segment = text[start:end]
        if end >= len(text):
            piece = segment.strip()
            if piece:
                out.append(piece)
            break

        take = _soft_break_length(segment, max_len=len(segment), min_keep=min_break)
        piece = text[start : start + take].strip()
        if piece:
            out.append(piece)

        next_start = start + take - overlap
        if next_start <= start:
            next_start = start + take
        start = next_start

    return out


def _attach_small_chunks(chunks: list[str], max_chars: int, min_size: int = MIN_CHUNK_CHARS) -> list[str]:
    if not chunks:
        return []
    out: list[str] = [chunks[0]]
    for c in chunks[1:]:
        c = c.strip()
        if not c:
            continue
        if len(c) < min_size and out:
            prev = out[-1]
            merged = f"{prev}\n\n{c}".strip()
            if len(merged) <= max_chars:
                out[-1] = merged
            else:
                out.append(c)
        else:
            out.append(c)
    return out


def chunk_text(
    text: str | None,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    config: ChunkConfig | None = None,
) -> list[str]:
    """
    Split document text into chunks for embedding.

    1. Normalize newlines and split on blank lines into paragraphs.
    2. Greedy-merge consecutive short paragraphs up to ``max_chars``.
    3. Any paragraph larger than ``max_chars`` is split with overlapping windows; cuts favor
       newlines, sentence punctuation, then whitespace.
    """
    cfg = config or ChunkConfig(max_chars=max_chars, overlap=overlap)
    max_c = cfg.max_chars
    ov = min(cfg.overlap, max_c - 1) if max_c > 1 else 0

    if text is None:
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    join_len = 2

    def flush_buf() -> None:
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n\n".join(buf))
            buf = []
            buf_len = 0

    for para in paragraphs:
        if len(para) > max_c:
            flush_buf()
            chunks.extend(_window_chunk(para, max_c, ov, cfg.min_soft_break))
            continue

        added = len(para) + (join_len if buf else 0)
        if buf_len + added <= max_c:
            buf.append(para)
            buf_len += added
        else:
            flush_buf()
            buf = [para]
            buf_len = len(para)

    flush_buf()

    final: list[str] = []
    for piece in chunks:
        if len(piece) > max_c:
            final.extend(_window_chunk(piece, max_c, ov, cfg.min_soft_break))
        else:
            final.append(piece)

    return _attach_small_chunks([c.strip() for c in final if c.strip()], max_c)
