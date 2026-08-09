from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = ["p", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"]


def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def unwrap_sec_document(content: bytes) -> bytes:
    """Extract the payload inside SEC's <TEXT> wrapper when present."""
    decoded = content.decode("utf-8", errors="replace")
    match = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", decoded, flags=re.I | re.S)
    if match:
        return match.group(1).encode("utf-8")
    return content


def html_to_paragraphs(content: bytes) -> list[str]:
    content = unwrap_sec_document(content)
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    paragraphs: list[str] = []
    # Prefer leaf-like block nodes. This avoids one giant outer <div> swallowing the document.
    for node in soup.find_all(BLOCK_TAGS):
        if node.find(BLOCK_TAGS):
            continue
        value = normalize_space(node.get_text(" ", strip=True))
        if len(value) >= 8:
            paragraphs.append(value)

    if not paragraphs:
        text = normalize_space(soup.get_text("\n", strip=True))
        paragraphs = [p.strip() for p in text.splitlines() if len(p.strip()) >= 8]
    return _deduplicate_adjacent(paragraphs)


def plain_to_paragraphs(content: bytes) -> list[str]:
    content = unwrap_sec_document(content)
    decoded = content.decode("utf-8", errors="replace")
    decoded = html_lib.unescape(decoded)
    if "<html" in decoded[:4000].lower() or "<body" in decoded[:4000].lower():
        return html_to_paragraphs(content)
    chunks = re.split(r"\n\s*\n", decoded)
    return [normalize_space(x) for x in chunks if len(normalize_space(x)) >= 8]


def _deduplicate_adjacent(paragraphs: list[str]) -> list[str]:
    out: list[str] = []
    for p in paragraphs:
        if not out or p != out[-1]:
            out.append(p)
    return out


def parse_document(path: Path) -> list[str]:
    content = path.read_bytes()
    probe = unwrap_sec_document(content)[:4000].lower()
    lower = path.name.lower()
    if lower.endswith((".htm", ".html")) or b"<html" in probe or b"<body" in probe:
        return html_to_paragraphs(content)
    return plain_to_paragraphs(content)


def write_clean_text(path: Path, output_path: Path) -> list[str]:
    paragraphs = parse_document(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    return paragraphs
