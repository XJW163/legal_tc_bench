from __future__ import annotations

import re
from dataclasses import dataclass, asdict

QUOTE = r'["“”]'
TERM_BODY = r"[A-Z][A-Za-z0-9&'’\-]*(?:\s+(?:of|the|and|or|for|to|in|on|by|with|[A-Z][A-Za-z0-9&'’\-]*)){0,10}"

DEFINED_PATTERNS = [
    ("means-clause", re.compile(rf"{QUOTE}(?P<term>{TERM_BODY}){QUOTE}\s+(?:means|shall mean|has the meaning)", re.I)),
    ("means-clause", re.compile(rf"(?P<term>{TERM_BODY})\s+(?:means|shall mean|has the meaning)\b")),
    # SEC contracts very often define terms parenthetically:
    # This Agreement (“Agreement”), ... February 28, 2023 (“Effective Date”)
    ("parenthetical-defined-term", re.compile(rf"\(\s*(?:the\s+)?{QUOTE}(?P<term>{TERM_BODY}){QUOTE}\s*\)", re.I)),
]


@dataclass
class Occurrence:
    paragraph_id: int
    char_start: int
    char_end: int
    surface: str
    context: str


@dataclass
class TermRecord:
    term_id: str
    source_term: str
    extraction_method: str
    definition_paragraph_ids: list[int]
    occurrences: list[Occurrence]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_defined_terms(paragraphs: list[str], min_occurrences: int = 3) -> list[TermRecord]:
    definitions: dict[str, set[int]] = {}
    canonical: dict[str, str] = {}
    methods: dict[str, set[str]] = {}

    for pid, paragraph in enumerate(paragraphs):
        for method, pattern in DEFINED_PATTERNS:
            for match in pattern.finditer(paragraph):
                term = re.sub(r"\s+", " ", match.group("term")).strip(" .,:;()[]")
                if not _valid_term(term):
                    continue
                key = term.casefold()
                canonical.setdefault(key, term)
                definitions.setdefault(key, set()).add(pid)
                methods.setdefault(key, set()).add(method)

    # Link occurrences longest-term-first so "Agreement" is not counted inside
    # "Collaboration Agreement" when both are defined terms.
    occurrence_map: dict[str, list[Occurrence]] = {key: [] for key in canonical}
    ordered_keys = sorted(canonical, key=lambda k: (-len(canonical[k]), k))
    for pid, paragraph in enumerate(paragraphs):
        occupied: list[tuple[int, int]] = []
        for key in ordered_keys:
            term = canonical[key]
            regex = re.compile(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.I)
            for match in regex.finditer(paragraph):
                span = (match.start(), match.end())
                if any(not (span[1] <= a or span[0] >= b) for a, b in occupied):
                    continue
                left = max(0, match.start() - 140)
                right = min(len(paragraph), match.end() + 140)
                occurrence_map[key].append(
                    Occurrence(pid, match.start(), match.end(), match.group(0), paragraph[left:right])
                )
                occupied.append(span)

    records: list[TermRecord] = []
    for key, term in canonical.items():
        occurrences = occurrence_map[key]
        if len(occurrences) >= min_occurrences:
            records.append(TermRecord(
                "", term, "+".join(sorted(methods[key])), sorted(definitions[key]), occurrences
            ))

    records.sort(key=lambda x: (-len(x.occurrences), x.source_term.casefold()))
    for i, record in enumerate(records, 1):
        record.term_id = f"T{i:04d}"
    return records


def _valid_term(term: str) -> bool:
    if len(term) < 2 or len(term) > 120:
        return False
    words = term.split()
    if len(words) > 11:
        return False
    stop = {"This", "That", "The", "A", "An", "For", "If", "In", "As", "It"}
    return term not in stop
