from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

SEC_ROOT = "https://www.sec.gov"


@dataclass(frozen=True)
class Filing:
    cik: str
    company: str
    form: str
    filed: str
    submission_path: str
    accession: str


@dataclass(frozen=True)
class Exhibit:
    filing: Filing
    sequence: str
    description: str
    filename: str
    exhibit_type: str
    url: str


class SecClient:
    """Small EDGAR client that observes SEC fair-access requirements."""

    def __init__(self, user_agent: str, requests_per_second: float = 5.0, timeout: int = 60):
        if "@" not in user_agent:
            raise ValueError("SEC_USER_AGENT must identify you and include a contact email.")
        if requests_per_second <= 0 or requests_per_second > 10:
            raise ValueError("requests_per_second must be in (0, 10].")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.min_interval = 1.0 / requests_per_second
        self.timeout = timeout
        self._last_request = 0.0

    def _throttle(self) -> None:
        delay = self.min_interval - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30), reraise=True)
    def get(self, url: str) -> requests.Response:
        self._throttle()
        response = self.session.get(url, timeout=self.timeout)
        self._last_request = time.monotonic()
        if response.status_code in {403, 429, 500, 502, 503, 504}:
            response.raise_for_status()
        response.raise_for_status()
        return response

    def download(self, url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return path
        tmp = path.with_suffix(path.suffix + ".part")
        response = self.get(url)
        tmp.write_bytes(response.content)
        tmp.replace(path)
        return path


def quarter_index_url(year: int, quarter: int) -> str:
    if quarter not in {1, 2, 3, 4}:
        raise ValueError("quarter must be 1..4")
    return f"{SEC_ROOT}/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"


def parse_master_index(text: str) -> list[Filing]:
    marker = "CIK|Company Name|Form Type|Date Filed|Filename"
    pos = text.find(marker)
    if pos < 0:
        raise ValueError("Unexpected EDGAR master.idx format")
    lines = text[pos:].splitlines()[1:]
    filings: list[Filing] = []
    reader = csv.reader(lines, delimiter="|")
    for row in reader:
        if len(row) != 5:
            continue
        cik, company, form, filed, path = (x.strip() for x in row)
        accession_match = re.search(r"(\d{10}-\d{2}-\d{6})\.txt$", path)
        if not accession_match:
            continue
        filings.append(Filing(cik, company, form, filed, path, accession_match.group(1)))
    return filings


def filing_index_url(filing: Filing) -> str:
    accession_plain = filing.accession.replace("-", "")
    return f"{SEC_ROOT}/Archives/edgar/data/{int(filing.cik)}/{accession_plain}/{filing.accession}-index.html"


AGREEMENT_WORDS = re.compile(
    r"\b(agreement|contract|indenture|lease|license|licence|plan|employment|purchase|merger|"
    r"confidentiality|non[- ]disclosure|credit|loan|guaranty|guarantee|services|supply|distribution|"
    r"collaboration|settlement|separation|consulting|compensation)\b",
    re.I,
)


def extract_contract_exhibits(html: str, filing: Filing) -> list[Exhibit]:
    soup = BeautifulSoup(html, "lxml")
    results: list[Exhibit] = []
    base = filing_index_url(filing)
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if not any("document" in h for h in headers) or not any("type" in h for h in headers):
            continue
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            values = [c.get_text(" ", strip=True) for c in cells]
            sequence = values[0]
            description = values[1] if len(values) > 1 else ""
            link = row.find("a", href=True)
            filename = link.get_text(" ", strip=True) if link else ""
            exhibit_type = values[3] if len(values) > 3 else ""
            if not filename or filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".css", ".js", ".xml", ".xsd")):
                continue
            is_ex10 = exhibit_type.upper().startswith("EX-10")
            is_agreement = bool(AGREEMENT_WORDS.search(description))
            if is_ex10 or is_agreement:
                results.append(Exhibit(filing, sequence, description, filename, exhibit_type, urljoin(base, link["href"])))
    # Deduplicate links occasionally repeated in the filing page.
    return list({e.url: e for e in results}.values())


def discover_exhibits(client: SecClient, filings: Iterable[Filing]) -> Iterable[Exhibit]:
    for filing in filings:
        try:
            html = client.get(filing_index_url(filing)).text
            yield from extract_contract_exhibits(html, filing)
        except requests.RequestException:
            continue


def save_exhibit(client: SecClient, exhibit: Exhibit, raw_dir: Path) -> tuple[Path, Path]:
    safe_accession = exhibit.filing.accession.replace("-", "")
    target_dir = raw_dir / safe_accession
    document_path = client.download(exhibit.url, target_dir / exhibit.filename)
    metadata_path = target_dir / f"{exhibit.filename}.metadata.json"
    import json
    metadata_path.write_text(json.dumps(asdict(exhibit), ensure_ascii=False, indent=2), encoding="utf-8")
    return document_path, metadata_path
