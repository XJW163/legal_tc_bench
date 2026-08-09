from pathlib import Path

from legal_tc_bench.checkpoint import Checkpoint


def test_checkpoint_persists_payload(tmp_path: Path):
    path = tmp_path / "review.csv.checkpoint.json"
    cp = Checkpoint(path)
    cp.mark(
        0,
        {
            "signature": [{"document_id": "d1", "term_id": "T1", "source_term": "Term"}],
            "annotations": [{
                "document_id": "d1",
                "term_id": "T1",
                "term_type": "DEFINED_LEGAL_TERM",
                "criticality": "CRITICAL",
                "include_in_benchmark": True,
                "confidence": 0.9,
            }],
        },
    )

    loaded = Checkpoint(path)
    entry = loaded.get(0)
    assert entry["meta"]["annotations"][0]["term_id"] == "T1"


def test_checkpoint_atomic_file_exists(tmp_path: Path):
    path = tmp_path / "review.csv.checkpoint.json"
    Checkpoint(path).mark(1, {"size": 5})
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
