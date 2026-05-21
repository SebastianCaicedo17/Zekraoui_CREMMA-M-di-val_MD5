"""Tests de reproductibilité pour src/utils.py."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.utils import fixer_seeds, log_experience


def test_fixer_seeds_numpy_reproductible():
    """Deux appels avec le même seed produisent la même séquence numpy."""
    fixer_seeds(42)
    seq1 = np.random.rand(10)

    fixer_seeds(42)
    seq2 = np.random.rand(10)

    np.testing.assert_array_equal(seq1, seq2)


def test_fixer_seeds_different_seeds():
    """Deux seeds différents produisent des séquences différentes."""
    fixer_seeds(0)
    seq0 = np.random.rand(10)

    fixer_seeds(1)
    seq1 = np.random.rand(10)

    assert not np.array_equal(seq0, seq1)


def test_log_experience_ecrit_jsonl(tmp_path: Path):
    """log_experience écrit une ligne JSON valide dans le journal."""
    journal = tmp_path / "journal.jsonl"
    journal.touch()

    log_experience(
        journal_path=journal,
        run_name="test_run",
        hyperparams={"lr": 1e-4, "epochs": 1},
        metrics={"cer_val": 0.20},
        checkpoint=None,
        notes="run de test",
    )

    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["run_name"] == "test_run"
    assert entry["metrics"]["cer_val"] == pytest.approx(0.20)
    assert "timestamp" in entry


def test_log_experience_accumule(tmp_path: Path):
    """Plusieurs appels accumulent les lignes sans écraser."""
    journal = tmp_path / "journal.jsonl"
    journal.touch()

    for i in range(3):
        log_experience(journal, run_name=f"run_{i}", hyperparams={}, metrics={"cer_val": i * 0.1})

    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3