"""Utilitaires globaux : reproductibilité, logging d'expériences."""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def fixer_seeds(seed: int = 42) -> None:
    """Fixe toutes les sources d'aléatoire pour garantir la reproductibilité.

    Args:
        seed: Valeur de la graine. Utiliser 42 par convention dans ce projet.

    Example:
        >>> fixer_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_experience(
    journal_path: str | Path,
    run_name: str,
    hyperparams: dict[str, Any],
    metrics: dict[str, float],
    checkpoint: str | None = None,
    notes: str = "",
) -> None:
    """Enregistre un run d'entraînement dans le journal JSONL.

    Args:
        journal_path: Chemin vers experiments/journal.jsonl.
        run_name: Identifiant unique du run (ex. "trocr_lora_r8_ep5").
        hyperparams: Dictionnaire des hyperparamètres utilisés.
        metrics: Dictionnaire des métriques (ex. {"cer_val": 0.12, "wer_val": 0.20}).
        checkpoint: Chemin vers le checkpoint sauvegardé, si applicable.
        notes: Observations libres sur le run.

    Raises:
        OSError: Si le fichier journal n'est pas accessible en écriture.

    Example:
        >>> log_experience(
        ...     "experiments/journal.jsonl",
        ...     run_name="trocr_lora_r8",
        ...     hyperparams={"lr": 5e-5, "epochs": 10, "lora_r": 8},
        ...     metrics={"cer_val": 0.14, "wer_val": 0.22},
        ... )
    """
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "run_name": run_name,
        "hyperparams": hyperparams,
        "metrics": metrics,
        "checkpoint": checkpoint,
        "notes": notes,
    }
    with open(journal_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
