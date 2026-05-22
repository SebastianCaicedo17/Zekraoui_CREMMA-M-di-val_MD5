# Zekraoui_CREMMA-Médiéval_MD5

**Projet MD5 — 2026 | Master Data/IA | HETIC**  
Module « Vision par ordinateur » — Volet 1/2 : HTR Manuscrits Anciens

Pipeline de bout en bout pour la reconnaissance automatique de texte manuscrit (HTR) sur des manuscrits médiévaux en vieux/moyen français, basé sur le corpus CREMMA Médiéval.

---

## Équipe

| Membre | Rôle |
|---|---|
| Sebastian QUESADA CAICEDO | Responsable Technique |
| Shaïma Dayb | Responsable Documentation |
| Ihsane ZEKRAOUI | Responsable Expérimentation |

---

## Structure du dépôt

```
.
├── src/                        # Code source
│   ├── __init__.py
│   ├── utils.py                # Reproductibilité, logging
│   ├── preprocessing.py        # Pipeline prétraitement images
│   ├── segmentation.py         # Segmentation layout & lignes
│   ├── htr.py                  # Fine-tuning & inférence HTR
│   ├── aggregation.py          # Agrégation multi-modèles
│   └── evaluation.py           # CER, WER, IoU, bootstrap
├── tests/                      # Suite pytest
│   ├── __init__.py
│   ├── test_utils.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   └── test_data_contract.py
├── experiments/
│   └── journal.jsonl           # Journal de tous les runs
├── dataset_nlp/                # JSON livrable pour le module NLP
├── segmentations/              # Fichiers PAGE XML / JSON polygones
├── README.md
├── MODEL_CARD.md
├── CONVENTIONS_TRANSCRIPTION.md
├── DATA_SOURCES.md
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/<org>/Zekraoui_CREMMA-Médiéval_MD5.git
cd Zekraoui_CREMMA-Médiéval_MD5
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduire les résultats

> Toutes les sources d'aléatoire sont fixées via `fixer_seeds(42)`.  
> Le test set est scellé — hash SHA-256 enregistré ci-dessous.

```bash
# 1. Prétraitement
python src/preprocessing.py --config configs/preprocess.yaml

# 2. Segmentation
python src/segmentation.py --config configs/segment.yaml

# 3. Fine-tuning TrOCR
python src/htr.py --model trocr --lora_r 8 --config configs/train.yaml

# 4. Évaluation finale (une seule fois sur le test set)
python src/evaluation.py --model checkpoints/best_trocr.pt

# 5. Constitution du JSON livrable
python src/aggregation.py --output dataset_nlp/
```

### Hashes SHA-256 des données

| Jeu | SHA-256 |
|---|---|
| Train set (10 manuscrits, 16 640 lignes) | `de26aaa226edad415b37e02a6e3e68a5f001968a3ec69daddb3c47b69d9767d8` |
| Val set   (2 manuscrits,  4 684 lignes)  | `b7891b9a73b9704e37623b7dfec43677a7a94da2f58aa94557254dfd1a03a4c5` |
| Test set  (2 manuscrits,  1 519 lignes)  | `86f5f48b7628128b4e86bc5684b13f7409972bf3d7dc39291d97b03d25f62949` |

---

## Résultats

| Métrique | Baseline | Modèle final | Seuil validation | Seuil excellence |
|---|---|---|---|---|
| CER global | — | — | < 15 % | < 8 % |
| WER global | — | — | < 25 % | < 15 % |
| IoU segmentation | — | — | > 0,75 | > 0,85 |
| Taux needs_review | — | — | < 30 % | < 20 % |

---

## Lancer les tests

```bash
pytest tests/ -v --cov=src
```

---

## Données & licences

Voir [DATA_SOURCES.md](DATA_SOURCES.md).

## Conventions de transcription

Voir [CONVENTIONS_TRANSCRIPTION.md](CONVENTIONS_TRANSCRIPTION.md).

## Model Card

Voir [MODEL_CARD.md](MODEL_CARD.md).