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
│   ├── data_loader.py          # Chargement ALTO XML & détection abréviations
│   ├── htr.py                  # Fine-tuning TrOCR LoRA & inférence
│   ├── aggregation.py          # Génération dataset NLP livrable
│   └── evaluation.py           # CER, WER, bootstrap IC, McNemar
├── tests/                      # Suite pytest (98 tests)
│   ├── test_utils.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_htr.py
│   └── test_data_contract.py
├── experiments/
│   ├── journal.jsonl           # Journal de tous les runs (7 epochs)
│   └── rapport_test_final.json # Rapport évaluation test set scellé
├── dataset_nlp/                # JSON livrable pour le module NLP
│   ├── val.json                # 2 733 entrées avec prédictions
│   ├── test.json               # 251 entrées avec prédictions
│   └── val_oracle.json         # 2 733 entrées avec transcriptions de référence
├── segmentations/              # Fichiers PAGE XML polygones
├── colab_htr_cremma.ipynb      # Notebook Colab/Kaggle clé en main
├── MODEL_CARD.md
├── CONVENTIONS_TRANSCRIPTION.md
├── DATA_SOURCES.md
└── requirements.txt
```

---

## Résultats

| Métrique | Valeur | Seuil validation | Seuil excellence |
|---|---|---|---|
| CER global | **13.92 %** ✅ | < 15 % | < 8 % |
| CER abréviations | 16.71 % | — | — |
| CER sans abréviations | 13.28 % | — | — |
| delta_CER (impact abréviations) | +3.43 % | — | — |
| WER global | 47.11 % | < 25 % | < 15 % |
| needs_review (test) | 1.2 % ✅ | < 30 % | < 20 % |
| IC 95 % CER global | [12.38 %, 15.49 %] | — | — |
| McNemar fine-tuné vs baseline | chi²=19.05, p≈0 ✅ | — | — |

**Conclusion problématique** : les abréviations médiévales augmentent le CER de +3.43 points (13.28 % → 16.71 %). L'impact est modéré mais statistiquement significatif (McNemar p≈0). Les caractères les plus impactants : `ͤ`, `ꝯ`, `̃`.

---

## Installation

```bash
git clone https://github.com/SebastianCaicedo17/Zekraoui_CREMMA-M-di-val_MD5.git
cd Zekraoui_CREMMA-Médiéval_MD5
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Reproduire les résultats

> Toutes les sources d'aléatoire sont fixées via `fixer_seeds(42)`.  
> Le test set est scellé — hash SHA-256 enregistré ci-dessous.

### Option A — Notebook Colab / Kaggle (recommandé)

Ouvrir `colab_htr_cremma.ipynb` et exécuter toutes les cellules dans l'ordre.  
Requiert un GPU (T4 minimum) pour le fine-tuning (~20 min sur T4).

### Option B — Local (CPU uniquement, sans fine-tuning)

```bash
# 1. Cloner les données CREMMA
git clone https://github.com/HTR-United/cremma-medieval.git data/cremma-medieval

# 2. Construire le dataset lignes (extraction des crops)
python -c "
from src.utils import fixer_seeds
from src.htr import construire_dataset
from pathlib import Path
fixer_seeds(42)
construire_dataset(split_path=Path('data/split.json'), out_dir=Path('data/lignes'))
"

# 3. Lancer les tests
pytest tests/ -v --cov=src
```

---

### Hashes SHA-256 des données

| Jeu | Manuscrits | Lignes | SHA-256 |
|---|---|---|---|
| Train | 10 | 6 891 | `de26aaa226edad415b37e02a6e3e68a5f001968a3ec69daddb3c47b69d9767d8` |
| Val   | 2  | 2 733 | `b7891b9a73b9704e37623b7dfec43677a7a94da2f58aa94557254dfd1a03a4c5` |
| Test  | 2  |   251 | `86f5f48b7628128b4e86bc5684b13f7409972bf3d7dc39291d97b03d25f62949` |

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
