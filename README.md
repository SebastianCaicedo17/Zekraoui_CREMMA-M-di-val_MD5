# Zekraoui_CREMMA-Médiéval_MD5

**Projet MD5 — 2026 | Master Data/IA | HETIC**  
Module « Vision par ordinateur » — Pipeline HTR + Analyse NLP sur manuscrits anciens

Pipeline de bout en bout pour la reconnaissance automatique de texte manuscrit (HTR)
sur des manuscrits médiévaux en vieux/moyen français (corpus CREMMA Médiéval),
suivi d'une analyse linguistique (NLP) : normalisation, lexique, NER.

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
│   ├── utils.py                # Reproductibilité, logging, seeds
│   ├── preprocessing.py        # Pipeline prétraitement images (binarisation, etc.)
│   ├── segmentation.py         # Segmentation layout & lignes (PAGE XML)
│   ├── data_loader.py          # Chargement ALTO XML & détection abréviations
│   ├── htr.py                  # Fine-tuning TrOCR LoRA & inférence
│   ├── aggregation.py          # Génération dataset NLP livrable (val/test JSON)
│   ├── evaluation.py           # CER, WER, bootstrap IC, McNemar
│   ├── nlp.py                  # Pipeline NLP : alphabet, lexique, normalisation, dict
│   └── ner.py                  # NER sur transcriptions médiévales (CamemBERT)
├── tests/                      # Suite pytest (66 tests)
│   ├── test_utils.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_htr.py
│   ├── test_data_contract.py
│   ├── test_nlp.py             # 48 tests — normalisation, lexique, métriques
│   └── test_ner.py             # 18 tests — NER mocké sans GPU requis
├── experiments/
│   ├── journal.jsonl           # Journal de tous les runs (2 runs × 7 époques)
│   └── rapport_test_final.json # Rapport évaluation test set scellé
├── nlp/                        # Outputs pipeline NLP
│   ├── alphabet.json           # 131 caractères distincts (96 prédictions / 114 oracle)
│   ├── regex_rules.json        # 111 règles de normalisation (20 diplo + 85 norm + 6 edit)
│   ├── lexique.json            # 15 837 formes issues des prédictions HTR
│   ├── comparaison_dictionnaire.json  # Couverture lexicale : 34.2 %
│   ├── precision_tokens.json   # Précision/Rappel/F1 token : 58.3 % / 58.0 % / 57.9 %
│   └── entites.json            # NER : 1 180 entités sur 2 733 lignes (889 KB)
├── data/
│   └── dictionnaire_medieval.json    # 4 046 formes médiévales de référence
├── dataset_nlp/                # JSON livrable pour le module NLP
│   ├── val.json                # 2 733 entrées avec prédictions HTR
│   ├── test.json               # 251 entrées avec prédictions HTR
│   └── val_oracle.json         # 2 733 entrées avec transcriptions de référence
├── segmentations/              # Fichiers PAGE XML polygones
├── colab_htr_cremma.ipynb      # Notebook Colab/Kaggle clé en main
├── MODEL_CARD.md
├── CONVENTIONS_TRANSCRIPTION.md
├── DATA_SOURCES.md
└── requirements.txt
```

---

## Résultats HTR

| Métrique | Valeur | Seuil validation | Seuil excellence |
|---|---|---|---|
| CER global | **13.88 %** ✅ | < 15 % | < 8 % |
| CER abréviations | 16.39 % | — | — |
| CER sans abréviations | 13.31 % | — | — |
| delta_CER (impact abréviations) | +3.08 % | — | — |
| WER global | 47.28 % | < 25 % | < 15 % |
| needs_review (test) | 1.2 % ✅ | < 30 % | < 20 % |
| IC 95 % CER global | [12.31 %, 15.66 %] | — | — |
| McNemar fine-tuné vs baseline | chi²=18.05, p≈0 ✅ | — | — |

**Baseline (avant fine-tuning)** : CER = 99.08 % — TrOCR n'a jamais vu du vieux français.  
**Après 6 époques LoRA** : CER = 13.88 % — réduction de 85 points de pourcentage.

Voir [MODEL_CARD.md](MODEL_CARD.md) pour le détail époque par époque et l'analyse des abréviations.

---

## Pipeline NLP (Volet 2)

Le volet NLP prend en entrée les transcriptions produites par le modèle HTR
et effectue cinq analyses linguistiques.

### Vue d'ensemble

```
Transcriptions HTR (val.json)
        |
        v
[1] Alphabet      -> nlp/alphabet.json        131 caractères distincts
        |
        v
[2] Normalisation -> nlp/regex_rules.json     111 règles (diplomatique + normalisé)
        |
        v
[3] Lexique       -> nlp/lexique.json         15 837 formes, 72.6 % hapax
        |
        v
[4] Dictionnaire  -> nlp/comparaison_dict.json  34.2 % couverture
        |
        v
[5] Métriques     -> nlp/precision_tokens.json  F1 token = 57.9 %
        |
        v
[6] NER           -> nlp/entites.json         1 180 entités (PER/LOC/ORG/MISC)
```

### Étape 1 — Alphabet

Inventaire de tous les caractères présents dans les prédictions HTR et les
transcriptions de référence (oracle). Identifie les caractères manqués par le modèle.

- **131 caractères** distincts au total
- **96** dans les prédictions, **114** dans l'oracle
- **35 caractères** présents dans l'oracle mais absents des prédictions
  (principalement des caractères d'abréviation : `ꝓ`, macrons, voyelles suscrites)

### Étape 2 — Normalisation

Deux modes de nettoyage basés sur 111 règles regex issues de `table.csv` :

| Mode | Ce qu'il fait | Usage |
|---|---|---|
| **Diplomatique** | Développe uniquement les abréviations Unicode (`ꝯ→con`, `ꝑ→p`, `⁊→et`) | Lexique, métriques token |
| **Normalisé** | En plus : `v→u`, `j→i`, `ê→e`, `ſ→s`, suppression balises `[---]` | NER, analyse linguistique |

Les règles sont entièrement traçables dans `nlp/regex_rules.json`.

### Étape 3 — Lexique

Inventaire de toutes les formes produites par le modèle HTR (construit depuis
les prédictions, pas depuis l'oracle) :

- **15 837 formes** distinctes
- **11 491 hapax** (72.6 %) — formes qui n'apparaissent qu'une fois
- Top 5 : `de` (1391), `et` (1288), `a` (1141), `la` (968), `le` (895)

Le taux élevé d'hapax reflète à la fois la richesse morphologique du vieux français
et les erreurs de transcription HTR qui génèrent des formes inexistantes.

### Étape 4 — Couverture dictionnaire

Comparaison des formes du lexique HTR avec un dictionnaire de référence médiéval
(4 046 formes extraites des transcriptions oracle + formes médiévales classiques).

| Métrique | Valeur |
|---|---|
| Taux de couverture global | **34.2 %** |
| Couverture — lignes sans abréviations | 34.4 % |
| Couverture — lignes avec abréviations | 25.0 % |

Répartition des 2 858 formes hors dictionnaire :

| Catégorie | Nombre |
|---|---|
| Probable erreur HTR | 1 477 |
| Forme rare ou dialectale | 1 247 |
| Fragment de mot | 108 |
| Abréviation non développée | 26 |

### Étape 5 — Précision token

Mesure de la qualité de transcription au niveau du mot (token), en comparant
les prédictions HTR aux transcriptions oracle ligne par ligne.

Méthode : intersection multiensemble (Counter) entre tokens prédits et tokens oracle.

| Métrique | Global | Avec abréviations | Sans abréviations |
|---|---|---|---|
| Précision | 58.3 % | 56.1 % | 58.6 % |
| Rappel | 58.0 % | 55.6 % | 58.4 % |
| **F1 token** | **57.9 %** | **55.6 %** | **58.3 %** |
| Nombre de lignes | 2 733 | 382 | 2 351 |

Le F1 token de 57.9 % est cohérent avec un WER de 47.3 % : environ la moitié des
mots sont correctement transcrits.

### Étape 6 — NER (Reconnaissance d'Entités Nommées)

Détection automatique des entités nommées dans les transcriptions normalisées.

**Modèle** : `Jean-Baptiste/camembert-ner` — CamemBERT fine-tuné sur WikiNER français.
Appliqué sur le texte en mode **normalisé** pour se rapprocher du français moderne
que le modèle connaît.

| Type | Entités | Formes uniques | Exemples |
|---|---|---|---|
| PER (personnes) | **865** | ~800 | damoisele, dorus |
| LOC (lieux) | **180** | ~170 | france, terre, siont |
| ORG (organisations) | **33** | ~30 | lemonde |
| MISC (divers) | **102** | ~90 | clerigie, romans |
| **Total** | **1 180** | — | — |

**Filtre qualité** : score ≥ 0.80, longueur > 2 caractères, liste de 50 faux positifs
grammaticaux médiévaux (`si`, `cil`, `li`, `sire`…).

**Limite** : aucun dataset NER annoté n'existe pour le corpus CREMMA — le F1 NER
ne peut pas être calculé. Les résultats sont **exploratoires**.

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
> Le test set est scellé — hash SHA-256 enregistré dans MODEL_CARD.md.

### Option A — Notebook Colab / Kaggle (recommandé)

Ouvrir `colab_htr_cremma.ipynb` et exécuter toutes les cellules dans l'ordre.  
Requiert un GPU (T4 minimum) pour le fine-tuning (~20 min sur T4).

### Option B — Pipeline NLP seul (CPU, sans GPU)

Si le dataset `dataset_nlp/val.json` est déjà disponible :

```bash
# Lancer le pipeline NLP complet (alphabet, lexique, normalisation, dict, métriques)
python -m src.nlp

# Lancer la NER (télécharge ~420 MB de modèle au premier appel)
python -m src.ner

# Lancer tous les tests
pytest tests/ -v --cov=src
```

### Option C — Local (reconstruction complète)

```bash
# 1. Cloner les données CREMMA
git clone https://github.com/HTR-United/cremma-medieval.git data/cremma-medieval

# 2. Construire le dataset lignes
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

## Hashes SHA-256

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

66 tests au total : 48 pour le module NLP, 18 pour le module NER (pipeline mocké,
aucun GPU requis).

---

## Données & licences

Voir [DATA_SOURCES.md](DATA_SOURCES.md).

## Conventions de transcription

Voir [CONVENTIONS_TRANSCRIPTION.md](CONVENTIONS_TRANSCRIPTION.md).

## Model Card

Voir [MODEL_CARD.md](MODEL_CARD.md) — architecture détaillée, courbes d'apprentissage,
performances test, analyse des abréviations.
