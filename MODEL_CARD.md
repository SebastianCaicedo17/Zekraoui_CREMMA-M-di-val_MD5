# Model Card — HTR CREMMA Médiéval 2026

## Description

Modèle de reconnaissance automatique de texte manuscrit (HTR) fine-tuné sur le corpus CREMMA Médiéval.  
Produit des transcriptions textuelles de manuscrits médiévaux en vieux/moyen français (XIe–XVIIe s.).

## Données d'entraînement

| Corpus | Période | Lignes | Licence |
|---|---|---|---|
| À compléter | — | — | — |

- Train SHA-256 : `À compléter`
- Split : 70 % train / 15 % val / 15 % test

## Architecture

| Composant | Détail |
|---|---|
| Modèle base | `microsoft/trocr-base-handwritten` |
| Fine-tuning | LoRA (`peft`), r = À compléter |
| Segmentation | Kraken BLLA |
| Prétraitement | CLAHE + Sauvola |

## Performances (test set scellé)

| Métrique | Valeur | IC 95 % |
|---|---|---|
| CER global | À compléter | — |
| WER global | À compléter | — |
| IoU segmentation | À compléter | — |
| Taux needs_review | À compléter | — |

## Limitations

- À compléter après entraînement
- Biais de représentation du corpus : voir section Discussion de l'article

## Usage

```python
from src.htr import transcrire_page

transcription = transcrire_page("chemin/vers/image.jpg")
print(transcription)
```

## Citation

```bibtex
@misc{zekraoui2026htr,
  title  = {HTR CREMMA Médiéval 2026},
  author = {À compléter},
  year   = {2026},
  note   = {Projet MD5 — Master Data/IA HETIC}
}
```