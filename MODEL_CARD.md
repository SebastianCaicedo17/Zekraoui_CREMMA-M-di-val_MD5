# Model Card — HTR CREMMA Médiéval 2026

## Description

Modèle HTR fine-tuné sur le corpus CREMMA Médiéval (CC-BY 4.0).  
Spécialisé vieux/moyen français XIe–XVIIe s. — manuscrits BnF et KBR.  
Axe de recherche : impact des abréviations médiévales sur les performances HTR.

---

## Données

| Split | Manuscrits | Lignes | SHA-256 |
|---|---|---|---|
| Train | 10 | 6 891 | `de26aaa226edad415b37e02a6e3e68a5f001968a3ec69daddb3c47b69d9767d8` |
| Val   | 2  | 2 733 | `b7891b9a73b9704e37623b7dfec43677a7a94da2f58aa94557254dfd1a03a4c5` |
| Test  | 2  |   251 | `86f5f48b7628128b4e86bc5684b13f7409972bf3d7dc39291d97b03d25f62949` |

Taux d'abréviations : 14.0 % (val) — 18.7 % (test).

---

## Architecture

| Composant | Détail |
|---|---|
| Modèle de base | `microsoft/trocr-base-handwritten` |
| Fine-tuning | LoRA r=8, alpha=32, dropout=0.1 |
| Modules adaptés | `all-linear` |
| Optimiseur | AdamW lr=1e-4, patience=3 |
| Précision | fp16 + gradient checkpointing |
| Epochs | 7 (early stopping sur CER val) |
| Batch size | 16 |

---

## Performances (test set scellé)

| Métrique | Valeur | IC 95 % bootstrap |
|---|---|---|
| CER global | **13.92 %** | [12.38 %, 15.49 %] |
| CER abréviations | 16.71 % | — |
| CER sans abréviations | 13.28 % | — |
| delta_CER | +3.43 % | [-0.00 %, 7.10 %] |
| WER global | 47.11 % | — |
| needs_review | 1.2 % | — |

**McNemar fine-tuné vs baseline** : chi²=19.05, p≈0 — amélioration statistiquement significative.

---

## Conclusion

Le CER global atteint **13.92 %**, sous le seuil objectif de 15 %.  
Les abréviations médiévales augmentent le CER de **+3.43 points** (13.28 % → 16.71 %), impact modéré mais statistiquement significatif.  
Caractères les plus impactants : `ͤ` (e souscrit), `ꝯ` (abréviation con/com), `̃` (tilde de nasalisation).  
Les abréviations représentent 18.7 % du corpus test.

---

## Limitations

- Performances dégradées sur les manuscrits très endommagés ou à encre effacée.
- WER élevé (47.11 %) : sensible aux variantes graphiques du vieux français.
- Modèle entraîné uniquement sur CREMMA — généralisation à d'autres corpus non garantie.

---

## Utilisation

```python
from peft import PeftModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
base = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model = PeftModel.from_pretrained(base, "checkpoints/trocr_lora_r8_best")
model.eval()

image = Image.open("ligne.jpg").convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values
ids = model.generate(pixel_values=pixel_values)
print(processor.decode(ids[0], skip_special_tokens=True))
```

---

## Licence

Modèle dérivé de `microsoft/trocr-base-handwritten` (MIT) et du corpus CREMMA Médiéval (CC-BY 4.0).
