# Model Card - HTR CREMMA Medieval 2026

## Description
Modele HTR fine-tune sur CREMMA Medieval (CC-BY 4.0). Vieux/moyen francais XIe-XVIIe s.

## Donnees
| Split | Manuscrits | Lignes | SHA-256 |
|---|---|---|---|
| Train | 10 | 6891 | de26aaa226edad415b37e02a6e3e68a5f001968a3ec69daddb3c47b69d9767d8 |
| Val   | 2  | 2733 | b7891b9a73b9704e37623b7dfec43677a7a94da2f58aa94557254dfd1a03a4c5 |
| Test  | 2  |  251 | 86f5f48b7628128b4e86bc5684b13f7409972bf3d7dc39291d97b03d25f62949 |

## Architecture
| Composant | Detail |
|---|---|
| Modele base | microsoft/trocr-base-handwritten |
| Fine-tuning | LoRA r=8, alpha=32, dropout=0.1 |
| Optimiseur  | AdamW lr=5e-5, patience=3, fp16, gradient_checkpointing |

## Performances (test set scelle)
| Metrique | Valeur | IC 95% |
|---|---|---|
| CER global | 0.1388 (13.9%) | [0.1231, 0.1566] |
| CER abrev. | 0.1639 | - |
| CER normal | 0.1331 | - |
| delta_CER  | 0.0308 | - |
| WER global | 0.4728 (47.3%) | - |
| needs_review | 1.2% | - |

## Conclusion
Le CER global est 0.139. Sur les lignes contenant des abréviations médiévales, le CER est plus élevé (0.164) par rapport aux lignes ordinaires (0.133) (IC 95% bootstrap : [-0.006, 0.070]). Les caractères d'abréviation les plus impactants sont : «ͤ», «ꝯ», «ͬ». Les abréviations représentent 18.7% du corpus. Ces résultats indiquent que les abréviations impactent modérément les performances HTR sur le corpus CREMMA.
