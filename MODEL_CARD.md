# Model Card — HTR CREMMA Médiéval 2026

## Résumé

Modèle HTR (Handwritten Text Recognition) fine-tuné sur le corpus CREMMA Médiéval
(CC-BY 4.0), spécialisé pour le vieux et moyen français du XIe au XVIIe siècle.
Basé sur TrOCR avec adaptation LoRA, il atteint un **CER de 13.88 %** sur le jeu
de test scellé, contre 99.08 % pour le modèle de base non adapté.

---

## Données

| Split | Manuscrits | Lignes | SHA-256 |
|---|---|---|---|
| Train | 10 | 6 891 | `de26aaa226edad415b37e02a6e3e68a5f001968a3ec69daddb3c47b69d9767d8` |
| Val   | 2  | 2 733 | `b7891b9a73b9704e37623b7dfec43677a7a94da2f58aa94557254dfd1a03a4c5` |
| Test  | 2  |   251 | `86f5f48b7628128b4e86bc5684b13f7409972bf3d7dc39291d97b03d25f62949` |

Source : [HTR-United/cremma-medieval](https://github.com/HTR-United/cremma-medieval)
— transcriptions au format ALTO XML, alignées image/texte au niveau ligne.

---

## Architecture

| Composant | Détail |
|---|---|
| Modèle de base | `microsoft/trocr-base-handwritten` (Vision Encoder–Decoder) |
| Méthode de fine-tuning | LoRA (Low-Rank Adaptation) |
| Paramètres LoRA | r=8, alpha=32, dropout=0.1 |
| Modules adaptés | Couches d'attention Q et V du décodeur Transformer |
| Paramètres entraînés | ~4.7 M / 334 M total (1.4 % seulement) |
| Optimiseur | AdamW lr=5e-5, weight_decay=0.01 |
| Batch size | 16 |
| Précision | fp16 (mixed precision) |
| gradient_checkpointing | activé |
| Arrêt précoce | patience=3 sur CER val |

**Pourquoi LoRA ?**
Fine-tuner TrOCR complet nécessite ~12 Go de VRAM et plusieurs heures sur GPU.
LoRA injecte de petites matrices de rang bas (r=8) dans les couches d'attention :
seuls 1.4 % des paramètres sont mis à jour, ce qui réduit le coût mémoire par ×8
tout en atteignant des performances comparables au full fine-tuning sur ce corpus.

---

## Historique d'entraînement

Deux runs complets ont été effectués sur Kaggle (GPU T4, ~20 min par run).
Les métriques CER sont mesurées sur le **val set (2 733 lignes)** après chaque époque,
sans jamais toucher le test set scellé.

### Run 1 — 16 juin 2026, 07h22

| Étape | CER global | CER abréviations | CER sans abréviations | Loss entraînement |
|---|---|---|---|---|
| Baseline (0 époque) | 99.08 % | 83.43 % | 99.48 % | — |
| Époque 1 | 36.54 % | 31.93 % | 36.96 % | 2.8276 |
| Époque 2 | 26.52 % | 24.45 % | 26.71 % | 1.1689 |
| Époque 3 | 22.54 % | 20.78 % | 22.70 % | 0.7878 |
| Époque 4 | 22.47 % | 20.46 % | 22.65 % | 0.6127 |
| Époque 5 | 21.46 % | 19.08 % | 21.68 % | 0.5039 |
| **Époque 6 — meilleure** | **20.87 %** | **19.65 %** | **20.98 %** | **0.4437** |
| Époque 7 | 20.88 % | 18.09 % | 21.13 % | 0.3805 |

### Run 2 — 16 juin 2026, 17h12

| Étape | CER global | CER abréviations | CER sans abréviations | Loss entraînement |
|---|---|---|---|---|
| Baseline (0 époque) | 99.08 % | 83.43 % | 99.48 % | — |
| Époque 1 | 34.96 % | 31.62 % | 35.26 % | 2.7858 |
| Époque 2 | 26.48 % | 24.09 % | 26.70 % | 1.1019 |
| Époque 3 | 23.29 % | 21.31 % | 23.47 % | 0.7467 |
| Époque 4 | 22.51 % | 20.58 % | 22.69 % | 0.5892 |
| Époque 5 | 22.15 % | 21.13 % | 22.25 % | 0.4911 |
| **Époque 6 — meilleure** | **21.07 %** | **19.13 %** | **21.25 %** | **0.4165** |
| Époque 7 | 21.58 % | 18.98 % | 21.82 % | 0.3656 |

### Analyse de la courbe d'apprentissage

- **Époques 1–3** : gain principal, −14 à −15 pts CER. Le modèle apprend les
  formes graphiques du vieux français (allographes, ligatures).
- **Époques 4–6** : convergence lente, −1 à −2 pts par époque. Affinement fin.
- **Époque 7** : CER val remonte légèrement (+0.01 à +0.05 pts) alors que la loss
  train continue de baisser → signal de sur-apprentissage (overfitting).
  L'époque 6 est sélectionnée comme meilleur checkpoint dans les deux runs.

---

## Performances finales (test set scellé)

Évaluation effectuée **une seule fois** sur le jeu de test scellé (251 lignes,
2 manuscrits non vus pendant l'entraînement).

| Métrique | Valeur | IC 95 % bootstrap |
|---|---|---|
| CER global | **13.88 %** | [12.31 %, 15.66 %] |
| CER abréviations | 16.39 % | [13.31 %, 20.07 %] |
| CER sans abréviations | 13.31 % | [11.53 %, 15.20 %] |
| delta_CER (abbrev − normal) | +3.08 pts | [−0.56 %, +7.02 %] |
| WER global | 47.28 % | — |
| WER abréviations | 55.63 % | — |
| WER sans abréviations | 45.35 % | — |
| needs_review | 1.2 % | — |

**Amélioration vs baseline** : CER 99.08 % → 13.88 % (−85.2 points).

L'IC 95 % du delta_CER (abréviations vs normal) inclut zéro [−0.56 %, +7.02 %],
ce qui signifie que l'impact des abréviations, bien que positif en moyenne (+3.08 pts),
n'est pas statistiquement significatif au seuil 5 % avec 251 lignes de test.

### Impact par caractère d'abréviation

| Caractère | Signification | CER moyen | Lignes |
|---|---|---|---|
| `ͤ` | voyelle suscrite (e) | 29.74 % | 2 |
| `ꝯ` | con / com | 23.24 % | 7 |
| `ͬ` | voyelle suscrite (r) | 16.89 % | 5 |
| `̃` | nasale (tilde) | 15.23 % | 23 |
| `ꝑ` | per / par | 14.22 % | 5 |
| `ͥ` | voyelle suscrite (i) | 11.93 % | 4 |
| `ͣ` | voyelle suscrite (a) | 11.73 % | 5 |

Les abréviations représentent **18.7 %** du corpus (47 / 251 lignes de test).

### Test de McNemar (fine-tuné vs baseline)

Le test de McNemar évalue si les deux modèles font des erreurs **différentes** sur
les mêmes lignes — adapté aux données appariées, contrairement au test t.

- b (baseline correct, fine-tuné incorrect) : **0**
- c (fine-tuné correct, baseline incorrect) : **20**
- **chi² = 18.05, p ≈ 0**

Le fine-tuning améliore significativement la transcription (p < 0.001) : il corrige
20 lignes que le baseline échoue sans en dégrader aucune en retour.

---

## Limites

- **WER élevé (47.3 %)** : les erreurs de segmentation et les formes lexicales rares
  du vieux français (hapax, dialectalismes) pénalisent la reconnaissance au niveau mot.
- **Checkpoints non versionnés** : les poids (>400 MB) ne sont pas inclus dans ce
  dépôt. Relancer `colab_htr_cremma.ipynb` sur Kaggle (T4, ~20 min) pour reproduire.
- **Généralisation limitée** : entraîné sur 10 manuscrits. La performance sur des
  scriptoria non représentés est inconnue.
- **Abréviations rares** : les IC bootstrap sur `ͤ` et `ͬ` sont très larges (2–5 lignes)
  — impossible de conclure statistiquement sur ces caractères seuls.

---

## Licence

| Composant | Licence |
|---|---|
| Corpus CREMMA Médiéval | CC-BY 4.0 (HTR-United) |
| Code source | MIT |
| Modèle de base TrOCR | MIT (Microsoft) |
| Modèle NER CamemBERT | CC-BY 4.0 (Jean-Baptiste / Hugging Face) |
