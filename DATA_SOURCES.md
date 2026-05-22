# Sources de données

Tous les corpus utilisés dans ce projet sont sous licence libre (CC-BY, CC-BY-SA ou domaine public), conformément aux contraintes du projet MD5.

---

## Corpus retenus

| Corpus | Source | Période | Manuscrits | Pages | Lignes | Licence | URL |
|---|---|---|---|---|---|---|---|
| CREMMA Médiéval | HTR-United / Inria | XIIIe–XVe s. | 14 | 279 | 22 843 | CC-BY 4.0 | https://github.com/HTR-United/cremma-medieval |

### Détail par manuscrit — CREMMA Médiéval

| Manuscrit | Pages | Lignes | Moy. car/ligne | Institution |
|---|---|---|---|---|
| bnf_arsenal_3516-imageDuMonde | 10 | 2 023 | 25,7 | BnF Arsenal |
| bnf_fr_13496-saintJerome | 1 | 161 | 29,4 | BnF |
| bnf_fr_17229-saintLambert | 1 | 164 | 27,4 | BnF |
| bnf_fr_1728 | 10 | 622 | 26,0 | BnF |
| bnf_fr_22549-septSages | 18 | 2 661 | 25,1 | BnF |
| bnf_fr_24428-bestiaire | 20 | 1 328 | 26,4 | BnF |
| bnf_fr_411-saintLambert | 1 | 153 | 33,8 | BnF |
| bnf_fr_412-wauchier | 66 | 6 323 | 38,0 | BnF |
| bnf_fr_844-manuscritDuRoi | 18 | 1 474 | 20,8 | BnF |
| bodmer_168-otinel | 23 | 1 975 | 33,2 | Bodmer |
| kbr_9232-examensMoraux | 16 | 1 358 | 33,4 | KBR Bruxelles |
| pennsylvania_660-pelerinageMademoiselleSapience | 11 | 320 | 45,5 | U. Pennsylvania |
| pennsylvania_codex_909-eneide | 32 | 2 509 | 35,9 | U. Pennsylvania |
| vaticane_reg_lat_1616-otinel | 52 | 1 772 | 32,7 | Vatican |

**Format** : ALTO XML v4 (balise `<String CONTENT="...">` dans `<TextLine>`)  
**Attribution** : Thibault Clérice et al., HTR-United (2021–2024)

---

## Corpus explorés mais non retenus

| Corpus | Raison d'exclusion |
|---|---|
| CATMuS Medieval | Périmètre très large (160 k+ lignes) — CREMMA suffisant pour ce projet |
| GalliCorpora | Redondance partielle avec CREMMA BnF, pas de valeur ajoutée claire |

---

## Modèles pré-entraînés

| Modèle | Source | Licence | Usage |
|---|---|---|---|
| `microsoft/trocr-base-handwritten` | HuggingFace | MIT | Base fine-tuning TrOCR |
| Kraken BLLA | kraken.re | Apache 2.0 | Segmentation lignes |
| `facebook/sam-vit-base` | Meta / HuggingFace | Apache 2.0 | Segmentation layout |

---

## Procédure de vérification des licences

1. Consulter la fiche du dataset sur HuggingFace ou Zenodo.
2. Vérifier que la licence est CC-BY, CC-BY-SA ou domaine public.
3. Documenter l'URL de la licence dans ce fichier.
4. Ne télécharger les données qu'après validation.

---

*Toute nouvelle source ajoutée en cours de projet doit être documentée ici avant utilisation.*