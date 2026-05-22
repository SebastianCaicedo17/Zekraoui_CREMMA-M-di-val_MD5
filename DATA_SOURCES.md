# Sources de données

Ce fichier documente le corpus unique retenu pour le projet : CREMMA Médiéval. Le projet ne compare pas plusieurs corpus ; il analyse un axe précis à l'intérieur de CREMMA, centré sur l'impact des abréviations sur les performances HTR.

---

## Corpus retenu

| Corpus | Source | Période | Manuscrits | Pages | Lignes | Format | Licence | URL |
|---|---|---|---:|---:|---:|---|---|---|
| CREMMA Médiéval | HTR-United / CREMMALab | XIIe-XVe siècle, principalement XIIIe-XVe | 14 | 279 | 22 843 | ALTO XML | CC-BY 4.0 | https://github.com/HTR-United/cremma-medieval |

---

## Description du corpus

CREMMA Médiéval est un corpus de transcriptions destiné à l'entraînement de modèles HTR pour manuscrits médiévaux. Il regroupe des manuscrits d'ancien et de moyen français, produits avec eScriptorium et Kraken, puis diffusés au format ALTO XML via HTR-United.

Le corpus est particulièrement adapté à notre problématique, car ses transcriptions suivent une approche graphémique : le texte cherche à garder un lien direct entre le signe visible dans l'image et le signe transcrit. Les abréviations sont donc conservées, ce qui permet d'étudier leur impact sur les erreurs HTR.

**Format technique :** ALTO XML v4, avec les transcriptions dans les attributs `<String CONTENT="...">` des lignes `<TextLine>`.

**Attribution :** Ariane Pinche et contributeurs CREMMALab / HTR-United.

---

## Détail par manuscrit

| Manuscrit | Pages | Lignes | Moy. car/ligne | Institution |
|---|---:|---:|---:|---|
| bnf_arsenal_3516-imageDuMonde | 10 | 2 023 | 25,7 | BnF Arsenal |
| bnf_fr_13496-saintJerome | 1 | 161 | 29,4 | BnF |
| bnf_fr_17229-saintLambert | 1 | 164 | 27,4 | BnF |
| bnf_fr_1728 | 10 | 622 | 26,0 | BnF |
| bnf_fr_22549-septSages | 18 | 2 661 | 25,1 | BnF |
| bnf_fr_24428-bestiaire | 20 | 1 328 | 26,4 | BnF |
| bnf_fr_411-saintLambert | 1 | 153 | 33,8 | BnF |
| bnf_fr_412-wauchier | 66 | 6 323 | 38,0 | BnF |
| bnf_fr_844-manuscritDuRoi | 18 | 1 474 | 20,8 | BnF |
| bodmer_168-otinel | 23 | 1 975 | 33,2 | Fondation Martin Bodmer |
| kbr_9232-examensMoraux | 16 | 1 358 | 33,4 | KBR Bruxelles |
| pennsylvania_660-pelerinageMademoiselleSapience | 11 | 320 | 45,5 | University of Pennsylvania |
| pennsylvania_codex_909-eneide | 32 | 2 509 | 35,9 | University of Pennsylvania |
| vaticane_reg_lat_1616-otinel | 52 | 1 772 | 32,7 | Vatican |

Cette répartition montre que le corpus est déséquilibré : certains manuscrits sont très représentés, notamment `bnf_fr_412-wauchier`, tandis que d'autres ne contiennent qu'une seule page. Les analyses de performance devront donc être faites à la fois globalement et par manuscrit.

---

## Audit préliminaire des abréviations

Un audit local a été réalisé sur les fichiers ALTO XML du dépôt CREMMA Médiéval, en excluant les fichiers `*.chocomufin.xml`.

| Indicateur | Valeur |
|---|---:|
| Fichiers ALTO XML analysés | 279 |
| Lignes non vides extraites | 22 843 |
| Caractères extraits, espaces inclus | 728 440 |
| Lignes avec abréviation stricte | 8 845, soit 38,72 % |
| Nombre de marqueurs stricts | 11 736 |
| Densité stricte | 16,11 marqueurs / 1 000 caractères |
| Lignes avec abréviation au sens large | 10 427, soit 45,65 % |
| Nombre de marqueurs larges | 15 200 |
| Densité large | 20,87 marqueurs / 1 000 caractères |

Détection stricte : signes explicitement abréviatifs ou médiévaux, par exemple `⁊`, `ꝑ`, `ꝯ`, `ꝰ`, lettres suscrites, tildes/macrons combinants et signes privés documentés par CREMMA comme `\uf158`.

Détection large : détection stricte + caractères latins précomposés avec tilde ou macron, par exemple `ẽ`, `õ`, `ã`, `ũ`, `ē`, `ō`.

Principaux marqueurs stricts observés :

| Marqueur | Occurrences | Interprétation |
|---|---:|---|
| `\uf158` | 2 474 | signe abréviatif privé, documenté comme petit `et` barré dans la table CREMMA |
| `̃` | 1 478 | tilde combinant, souvent lié aux nasalités abrégées |
| `⁊` | 1 428 | signe tironien `et` |
| `ꝑ` | 741 | `p` barré |
| `ꝯ` | 711 | signe `con` |
| `͛` | 665 | zigzag combinant |
| `ͣ` | 630 | lettre `a` suscrite |
| `̄` | 571 | macron combinant |
| `ͥ` | 450 | lettre `i` suscrite |
| `ꝰ` | 290 | signe `us` |

La répartition n'est pas homogène selon les manuscrits. Par exemple, `bnf_arsenal_3516-imageDuMonde` présente environ 71,5 % de lignes avec abréviation stricte, tandis que `bnf_fr_844-manuscritDuRoi` en présente environ 8,7 %. Cette variation est intéressante pour l'axe d'analyse : elle permet d'étudier si la densité d'abréviations explique une partie des écarts de CER entre manuscrits.

---

## Axe d'analyse retenu

**Reconnaissance automatique de manuscrits médiévaux : impact des abréviations sur les performances HTR.**

Problématique :

> Dans quelle mesure la présence et la densité des abréviations dans CREMMA Médiéval influencent-elles les performances d'un modèle HTR, mesurées par le CER et le WER ?

Variables à extraire :

- présence ou absence d'abréviation dans chaque ligne ;
- nombre d'abréviations par ligne ;
- densité d'abréviations : `nombre_abreviations / nombre_caracteres` ;
- longueur de ligne ;
- manuscrit source ;
- siècle si l'information est exploitable ;
- CER et WER par ligne après prédiction HTR.

Comparaisons prévues :

- lignes sans abréviation vs lignes avec abréviation ;
- faible densité vs forte densité d'abréviations ;
- analyse par manuscrit si le volume est suffisant ;
- analyse qualitative des erreurs sur les caractères médiévaux et signes abréviatifs.

---

## Règles anti-fuite de données

- Faire le split au niveau manuscrit ou folio, jamais au niveau ligne isolée si cela mélange des pages très proches.
- Ne jamais mélanger des lignes d'un même manuscrit entre train, validation et test lorsque c'est possible.
- Calculer un hash SHA-256 pour les listes train, validation et test.
- Ne pas ouvrir le test set pendant le développement.
- Choisir les seuils liés aux abréviations et à `needs_review` uniquement sur la validation.

---

## Risques identifiés

| Risque | Impact | Mesure de réduction |
|---|---|---|
| Distribution déséquilibrée des manuscrits | Résultats trop dépendants d'un manuscrit dominant | Split et analyse par manuscrit |
| Abréviations rares ou très spécifiques | Forte variance du CER par ligne | Regrouper les lignes par densité d'abréviations |
| Espaces non homogènes | Hausse artificielle du WER | Interpréter le WER avec prudence et privilégier le CER |
| Caractères médiévaux Unicode | Erreurs artificielles si l'encodage est instable | Normalisation Unicode documentée |
| Lignes très courtes | CER instable | Marquer `needs_review` ou analyser séparément |
| Manuscrits très peu représentés | Comparaison par manuscrit fragile | Regrouper ou commenter les limites |

---

## Modèles pré-entraînés envisagés

| Modèle | Source | Licence | Usage |
|---|---|---|---|
| `microsoft/trocr-base-handwritten` | HuggingFace | MIT | Baseline et fine-tuning TrOCR |
| Kraken BLLA | kraken.re | Apache 2.0 | Segmentation lignes |
| Modèles Kraken CREMMA | Releases CREMMA | CC-BY 4.0 | Baseline spécialisée possible |

---

## Procédure de vérification des licences

1. Consulter la fiche GitHub du dataset CREMMA Médiéval.
2. Vérifier que la licence est compatible avec un usage de recherche non commercial.
3. Télécharger les données dans `data/raw/`, qui reste ignoré par Git.
4. Enregistrer les hachages SHA-256 des fichiers utilisés.
5. Citer CREMMA Médiéval et ses auteurs dans l'article final.
