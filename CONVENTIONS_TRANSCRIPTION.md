# Conventions de transcription

Ce document décrit les choix éditoriaux appliqués aux transcriptions du projet. Le corpus retenu est CREMMA Médiéval, et l'axe d'analyse porte sur l'impact des abréviations sur les performances HTR.

---

## Niveau de transcription retenu

Le niveau retenu est une transcription semi-diplomatique harmonisée.

Objectifs :

- conserver les graphies historiques utiles à l'HTR ;
- éviter une modernisation trop forte du texte ;
- garder une sortie exploitable pour le volet NLP ;
- documenter toute transformation appliquée aux données sources.

Les données brutes restent inchangées dans `data/raw/`. Toute normalisation doit être appliquée dans `data/processed/` avec un script reproductible.

---

## Règles principales

| Élément | Convention retenue |
|---|---|
| Abréviations | Conserver les signes d'abréviation présents dans CREMMA ; ne pas les développer ni les moderniser avant l'entraînement |
| `u/v` et `i/j` | Conserver la convention du corpus source avant audit ; ne pas normaliser sans transformation explicite |
| Ponctuation | Conserver la ponctuation transcrite ; ne pas moderniser |
| Casse | Conserver la casse source |
| Lacunes | Conserver les marqueurs sources ; ne pas supprimer silencieusement |
| Lecture incertaine | Conserver les marqueurs sources et ajouter `needs_review` si nécessaire |
| Espaces | Ne pas corriger manuellement avant audit |
| Unicode | Conserver les caractères médiévaux utiles, puis normaliser les données traitées en NFC avant entraînement |

---

## Caractères illisibles et lacunes

| Cas | Convention cible |
|---|---|
| Caractère isolé illisible | Utiliser `[?]` dans les données traitées si le caractère ne peut pas être identifié |
| Mot illisible | Utiliser `[ill.]` dans les données traitées |
| Lacune matérielle | Conserver le marqueur source ou mapper vers `[---]` dans les données traitées |
| Lecture incertaine | Conserver l'information sous la forme `[mot?]` si un mot est proposé |
| Ligne très dégradée | Marquer `needs_review=true` et exclure de l'entraînement si la transcription n'est pas exploitable |

Les marqueurs sources ne doivent jamais être supprimés silencieusement. Quand un marqueur est harmonisé, la transformation doit être documentée dans le script de préparation et dans les métadonnées de la ligne.

---

## Gestion de `needs_review`

Le champ `needs_review` est un booléen utilisé pour signaler les lignes qui nécessitent une relecture humaine avant usage NLP ou avant intégration dans le jeu final.

`needs_review=true` doit être appliqué dans les cas suivants :

- présence de `[?]`, `[ill.]`, `[---]` ou `[mot?]` ;
- ligne vide après nettoyage ;
- ligne très courte, par exemple moins de 3 caractères utiles ;
- transcription source explicitement incertaine ;
- langue ou siècle non identifié ;
- polygone manquant, invalide ou hors limites de l'image ;
- score de confiance HTR inférieur au seuil choisi sur validation ;
- désaccord important entre deux modèles si TrOCR et Kraken sont comparés.

Le seuil de confiance ne doit pas être choisi sur le test set. Il sera calibré sur la validation. En attendant l'entraînement, le seuil provisoire est documenté comme `à calibrer`.

Chaque ligne marquée `needs_review=true` doit aussi contenir un champ `needs_review_reasons`, par exemple :

```json
["illegible_text", "low_confidence", "invalid_polygon"]
```

---

## Segmentation et polygones

Chaque ligne transcrite doit être reliée à une zone spatiale de l'image source. Cette information est obligatoire pour permettre la vérification visuelle, l'export PAGE XML et la réutilisation dans des outils comme Kraken ou eScriptorium.

Convention cible :

- un `line_id` unique par ligne ;
- un `image_id` permettant de retrouver l'image source ;
- un `polygon` associé à chaque ligne, ou un lien explicite vers le fichier PAGE XML contenant ce polygone ;
- coordonnées exprimées en pixels ;
- origine du repère en haut à gauche de l'image ;
- points du polygone stockés sous forme `[[x1, y1], [x2, y2], ...]` ;
- polygones conservés dans l'ordre de lecture ;
- aucune coordonnée négative ;
- aucune coordonnée supérieure à la largeur ou à la hauteur de l'image.

Exemple JSON cible :

```json
{
  "line_id": "page_001_line_014",
  "image_id": "page_001",
  "text": "exemple de ligne transcrite",
  "polygon": [[120, 340], [860, 342], [858, 392], [118, 390]],
  "polygon_format": "xy_pixels_top_left_origin",
  "needs_review": false
}
```

Si une ligne possède une transcription mais aucun polygone exploitable, elle doit être conservée dans l'audit, mais marquée `needs_review=true`.

---

## Unicode et encodage

Les fichiers texte produits par le pipeline doivent être encodés en UTF-8.

Règles Unicode :

- conserver les caractères médiévaux présents dans les sources si leur usage est cohérent ;
- ne pas convertir les caractères accentués en ASCII ;
- ne pas supprimer les signes diacritiques ;
- normaliser les textes traités en NFC avec `unicodedata.normalize("NFC", text)` ;
- appliquer la même normalisation à train, validation et test ;
- journaliser les caractères rares ou inconnus pendant l'audit ;
- ne remplacer un caractère par `[?]` que s'il est réellement illisible ou corrompu.

Les données brutes ne sont pas modifiées. La normalisation Unicode s'applique uniquement aux données traitées et doit être reproductible.

---

## Repérage des abréviations

Comme l'axe d'analyse porte sur les abréviations, chaque ligne doit recevoir des métadonnées permettant de la classer.

Champs à produire :

- `has_abbreviation` : booléen indiquant si la ligne contient au moins une abréviation ;
- `abbreviation_count` : nombre de signes ou marqueurs d'abréviation détectés ;
- `abbreviation_density` : `abbreviation_count / nombre_caracteres` ;
- `abbreviation_markers` : liste des signes détectés dans la ligne.

Les abréviations ne doivent pas être remplacées par leur forme développée dans les données d'entraînement principales. Si une version normalisée est testée plus tard, elle devra être traitée comme une variante expérimentale séparée et documentée dans le journal d'expériences.

Exemple de métadonnées :

```json
{
  "line_id": "page_001_line_014",
  "has_abbreviation": true,
  "abbreviation_count": 2,
  "abbreviation_density": 0.041,
  "abbreviation_markers": ["ꝑ", "̃"]
}
```

---

## Métadonnées obligatoires par ligne

Chaque ligne préparée pour l'entraînement doit inclure :

- `source_corpus`
- `manuscript_id`
- `image_id`
- `line_id`
- `text`
- `language`
- `century`
- `split`
- `polygon` ou lien vers le fichier PAGE XML
- `polygon_format`
- `needs_review`
- `needs_review_reasons`
- `has_abbreviation`
- `abbreviation_count`
- `abbreviation_density`
- `abbreviation_markers`
- `transcription_policy`

---

## Points à vérifier pendant l'audit

- Quelle proportion de lignes contient des signes d'abréviation ?
- Quels signes d'abréviation sont les plus fréquents dans CREMMA ?
- La densité d'abréviations varie-t-elle selon les manuscrits ou les siècles ?
- Les graphies `u/v` et `i/j` sont-elles conservées dans les transcriptions CREMMA ?
- Les lignes très courtes ou dégradées doivent-elles être exclues ou marquées `needs_review` ?
- Toutes les lignes possèdent-elles un polygone valide ?
- Quels caractères Unicode rares apparaissent dans CREMMA ?

---

## Décision à confirmer après audit

L'audit CREMMA doit confirmer la stratégie d'analyse des abréviations. Deux variantes pourront ensuite être comparées sur la validation :

- Variante A : entraînement avec les abréviations conservées telles qu'elles apparaissent dans CREMMA.
- Variante B : entraînement avec une normalisation expérimentale de certains signes d'abréviation, uniquement si cette normalisation est justifiée et documentée.

Le test set restera scellé et ne servira pas à choisir ces variantes.
