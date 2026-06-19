"""Module NER — Reconnaissance d'entités nommées sur transcriptions HTR médiévales.

Pipeline :
  1. charger_modele_ner  -> pipeline HuggingFace CamemBERT-NER
  2. annoter_entites     -> entités d'une ligne nettoyée
  3. annoter_dataset     -> nlp/entites.json
  4. agregger_entites    -> statistiques par type et par manuscrit

Le texte est normalisé (mode 'normalise' de src.nlp) avant annotation pour
maximiser la détection par un modèle entraîné sur du français moderne.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from src.nlp import NLP_DIR, charger_regles_nettoyage, nettoyer_ligne

# Modèle recommandé : CamemBERT fine-tuné sur WikiNER français
_MODEL_NAME = "Jean-Baptiste/camembert-ner"

# Types d'entités reconnus
TYPES_ENTITES = {"PER", "LOC", "ORG", "MISC"}

# Seuil de confiance minimal pour conserver une entité
_SCORE_MIN = 0.80

# Mots grammaticaux médiévaux fréquemment confondus avec des entités
_FAUX_POSITIFS = {
    "si", "se", "ce", "cil", "cel", "cist", "ceste", "les", "des", "del",
    "dou", "au", "es", "que", "qui", "ou", "ou", "en", "de", "le", "la",
    "li", "sire", "dame", "seigneur", "conme", "comme", "et", "est", "fu",
    "a", "e", "i", "u", "q", "s", "b", "c", "d", "f", "g", "h", "j",
    "k", "l", "m", "n", "o", "p", "r", "t", "v", "w", "x", "y", "z",
    "ne", "ni", "or", "mais", "car", "car", "tant", "lors", "puis",
    "bien", "moult", "tres", "tout", "toute", "tous", "grant", "petit",
    "sa", "son", "ses", "mon", "ton", "lor", "nostre", "vostre",
}


# ─── Chargement du modèle ────────────────────────────────────────────────────

def charger_modele_ner(model_name: str = _MODEL_NAME):
    """Charge le pipeline NER HuggingFace.

    Télécharge le modèle au premier appel (~420 MB), puis utilise le cache
    local (~/.cache/huggingface/). Aucune clé API requise.

    Args:
        model_name: Identifiant HuggingFace du modèle NER.
                    Par défaut ``Jean-Baptiste/camembert-ner``.

    Returns:
        Pipeline HuggingFace prêt à l'emploi.

    Example:
        >>> pipeline_ner = charger_modele_ner()
    """
    from transformers import pipeline

    print(f"[ner] Chargement du modele {model_name} ...")
    pipe = pipeline(
        "ner",
        model=model_name,
        aggregation_strategy="simple",
    )
    print(f"[ner] Modele charge.")
    return pipe


# ─── Annotation d'une ligne ──────────────────────────────────────────────────

def annoter_entites(texte: str, pipeline_ner) -> list[dict]:
    """Détecte les entités nommées dans une ligne de texte normalisée.

    Args:
        texte: Texte nettoyé en mode ``normalise`` (abréviations développées,
               graphie u/v normalisée).
        pipeline_ner: Pipeline HuggingFace NER chargé par :func:`charger_modele_ner`.

    Returns:
        Liste de dicts ``{"mot", "type", "score", "debut", "fin"}``.
        Liste vide si le texte est vide ou ne contient aucune entité.

    Example:
        >>> pipeline_ner = charger_modele_ner()
        >>> annoter_entites("le roi Philippe alla vers Paris", pipeline_ner)
        [{'mot': 'Philippe', 'type': 'PER', 'score': 0.97, ...},
         {'mot': 'Paris', 'type': 'LOC', 'score': 0.99, ...}]
    """
    if not texte or not texte.strip():
        return []

    try:
        resultats = pipeline_ner(texte)
    except Exception:
        return []

    entites = []
    for ent in resultats:
        type_ent = ent.get("entity_group", ent.get("entity", "")).upper()
        # Normaliser les préfixes B-/I- éventuels
        for prefix in ("B-", "I-"):
            if type_ent.startswith(prefix):
                type_ent = type_ent[len(prefix):]
        if type_ent not in TYPES_ENTITES:
            continue

        mot = ent.get("word", "").strip()
        score = float(ent.get("score", 0.0))

        # Filtres qualité : écarter fragments, mots trop courts, faux positifs
        if len(mot) <= 2:
            continue
        if score < _SCORE_MIN:
            continue
        if mot.lower() in _FAUX_POSITIFS:
            continue

        entites.append({
            "mot": mot,
            "type": type_ent,
            "score": round(score, 4),
            "debut": ent.get("start", 0),
            "fin": ent.get("end", 0),
        })
    return entites


# ─── Annotation du dataset ───────────────────────────────────────────────────

def annoter_dataset(
    entrees: list[dict],
    pipeline_ner,
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    out_path: Path = NLP_DIR / "entites.json",
    model_name: str = _MODEL_NAME,
) -> dict:
    """Annote toutes les lignes du dataset et produit les statistiques NER.

    Pour chaque entrée :
    1. Nettoie la transcription en mode ``normalise`` (développe abréviations,
       normalise u/v, supprime balises éditoriales).
    2. Applique le pipeline NER.
    3. Agrège les entités.

    Args:
        entrees: Liste de dicts avec au moins ``line_id``, ``transcription``,
                 ``has_abbreviation``.
        pipeline_ner: Pipeline NER chargé.
        regles_diplo: Règles diplomatique issues de :func:`~src.nlp.charger_regles_nettoyage`.
        regles_norm: Règles normalise.
        out_path: Destination pour entites.json.
        model_name: Nom du modèle (pour les métadonnées).

    Returns:
        Dict avec ``meta``, ``statistiques``, ``repartition_manuscrits``
        et ``entites_par_ligne``.

    Example:
        >>> items = json.loads(Path("dataset_nlp/val.json").read_text())
        >>> diplo, norm = charger_regles_nettoyage()
        >>> pipe = charger_modele_ner()
        >>> rapport = annoter_dataset(items, pipe, diplo, norm)
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)

    entites_par_ligne: list[dict] = []
    nb_lignes = 0

    for item in entrees:
        texte_brut = item.get("transcription", "")
        texte_net = nettoyer_ligne(texte_brut, regles_diplo, regles_norm, mode="normalise")
        entites = annoter_entites(texte_net, pipeline_ner)

        manuscrit = item.get("manuscrit", item.get("image_id", "inconnu"))
        if "_" in manuscrit:
            manuscrit = manuscrit.split("_")[0]

        entites_par_ligne.append({
            "line_id": item.get("line_id", str(nb_lignes)),
            "transcription_originale": texte_brut,
            "transcription_nettoyee": texte_net,
            "manuscrit": manuscrit,
            "has_abbreviation": item.get("has_abbreviation", False),
            "entites": entites,
        })
        nb_lignes += 1

        if nb_lignes % 500 == 0:
            print(f"[ner] {nb_lignes}/{len(entrees)} lignes annotees...")

    stats = agregger_entites(entites_par_ligne)

    result = {
        "meta": {
            "modele": model_name,
            "mode_nettoyage": "normalise",
            "nb_lignes_annotees": nb_lignes,
        },
        "statistiques": stats["par_type"],
        "repartition_manuscrits": stats["par_manuscrit"],
        "entites_par_ligne": entites_par_ligne,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ner] {nb_lignes} lignes annotees -> {out_path}")
    for type_ent, info in stats["par_type"].items():
        top3 = ", ".join(e["entite"] for e in info["top20"][:3])
        print(f"  {type_ent}: {info['nb_total']} entites  (top: {top3})")
    return result


# ─── Agrégation ──────────────────────────────────────────────────────────────

def agregger_entites(entites_par_ligne: list[dict]) -> dict:
    """Agrège les entités par type et par manuscrit.

    Args:
        entites_par_ligne: Liste produite par :func:`annoter_dataset`.

    Returns:
        Dict ``{"par_type": {...}, "par_manuscrit": {...}}``.

    Example:
        >>> stats = agregger_entites(entites_par_ligne)
        >>> print(stats["par_type"]["PER"]["nb_total"])
    """
    compteur_type: dict[str, Counter] = {t: Counter() for t in TYPES_ENTITES}
    compteur_manuscrit: dict[str, dict[str, int]] = defaultdict(lambda: {t: 0 for t in TYPES_ENTITES})

    for ligne in entites_par_ligne:
        manuscrit = ligne.get("manuscrit", "inconnu")
        for ent in ligne.get("entites", []):
            type_ent = ent["type"]
            mot = ent["mot"].lower().strip()
            if type_ent in compteur_type and mot:
                compteur_type[type_ent][mot] += 1
                compteur_manuscrit[manuscrit][type_ent] += 1

    par_type: dict[str, dict] = {}
    for type_ent in TYPES_ENTITES:
        ctr = compteur_type[type_ent]
        top20 = [{"entite": mot, "freq": freq} for mot, freq in ctr.most_common(20)]
        par_type[type_ent] = {
            "nb_total": sum(ctr.values()),
            "nb_formes_uniques": len(ctr),
            "top20": top20,
        }

    par_manuscrit = {ms: dict(counts) for ms, counts in compteur_manuscrit.items()}

    return {"par_type": par_type, "par_manuscrit": par_manuscrit}


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import fixer_seeds

    fixer_seeds(42)

    BASE = Path("dataset_nlp")
    predictions_path = BASE / "val.json"

    if not predictions_path.exists():
        print(f"[ner] {predictions_path} absent. Lance d'abord python -m src.nlp")
        raise SystemExit(1)

    print("\n=== NER — Chargement des regles de nettoyage ===")
    regles_diplo, regles_norm = charger_regles_nettoyage()

    print("\n=== NER — Chargement du modele ===")
    pipeline_ner = charger_modele_ner()

    print("\n=== NER — Annotation du val set ===")
    entrees = json.loads(predictions_path.read_text(encoding="utf-8"))
    rapport = annoter_dataset(entrees, pipeline_ner, regles_diplo, regles_norm)

    print("\n=== NER termine ===")
    out = NLP_DIR / "entites.json"
    print(f"  {out.stat().st_size / 1024:.1f} KB  {out}")
