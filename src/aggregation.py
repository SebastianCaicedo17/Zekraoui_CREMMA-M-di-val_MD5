"""Agrégation des transcriptions multi-modèles et constitution du JSON livrable.

Produit le dataset_nlp/ conforme au data contract pour le module NLP.
"""

import json
from pathlib import Path
from typing import Optional

import jsonschema
import numpy as np

# ---------------------------------------------------------------------------
# Data contract — schéma JSON
# ---------------------------------------------------------------------------

SCHEMA_DATA_CONTRACT = {
    "type": "object",
    "required": ["image_id", "line_id", "transcription", "confidence",
                 "needs_review", "polygon", "page_xml_ref"],
    "properties": {
        "image_id":       {"type": "string", "minLength": 1},
        "line_id":        {"type": "string", "minLength": 1},
        "transcription":  {"type": "string"},
        "confidence":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "needs_review":   {"type": "boolean"},
        "has_abbreviation": {"type": "boolean"},
        "polygon": {
            "type": "array",
            "minItems": 3,
            "items": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
        },
        "page_xml_ref":   {"type": "string", "minLength": 1},
    },
    "additionalProperties": True,
}


# ---------------------------------------------------------------------------
# Alignement Needleman-Wunsch
# ---------------------------------------------------------------------------

def aligner_needleman_wunsch(
    seq1: str,
    seq2: str,
    match: int = 2,
    mismatch: int = -1,
    gap: int = -2,
) -> tuple[str, str]:
    """Aligne deux chaînes de caractères par l'algorithme Needleman-Wunsch.

    Args:
        seq1: Première séquence (ex. transcription TrOCR).
        seq2: Deuxième séquence (ex. transcription Kraken).
        match: Score pour un appariement identique.
        mismatch: Pénalité pour une substitution.
        gap: Pénalité pour un gap (insertion/suppression).

    Returns:
        Tuple ``(seq1_alignée, seq2_alignée)`` avec ``'-'`` pour les gaps.

    Example:
        >>> a1, a2 = aligner_needleman_wunsch("ATCG", "ACG")
        >>> a1, a2
        ('ATCG', 'A-CG')
    """
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = gap * i
    for j in range(m + 1):
        dp[0, j] = gap * j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i, j] = max(dp[i-1, j-1] + s, dp[i-1, j] + gap, dp[i, j-1] + gap)

    a1, a2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            if dp[i, j] == dp[i-1, j-1] + s:
                a1.append(seq1[i-1]); a2.append(seq2[j-1]); i -= 1; j -= 1; continue
        if i > 0 and dp[i, j] == dp[i-1, j] + gap:
            a1.append(seq1[i-1]); a2.append('-'); i -= 1
        else:
            a1.append('-'); a2.append(seq2[j-1]); j -= 1

    return "".join(reversed(a1)), "".join(reversed(a2))


# ---------------------------------------------------------------------------
# Fusion pondérée
# ---------------------------------------------------------------------------

def fusionner_transcriptions(
    trans1: str,
    conf1: float,
    trans2: str,
    conf2: float,
) -> tuple[str, float]:
    """Fusionne deux transcriptions par vote pondéré sur l'alignement NW.

    Quand les deux modèles s'accordent, le caractère est conservé.
    En cas de désaccord, le modèle avec la confiance la plus élevée l'emporte.
    Les gaps ne sont inclus que si la confiance du modèle source dépasse 0.5.

    Args:
        trans1: Transcription du modèle 1 (ex. TrOCR).
        conf1: Score de confiance du modèle 1 (0–1).
        trans2: Transcription du modèle 2 (ex. Kraken).
        conf2: Score de confiance du modèle 2 (0–1).

    Returns:
        Tuple ``(transcription_fusionnée, confiance_moyenne)``.

    Example:
        >>> fusionner_transcriptions("bonjour", 0.9, "bonjur", 0.7)
        ('bonjour', 0.8)
    """
    if not trans1:
        return trans2, conf2
    if not trans2:
        return trans1, conf1

    a1, a2 = aligner_needleman_wunsch(trans1, trans2)
    resultat = []
    for c1, c2 in zip(a1, a2):
        if c1 == c2:
            resultat.append(c1)
        elif c1 == "-":
            if conf2 > 0.5:
                resultat.append(c2)
        elif c2 == "-":
            if conf1 > 0.5:
                resultat.append(c1)
        else:
            resultat.append(c1 if conf1 >= conf2 else c2)

    return "".join(resultat), round((conf1 + conf2) / 2, 4)


# ---------------------------------------------------------------------------
# Flag needs_review
# ---------------------------------------------------------------------------

def flaguer_needs_review(
    transcription: str,
    confidence: float,
    has_abbreviation: bool = False,
    seuil_conf: float = 0.6,
    longueur_min: int = 5,
    trans2: Optional[str] = None,
) -> bool:
    """Détermine si une ligne nécessite une relecture humaine.

    Critères (union) :
    - Ligne trop courte (< ``longueur_min`` caractères)
    - Confiance < ``seuil_conf``
    - Abréviation détectée ET confiance < ``seuil_conf × 1.2`` (seuil relevé)
    - Discordance forte entre deux modèles (> 30 % d'édition relative)

    Args:
        transcription: Texte prédit par le modèle principal.
        confidence: Score de confiance (0–1).
        has_abbreviation: True si la ligne contient des abréviations.
        seuil_conf: Seuil de confiance minimum acceptable.
        longueur_min: Longueur minimale d'une ligne valide.
        trans2: Transcription optionnelle d'un second modèle pour comparaison.

    Returns:
        True si la ligne doit être revue manuellement.

    Example:
        >>> flaguer_needs_review("ab", 0.9, False)
        True
    """
    if len(transcription.strip()) < longueur_min:
        return True
    if confidence < seuil_conf:
        return True
    if has_abbreviation and confidence < seuil_conf * 1.2:
        return True
    if trans2 is not None:
        import editdistance
        dist = editdistance.eval(transcription, trans2)
        if dist / max(1, len(transcription)) > 0.3:
            return True
    return False


# ---------------------------------------------------------------------------
# Validation et production d'entrées data contract
# ---------------------------------------------------------------------------

def valider_entree(entree: dict) -> list[str]:
    """Valide une entrée JSON contre le data contract.

    Args:
        entree: Dictionnaire à valider.

    Returns:
        Liste des messages d'erreur (liste vide si l'entrée est valide).

    Example:
        >>> erreurs = valider_entree({"image_id": "...", ...})
    """
    erreurs: list[str] = []
    try:
        jsonschema.validate(instance=entree, schema=SCHEMA_DATA_CONTRACT)
    except jsonschema.ValidationError as e:
        erreurs.append(e.message)
    return erreurs


def produire_entree_livrable(
    image_id: str,
    line_id: str,
    transcription: str,
    confidence: float,
    polygon: list,
    page_xml_ref: str,
    has_abbreviation: bool = False,
    seuil_conf: float = 0.6,
    trans2: Optional[str] = None,
) -> dict:
    """Produit une entrée JSON validée conforme au data contract.

    Args:
        image_id: Identifiant de l'image source.
        line_id: Identifiant de la ligne dans le PAGE/ALTO XML.
        transcription: Texte prédit par le modèle HTR.
        confidence: Score de confiance (0–1).
        polygon: Polygone de contour ``[[x, y], ...]`` (≥ 3 points).
        page_xml_ref: Chemin relatif vers le fichier PAGE XML correspondant.
        has_abbreviation: True si la transcription contient des abréviations.
        seuil_conf: Seuil pour le flag needs_review.
        trans2: Transcription d'un second modèle (pour détection discordances).

    Returns:
        Dictionnaire validé.

    Raises:
        ValueError: Si l'entrée ne satisfait pas le schéma.

    Example:
        >>> entree = produire_entree_livrable("img_001", "l_001", "bonjour", 0.95, ...)
    """
    needs_review = flaguer_needs_review(
        transcription, confidence, has_abbreviation, seuil_conf, trans2=trans2
    )
    entree = {
        "image_id": image_id,
        "line_id": line_id,
        "transcription": transcription,
        "confidence": round(float(confidence), 4),
        "needs_review": needs_review,
        "has_abbreviation": has_abbreviation,
        "polygon": [[float(x), float(y)] for x, y in polygon],
        "page_xml_ref": page_xml_ref,
    }
    erreurs = valider_entree(entree)
    if erreurs:
        raise ValueError(f"Entrée invalide pour '{line_id}' : {erreurs}")
    return entree


# ---------------------------------------------------------------------------
# Génération du dataset livrable
# ---------------------------------------------------------------------------

def generer_dataset_nlp(
    manifest_path: Path,
    out_path: Path,
    predictions: Optional[list[str]] = None,
    confidences: Optional[list[float]] = None,
    polygons: Optional[list[list]] = None,
    seuil_conf: float = 0.6,
) -> list[dict]:
    """Génère le fichier JSON livrable pour le module NLP depuis un manifest HTR.

    Si ``predictions`` est absent, utilise la vérité terrain comme transcription
    (mode « oracle » pour valider le pipeline sans modèle entraîné).

    Args:
        manifest_path: Manifest JSON produit par :func:`src.htr.construire_dataset`.
        out_path: Chemin de sortie du fichier JSON livrable.
        predictions: Transcriptions prédites (une par ligne du manifest).
        confidences: Scores de confiance (1.0 si absent).
        polygons: Polygones de contour (placeholder unitaire si absent).
        seuil_conf: Seuil pour le flag needs_review.

    Returns:
        Liste des entrées du dataset validées contre le data contract.

    Example:
        >>> dataset = generer_dataset_nlp(Path("data/lignes/manifest_val.json"),
        ...                               Path("dataset_nlp/val.json"))
    """
    items = json.loads(manifest_path.read_text(encoding="utf-8"))

    preds = predictions if predictions is not None else [it["text"] for it in items]
    confs = confidences if confidences is not None else [1.0] * len(items)
    polys = polygons if polygons is not None else [[[0, 0], [1, 0], [1, 1], [0, 1]]] * len(items)

    dataset: list[dict] = []
    invalides = 0
    for i, item in enumerate(items):
        image_id = f"{item['manuscrit']}_{Path(item['source_xml']).stem}"
        page_xml_ref = f"segmentations/{Path(item['source_xml']).stem}.xml"
        try:
            entree = produire_entree_livrable(
                image_id=image_id,
                line_id=item.get("line_id", f"line_{i:04d}"),
                transcription=preds[i],
                confidence=confs[i],
                polygon=polys[i],
                page_xml_ref=page_xml_ref,
                has_abbreviation=item.get("has_abbreviation", False),
                seuil_conf=seuil_conf,
            )
            dataset.append(entree)
        except ValueError:
            invalides += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    nb_review = sum(1 for e in dataset if e["needs_review"])
    nb_abrev = sum(1 for e in dataset if e["has_abbreviation"])
    print(f"[info] {len(dataset)} entrées valides  ({invalides} ignorées) -> {out_path}")
    print(f"       needs_review     : {nb_review:5d} ({nb_review / max(1, len(dataset)):.1%})")
    print(f"       has_abbreviation : {nb_abrev:5d} ({nb_abrev / max(1, len(dataset)):.1%})")

    return dataset


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.utils import fixer_seeds

    fixer_seeds(42)

    manifest = Path("data/lignes/manifest_val.json")
    if not manifest.exists():
        print("[erreur] Manifest introuvable. Lancez d'abord : python -m src.htr")
        sys.exit(1)

    print("=== Génération dataset_nlp (mode oracle — vérité terrain) ===")
    dataset = generer_dataset_nlp(
        manifest_path=manifest,
        out_path=Path("dataset_nlp/val_oracle.json"),
    )

    nb_review = sum(1 for e in dataset if e["needs_review"])
    nb_abrev_review = sum(1 for e in dataset if e["needs_review"] and e["has_abbreviation"])
    print(f"\n=== Analyse needs_review ===")
    print(f"Total needs_review      : {nb_review} ({nb_review / max(1, len(dataset)):.1%})")
    print(f"needs_review + abrev.   : {nb_abrev_review} ({nb_abrev_review / max(1, nb_review):.1%} des reviews)")
