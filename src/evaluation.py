"""Évaluation finale du pipeline HTR : CER, WER, bootstrap, McNemar, analyse qualitative.

Répond à la problématique : *Dans quelle mesure les abréviations médiévales
influencent-elles les performances d'un pipeline HTR appliqué au corpus CREMMA ?*
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import editdistance
import numpy as np

# ---------------------------------------------------------------------------
# WER (Word Error Rate)
# ---------------------------------------------------------------------------

def calculer_wer(prediction: str, reference: str) -> float:
    """Calcule le Word Error Rate par distance d'édition sur les mots.

    Args:
        prediction: Transcription prédite.
        reference: Transcription de référence (vérité terrain).

    Returns:
        WER ∈ [0, ∞) ; peut dépasser 1.0 si la prédiction est plus longue.

    Example:
        >>> calculer_wer("le roi de France", "le roi de France")
        0.0
        >>> calculer_wer("le roi France", "le roi de France")
        0.25
    """
    mots_ref = reference.split()
    mots_pred = prediction.split()
    if not mots_ref:
        return 0.0 if not mots_pred else 1.0
    return editdistance.eval(mots_pred, mots_ref) / len(mots_ref)


def calculer_wer_moyen(predictions: list[str], references: list[str]) -> float:
    """WER moyen sur une liste de paires (macro-average).

    Args:
        predictions: Liste des transcriptions prédites.
        references: Liste des vérités terrain.

    Returns:
        WER moyen ∈ [0, ∞).

    Example:
        >>> calculer_wer_moyen(["bonjour monde"], ["bonjour monde"])
        0.0
    """
    if not predictions:
        return 0.0
    wers = [calculer_wer(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(wers))


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ic(
    predictions: list[str],
    references: list[str],
    has_abbreviations: list[bool],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Calcule les intervalles de confiance bootstrap pour CER global, CER_abbrev,
    CER_no_abbrev et delta_CER.

    Args:
        predictions: Transcriptions prédites.
        references: Vérités terrain.
        has_abbreviations: Flags booléens par ligne (True = contient une abréviation).
        n: Nombre de ré-échantillonnages.
        alpha: Niveau de signification (IC à 1 - alpha).
        seed: Graine aléatoire pour reproductibilité.

    Returns:
        Dictionnaire avec clés ``CER_global``, ``CER_abbrev``, ``CER_no_abbrev``,
        ``delta_CER`` — chacune sous forme ``{"mean": ..., "lower": ..., "upper": ...}``.

    Example:
        >>> ic = bootstrap_ic(["bonjour"], ["bonjour"], [False], n=100)
        >>> ic["CER_global"]["mean"]
        0.0
    """
    rng = random.Random(seed)

    def _cer(pred: str, ref: str) -> float:
        if not ref:
            return 0.0 if not pred else 1.0
        return editdistance.eval(pred, ref) / len(ref)

    indices = list(range(len(predictions)))
    idx_abrev = [i for i in indices if has_abbreviations[i]]
    idx_no = [i for i in indices if not has_abbreviations[i]]

    samples: dict[str, list[float]] = {
        "CER_global": [],
        "CER_abbrev": [],
        "CER_no_abbrev": [],
        "delta_CER": [],
    }

    for _ in range(n):
        boot = [rng.choice(indices) for _ in indices]
        samples["CER_global"].append(
            float(np.mean([_cer(predictions[i], references[i]) for i in boot]))
        )

        if idx_abrev:
            ba = [rng.choice(idx_abrev) for _ in idx_abrev]
            ca = float(np.mean([_cer(predictions[i], references[i]) for i in ba]))
            samples["CER_abbrev"].append(ca)
        else:
            samples["CER_abbrev"].append(float("nan"))

        if idx_no:
            bn = [rng.choice(idx_no) for _ in idx_no]
            cn = float(np.mean([_cer(predictions[i], references[i]) for i in bn]))
            samples["CER_no_abbrev"].append(cn)
        else:
            samples["CER_no_abbrev"].append(float("nan"))

        if idx_abrev and idx_no:
            samples["delta_CER"].append(
                samples["CER_abbrev"][-1] - samples["CER_no_abbrev"][-1]
            )
        else:
            samples["delta_CER"].append(float("nan"))

    result: dict = {}
    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100
    for key, vals in samples.items():
        arr = [v for v in vals if not np.isnan(v)]
        if arr:
            result[key] = {
                "mean": float(np.mean(arr)),
                "lower": float(np.percentile(arr, lo)),
                "upper": float(np.percentile(arr, hi)),
            }
        else:
            result[key] = {"mean": None, "lower": None, "upper": None}
    return result


# ---------------------------------------------------------------------------
# Test de McNemar
# ---------------------------------------------------------------------------

def mcnemar(
    erreurs_modele1: list[bool],
    erreurs_modele2: list[bool],
) -> dict:
    """Test de McNemar (correction de continuité) pour comparer deux modèles.

    Calcule le chi² avec correction de Yates sur les discordances (b, c).

    Args:
        erreurs_modele1: Liste de booléens (True = ligne erronée) pour le modèle 1.
        erreurs_modele2: Idem pour le modèle 2.

    Returns:
        Dictionnaire : ``b``, ``c``, ``chi2``, ``p_value``, ``significatif`` (alpha=0.05).

    Example:
        >>> r = mcnemar([True, False, True], [False, False, True])
        >>> r["b"] + r["c"] >= 0
        True
    """
    b = sum(1 for e1, e2 in zip(erreurs_modele1, erreurs_modele2) if e1 and not e2)
    c = sum(1 for e1, e2 in zip(erreurs_modele1, erreurs_modele2) if not e1 and e2)

    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p_value": 1.0, "significatif": False}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value approximation via chi² à 1 degré de liberté
    # Utiliser scipy si disponible, sinon approximation basique
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = float(1 - chi2_dist.cdf(chi2, df=1))
    except ImportError:
        # Approximation normale : chi2 ≈ z², p ≈ erfc(sqrt(chi2/2)/sqrt(2))
        import math
        p_value = float(math.erfc(math.sqrt(chi2 / 2) / math.sqrt(2)))

    return {
        "b": b,
        "c": c,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significatif": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Analyse qualitative des erreurs liées aux abréviations
# ---------------------------------------------------------------------------

def analyser_erreurs_abreviations(
    predictions: list[str],
    references: list[str],
    has_abbreviations: list[bool],
) -> dict:
    """Analyse qualitative des erreurs HTR en fonction des abréviations médiévales.

    Pour chaque caractère d'abréviation présent dans les références, calcule :
    - le CER moyen des lignes contenant ce caractère
    - le nombre de lignes impactées

    Args:
        predictions: Transcriptions prédites.
        references: Vérités terrain.
        has_abbreviations: Flags (True = ligne contient une abréviation).

    Returns:
        Dictionnaire avec clés :
        - ``par_caractere`` : dict char -> {``cer_moyen``, ``nb_lignes``}
        - ``cer_lignes_abrev`` : CER moyen sur les lignes abrégées
        - ``cer_lignes_normales`` : CER moyen sur les lignes normales
        - ``delta_absolu`` : différence absolue des deux CER
        - ``ratio_impact`` : proportion de lignes abrégées dans le corpus

    Example:
        >>> r = analyser_erreurs_abreviations(["abc"], ["abc"], [False])
        >>> r["cer_lignes_normales"]
        0.0
    """
    from src.data_loader import detecter_abreviations

    def _cer(pred: str, ref: str) -> float:
        if not ref:
            return 0.0 if not pred else 1.0
        return editdistance.eval(pred, ref) / len(ref)

    per_char: dict[str, list[float]] = defaultdict(list)
    cers_abrev: list[float] = []
    cers_normal: list[float] = []

    for pred, ref, has_abr in zip(predictions, references, has_abbreviations):
        cer = _cer(pred, ref)
        if has_abr:
            cers_abrev.append(cer)
            for ch in detecter_abreviations(ref):
                per_char[ch].append(cer)
        else:
            cers_normal.append(cer)

    par_caractere = {
        ch: {
            "cer_moyen": round(float(np.mean(vals)), 4),
            "nb_lignes": len(vals),
        }
        for ch, vals in sorted(per_char.items(), key=lambda kv: -np.mean(kv[1]))
    }

    cer_abrev = float(np.mean(cers_abrev)) if cers_abrev else None
    cer_normal = float(np.mean(cers_normal)) if cers_normal else None
    delta = round(cer_abrev - cer_normal, 4) if (cer_abrev is not None and cer_normal is not None) else None
    ratio = len(cers_abrev) / max(1, len(predictions))

    return {
        "par_caractere": par_caractere,
        "cer_lignes_abrev": round(cer_abrev, 4) if cer_abrev is not None else None,
        "cer_lignes_normales": round(cer_normal, 4) if cer_normal is not None else None,
        "delta_absolu": delta,
        "ratio_impact": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

def rapport_evaluation(
    predictions: list[str],
    references: list[str],
    has_abbreviations: list[bool],
    out_path: Optional[Path] = None,
    predictions_modele2: Optional[list[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Produit le rapport d'évaluation complet répondant à la problématique.

    Calcule : CER stratifié, WER, intervalles de confiance bootstrap (N=1000),
    test de McNemar (si deux modèles), analyse qualitative par caractère
    d'abréviation.

    Args:
        predictions: Transcriptions du modèle principal.
        references: Vérités terrain.
        has_abbreviations: Flags abréviation par ligne.
        out_path: Si fourni, sauvegarde le rapport JSON à cet emplacement.
        predictions_modele2: Transcriptions d'un second modèle (pour McNemar).
        n_bootstrap: Nombre de ré-échantillonnages bootstrap.
        seed: Graine pour reproductibilité.

    Returns:
        Dictionnaire complet du rapport.
    """
    from src.htr import calculer_cer, evaluer_stratifie

    strat = evaluer_stratifie(predictions, references, has_abbreviations)
    wer_global = calculer_wer_moyen(predictions, references)

    idx_abrev = [i for i, f in enumerate(has_abbreviations) if f]
    idx_no = [i for i, f in enumerate(has_abbreviations) if not f]
    wer_abrev = calculer_wer_moyen(
        [predictions[i] for i in idx_abrev], [references[i] for i in idx_abrev]
    ) if idx_abrev else None
    wer_no_abbrev = calculer_wer_moyen(
        [predictions[i] for i in idx_no], [references[i] for i in idx_no]
    ) if idx_no else None

    ic = bootstrap_ic(predictions, references, has_abbreviations, n=n_bootstrap, seed=seed)
    analyse = analyser_erreurs_abreviations(predictions, references, has_abbreviations)

    rapport: dict = {
        "cer": strat,
        "wer": {
            "global": round(wer_global, 4),
            "abbrev": round(wer_abrev, 4) if wer_abrev is not None else None,
            "no_abbrev": round(wer_no_abbrev, 4) if wer_no_abbrev is not None else None,
            "delta": round(wer_abrev - wer_no_abbrev, 4)
            if (wer_abrev is not None and wer_no_abbrev is not None) else None,
        },
        "bootstrap_ic": ic,
        "analyse_abreviations": analyse,
        "mcnemar": None,
        "conclusion_problematique": _rediger_conclusion(strat, ic, analyse),
    }

    if predictions_modele2 is not None:
        def _err(pred: str, ref: str) -> bool:
            return editdistance.eval(pred, ref) > 0

        e1 = [_err(p, r) for p, r in zip(predictions, references)]
        e2 = [_err(p, r) for p, r in zip(predictions_modele2, references)]
        rapport["mcnemar"] = mcnemar(e1, e2)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[info] Rapport écrit -> {out_path}")

    _afficher_rapport(rapport)
    return rapport


def _rediger_conclusion(strat: dict, ic: dict, analyse: dict) -> str:
    """Formule une conclusion textuelle sur l'impact des abréviations."""
    cer_g = strat.get("CER_global")
    cer_a = strat.get("CER_abbrev")
    cer_n = strat.get("CER_no_abbrev")
    delta = strat.get("delta_CER")

    if cer_a is None:
        return "Aucune ligne abrégée dans le corpus évalué — conclusion impossible."

    tendance = "plus élevé" if (delta is not None and delta > 0) else "comparable ou inférieur"
    ic_delta = ic.get("delta_CER", {})
    ic_str = ""
    if ic_delta.get("lower") is not None:
        ic_str = (
            f" (IC 95% bootstrap : [{ic_delta['lower']:.3f}, {ic_delta['upper']:.3f}])"
        )

    top_chars = list(analyse.get("par_caractere", {}).keys())[:3]
    chars_str = ", ".join(f"«{c}»" for c in top_chars) if top_chars else "aucun"

    return (
        f"Le CER global est {cer_g:.3f}. "
        f"Sur les lignes contenant des abréviations médiévales, "
        f"le CER est {tendance} ({cer_a:.3f}) par rapport aux lignes ordinaires ({cer_n:.3f}){ic_str}. "
        f"Les caractères d'abréviation les plus impactants sont : {chars_str}. "
        f"Les abréviations représentent {analyse['ratio_impact']:.1%} du corpus. "
        f"Ces résultats indiquent que les abréviations "
        f"{'dégradent significativement' if (delta is not None and delta > 0.05) else 'impactent modérément'} "
        f"les performances HTR sur le corpus CREMMA."
    )


def _afficher_rapport(rapport: dict) -> None:
    """Affiche un résumé lisible du rapport dans la console."""
    cer = rapport["cer"]
    wer = rapport["wer"]
    ic = rapport["bootstrap_ic"]

    print("\n=== Rapport d'évaluation HTR ===")
    print(f"CER global      : {cer.get('CER_global', 'N/A'):.4f}")
    print(f"CER abréviations: {cer.get('CER_abbrev') or 'N/A'}")
    print(f"CER sans abrév. : {cer.get('CER_no_abbrev') or 'N/A'}")
    print(f"delta_CER       : {cer.get('delta_CER') or 'N/A'}")
    print(f"WER global      : {wer['global']:.4f}")
    ic_g = ic.get("CER_global", {})
    if ic_g.get("lower") is not None:
        print(f"IC 95% CER_global : [{ic_g['lower']:.4f}, {ic_g['upper']:.4f}]")
    print(f"\nConclusion :\n{rapport['conclusion_problematique']}")

    mcn = rapport.get("mcnemar")
    if mcn:
        sig = "OUI" if mcn["significatif"] else "NON"
        print(f"\nMcNemar : chi2={mcn['chi2']}, p={mcn['p_value']}, significatif={sig}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from src.utils import fixer_seeds

    fixer_seeds(42)

    manifest_path = Path("data/lignes/manifest_test.json")
    if not manifest_path.exists():
        print("[erreur] Manifest test introuvable. Lancez d'abord : python -m src.htr")
        sys.exit(1)

    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = [it["text"] for it in items]
    has_abr = [it.get("has_abbreviation", False) for it in items]

    # Mode oracle : prédictions = vérité terrain (CER = 0 attendu)
    print("=== Évaluation oracle (vérité terrain = prédictions) ===")
    rapport_evaluation(
        predictions=references,
        references=references,
        has_abbreviations=has_abr,
        out_path=Path("experiments/rapport_oracle.json"),
        n_bootstrap=200,
    )