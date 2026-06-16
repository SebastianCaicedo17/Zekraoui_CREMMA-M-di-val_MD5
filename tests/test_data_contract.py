"""Tests Phase 5 — data contract, agrégation et évaluation finale.

Couvre :
- Schéma JSON (data contract)
- Présence des polygones
- Absence de fuite train/test
- Taux needs_review
- Fusion Needleman-Wunsch
- Calcul WER
- Bootstrap IC (smoke test)
- Test de McNemar
- Analyse qualitative des abréviations
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.aggregation import (
    SCHEMA_DATA_CONTRACT,
    aligner_needleman_wunsch,
    flaguer_needs_review,
    fusionner_transcriptions,
    generer_dataset_nlp,
    produire_entree_livrable,
    valider_entree,
)
from src.evaluation import (
    analyser_erreurs_abreviations,
    bootstrap_ic,
    calculer_wer,
    calculer_wer_moyen,
    mcnemar,
    rapport_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTREE_VALIDE = {
    "image_id": "ms001_page01",
    "line_id": "l_0001",
    "transcription": "bonjour le monde",
    "confidence": 0.92,
    "needs_review": False,
    "has_abbreviation": False,
    "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "page_xml_ref": "segmentations/page01.xml",
}


def _manifest_tmp(tmp_path: Path, items: list[dict]) -> Path:
    """Écrit un manifest JSON minimal dans tmp_path."""
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def _item_manifest(
    text: str = "bonjour monde",
    manuscrit: str = "ms_test",
    has_abbreviation: bool = False,
) -> dict:
    return {
        "text": text,
        "manuscrit": manuscrit,
        "source_xml": "data/page01.xml",
        "split": "val",
        "has_abbreviation": has_abbreviation,
        "line_id": "l_0001",
    }


# ===========================================================================
# 1 — Data contract : validation de schéma
# ===========================================================================

class TestDataContractSchema:

    def test_entree_valide_aucune_erreur(self):
        assert valider_entree(_ENTREE_VALIDE) == []

    def test_champ_manquant_image_id(self):
        entree = {k: v for k, v in _ENTREE_VALIDE.items() if k != "image_id"}
        erreurs = valider_entree(entree)
        assert len(erreurs) > 0

    def test_champ_manquant_polygon(self):
        entree = {k: v for k, v in _ENTREE_VALIDE.items() if k != "polygon"}
        assert len(valider_entree(entree)) > 0

    def test_confidence_hors_borne_superieure(self):
        entree = {**_ENTREE_VALIDE, "confidence": 1.5}
        assert len(valider_entree(entree)) > 0

    def test_confidence_hors_borne_inferieure(self):
        entree = {**_ENTREE_VALIDE, "confidence": -0.1}
        assert len(valider_entree(entree)) > 0

    def test_polygon_trop_petit(self):
        entree = {**_ENTREE_VALIDE, "polygon": [[0.0, 0.0], [1.0, 1.0]]}
        assert len(valider_entree(entree)) > 0

    def test_image_id_vide(self):
        entree = {**_ENTREE_VALIDE, "image_id": ""}
        assert len(valider_entree(entree)) > 0

    def test_needs_review_type_incorrect(self):
        entree = {**_ENTREE_VALIDE, "needs_review": "oui"}
        assert len(valider_entree(entree)) > 0

    def test_champs_supplementaires_autorises(self):
        """additionalProperties=True : des champs extra ne doivent pas lever d'erreur."""
        entree = {**_ENTREE_VALIDE, "extra_field": "valeur"}
        assert valider_entree(entree) == []


# ===========================================================================
# 2 — Présence des polygones pour chaque ligne transcrite
# ===========================================================================

class TestPolygones:

    def test_produire_entree_polygon_present(self):
        entree = produire_entree_livrable(
            image_id="img_001",
            line_id="l_001",
            transcription="bonjour le monde",
            confidence=0.95,
            polygon=[[0, 0], [10, 0], [10, 5], [0, 5]],
            page_xml_ref="segmentations/p01.xml",
        )
        assert "polygon" in entree
        assert len(entree["polygon"]) >= 3

    def test_polygon_coordonnees_numeriques(self):
        entree = produire_entree_livrable(
            image_id="img_001",
            line_id="l_001",
            transcription="texte valide ici",
            confidence=0.80,
            polygon=[[0, 0], [100, 0], [100, 30], [0, 30]],
            page_xml_ref="segmentations/p01.xml",
        )
        for pt in entree["polygon"]:
            assert len(pt) == 2
            assert all(isinstance(v, float) for v in pt)

    def test_dataset_toutes_entrees_ont_polygon(self, tmp_path):
        items = [_item_manifest(f"ligne {i} suffisamment longue") for i in range(5)]
        manifest = _manifest_tmp(tmp_path, items)
        out = tmp_path / "out.json"
        dataset = generer_dataset_nlp(manifest, out)
        for entree in dataset:
            assert "polygon" in entree
            assert len(entree["polygon"]) >= 3


# ===========================================================================
# 3 — Absence de fuite train/test (hash-based)
# ===========================================================================

class TestAbsenceFuiteTrainTest:

    def test_pas_de_chevauchement_image_id(self, tmp_path):
        """Les image_id du train et du test ne doivent pas se croiser."""
        items_train = [_item_manifest(f"texte train {i} long") for i in range(3)]
        items_test = [_item_manifest(f"texte test {i} long") for i in range(3)]
        for it in items_test:
            it["manuscrit"] = "ms_test_only"

        (tmp_path / "train").mkdir(exist_ok=True)
        (tmp_path / "test").mkdir(exist_ok=True)
        manifest_train = _manifest_tmp(tmp_path / "train", items_train)
        manifest_test = _manifest_tmp(tmp_path / "test", items_test)

        ds_train = generer_dataset_nlp(manifest_train, tmp_path / "train.json")
        ds_test = generer_dataset_nlp(manifest_test, tmp_path / "test.json")

        ids_train = {e["image_id"] for e in ds_train}
        ids_test = {e["image_id"] for e in ds_test}
        assert ids_train.isdisjoint(ids_test), "Fuite détectée : image_id partagés train/test"


# ===========================================================================
# 4 — Taux needs_review
# ===========================================================================

class TestNeedsReview:

    def test_ligne_courte_flaggee(self):
        assert flaguer_needs_review("ab", confidence=0.95) is True

    def test_confidence_faible_flaggee(self):
        assert flaguer_needs_review("une ligne suffisamment longue", confidence=0.3) is True

    def test_abreviation_confiance_limite(self):
        # seuil_conf=0.6, seuil abréviation = 0.6 * 1.2 = 0.72
        assert flaguer_needs_review("ꝯm̃ce le roi", confidence=0.65, has_abbreviation=True) is True

    def test_ligne_ok_non_flaggee(self):
        assert flaguer_needs_review("une ligne normale bien transcrite", confidence=0.85) is False

    def test_discordance_inter_modeles(self):
        ref = "bonjour le monde entier"
        pred2 = "xyz abc defgh ijkl"   # > 30 % d'édition
        assert flaguer_needs_review(ref, confidence=0.9, trans2=pred2) is True

    def test_taux_needs_review_inferieur_seuil(self, tmp_path):
        """Le taux needs_review doit rester raisonnable sur des prédictions correctes."""
        items = [
            _item_manifest("ligne suffisamment longue et bien transcrite")
            for _ in range(20)
        ]
        manifest = _manifest_tmp(tmp_path, items)
        out = tmp_path / "out.json"
        preds = [it["text"] for it in items]
        confs = [0.95] * len(items)
        polys = [[[0, 0], [10, 0], [10, 5], [0, 5]]] * len(items)
        dataset = generer_dataset_nlp(manifest, out, predictions=preds,
                                      confidences=confs, polygons=polys)
        taux = sum(1 for e in dataset if e["needs_review"]) / len(dataset)
        assert taux < 0.3, f"Taux needs_review trop élevé : {taux:.1%}"


# ===========================================================================
# 5 — Fusion Needleman-Wunsch
# ===========================================================================

class TestNeedlemanWunsch:

    def test_sequences_identiques(self):
        a1, a2 = aligner_needleman_wunsch("ATCG", "ATCG")
        assert a1 == a2 == "ATCG"

    def test_gap_dans_seq2(self):
        a1, a2 = aligner_needleman_wunsch("ATCG", "ACG")
        assert len(a1) == len(a2)
        assert "-" in a2

    def test_longueurs_alignement_egales(self):
        a1, a2 = aligner_needleman_wunsch("bonjour", "bnjour")
        assert len(a1) == len(a2)

    def test_fusion_accord_total(self):
        trans, conf = fusionner_transcriptions("bonjour", 0.9, "bonjour", 0.8)
        assert trans == "bonjour"
        assert conf == pytest.approx(0.85)

    def test_fusion_desaccord_modele1_gagne(self):
        trans, _ = fusionner_transcriptions("bonjour", 0.9, "bonjur", 0.5)
        assert "o" in trans

    def test_fusion_chaine_vide_gauche(self):
        trans, conf = fusionner_transcriptions("", 0.0, "monde", 0.8)
        assert trans == "monde"
        assert conf == pytest.approx(0.8)


# ===========================================================================
# 6 — WER
# ===========================================================================

class TestWER:

    def test_wer_identique(self):
        assert calculer_wer("le roi de France", "le roi de France") == pytest.approx(0.0)

    def test_wer_reference_vide_prediction_vide(self):
        assert calculer_wer("", "") == pytest.approx(0.0)

    def test_wer_reference_vide_prediction_non_vide(self):
        assert calculer_wer("mot", "") == pytest.approx(1.0)

    def test_wer_valeur_connue(self):
        # "le roi France" vs "le roi de France" : 1 suppression sur 4 mots
        assert calculer_wer("le roi France", "le roi de France") == pytest.approx(1 / 4)

    def test_wer_moyen_parfait(self):
        preds = ["bonjour monde", "le roi"]
        refs = ["bonjour monde", "le roi"]
        assert calculer_wer_moyen(preds, refs) == pytest.approx(0.0)

    def test_wer_moyen_liste_vide(self):
        assert calculer_wer_moyen([], []) == pytest.approx(0.0)


# ===========================================================================
# 7 — Bootstrap IC (smoke test)
# ===========================================================================

class TestBootstrapIC:

    def test_smoke_oracle(self):
        """Sur des prédictions parfaites, CER_global doit être 0."""
        refs = ["bonjour", "monde", "ꝯm̃ce"]
        flags = [False, False, True]
        ic = bootstrap_ic(refs, refs, flags, n=50, seed=42)
        assert ic["CER_global"]["mean"] == pytest.approx(0.0, abs=1e-6)

    def test_ic_lower_inferieur_upper(self):
        refs = ["bonjour le monde", "ꝯm̃ce le liure", "texte ordinaire ici"]
        preds = ["bnjour le monde", "ꝯmce le liure", "texte ordinaire ici"]
        flags = [False, True, False]
        ic = bootstrap_ic(preds, refs, flags, n=100, seed=0)
        for key in ["CER_global", "CER_abbrev", "CER_no_abbrev"]:
            entry = ic[key]
            if entry["lower"] is not None:
                assert entry["lower"] <= entry["upper"]

    def test_pas_de_lignes_abreviees(self):
        refs = ["bonjour", "monde"]
        preds = ["bonjour", "monde"]
        flags = [False, False]
        ic = bootstrap_ic(preds, refs, flags, n=50)
        assert ic["CER_abbrev"]["mean"] is None


# ===========================================================================
# 8 — Test de McNemar
# ===========================================================================

class TestMcNemar:

    def test_pas_de_discordance(self):
        erreurs = [True, False, True, False]
        r = mcnemar(erreurs, erreurs)
        assert r["b"] == 0 and r["c"] == 0
        assert r["p_value"] == pytest.approx(1.0)
        assert r["significatif"] is False

    def test_discordance_significative(self):
        # b >> c -> chi2 élevé -> p < 0.05
        e1 = [True] * 50 + [False] * 50
        e2 = [False] * 50 + [False] * 50
        r = mcnemar(e1, e2)
        assert r["b"] == 50 and r["c"] == 0
        assert r["significatif"] is True

    def test_format_retour(self):
        r = mcnemar([True, False], [False, True])
        for key in ("b", "c", "chi2", "p_value", "significatif"):
            assert key in r


# ===========================================================================
# 9 — Analyse qualitative abréviations
# ===========================================================================

class TestAnalyseAbreviations:

    def test_cer_abrev_plus_eleve(self):
        preds = ["bonjour monde", "???"]
        refs = ["bonjour monde", "ꝯm̃ce le liure du roi"]
        flags = [False, True]
        r = analyser_erreurs_abreviations(preds, refs, flags)
        assert r["cer_lignes_abrev"] is not None
        assert r["cer_lignes_abrev"] > r["cer_lignes_normales"]

    def test_delta_absolu_positif(self):
        preds = ["bonjour monde texte normal", "???"]
        refs = ["bonjour monde texte normal", "ꝯm̃ce le liure"]
        flags = [False, True]
        r = analyser_erreurs_abreviations(preds, refs, flags)
        assert r["delta_absolu"] > 0

    def test_sans_lignes_abreviees(self):
        preds = refs = ["bonjour", "monde"]
        flags = [False, False]
        r = analyser_erreurs_abreviations(preds, refs, flags)
        assert r["cer_lignes_abrev"] is None
        assert r["cer_lignes_normales"] == pytest.approx(0.0)
        assert r["par_caractere"] == {}

    def test_ratio_impact(self):
        preds = refs = ["bonjour", "monde", "ꝯm̃ce"]
        flags = [False, False, True]
        r = analyser_erreurs_abreviations(preds, refs, flags)
        assert r["ratio_impact"] == pytest.approx(1 / 3, rel=0.01)


# ===========================================================================
# 10 — Rapport complet (smoke test)
# ===========================================================================

class TestRapportEvaluation:

    def test_rapport_oracle_cer_zero(self, tmp_path):
        refs = ["bonjour monde texte", "ꝯm̃ce le liure", "normale suffisamment longue"]
        flags = [False, True, False]
        r = rapport_evaluation(
            predictions=refs,
            references=refs,
            has_abbreviations=flags,
            out_path=tmp_path / "rapport.json",
            n_bootstrap=50,
        )
        assert r["cer"]["CER_global"] == pytest.approx(0.0, abs=1e-6)
        assert (tmp_path / "rapport.json").exists()

    def test_rapport_inclut_toutes_cles(self, tmp_path):
        refs = ["bonjour monde", "ꝯm̃ce liure"]
        flags = [False, True]
        r = rapport_evaluation(
            predictions=refs, references=refs, has_abbreviations=flags,
            n_bootstrap=20,
        )
        for key in ("cer", "wer", "bootstrap_ic", "analyse_abreviations",
                    "mcnemar", "conclusion_problematique"):
            assert key in r

    def test_rapport_avec_deux_modeles(self, tmp_path):
        refs = ["bonjour monde", "ꝯm̃ce liure long"]
        preds1 = ["bonjour monde", "ꝯm̃ce liure long"]
        preds2 = ["bnjour monde", "ce liure lon"]
        flags = [False, True]
        r = rapport_evaluation(
            predictions=preds1, references=refs, has_abbreviations=flags,
            predictions_modele2=preds2, n_bootstrap=20,
        )
        assert r["mcnemar"] is not None
        assert "b" in r["mcnemar"]