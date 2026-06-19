"""Tests pytest pour src/ner.py — NER sur transcriptions médiévales."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.ner import (
    annoter_entites,
    annoter_dataset,
    agregger_entites,
    TYPES_ENTITES,
)


# ─── Fixture : pipeline NER mocké ────────────────────────────────────────────

def _pipeline_mock(texte: str):
    """Simule le pipeline HuggingFace sans télécharger le modèle."""
    resultats = []
    mots_per = {"philippe": "PER", "marie": "PER", "jean": "PER", "dieu": "PER"}
    mots_loc = {"paris": "LOC", "france": "LOC", "rome": "LOC"}
    mots_org = {"eglise": "ORG"}
    for mot in texte.lower().split():
        mot_clean = mot.strip(".,;:!?")
        for dico in (mots_per, mots_loc, mots_org):
            if mot_clean in dico:
                debut = texte.lower().find(mot_clean)
                resultats.append({
                    "entity_group": dico[mot_clean],
                    "word": mot_clean,
                    "score": 0.95,
                    "start": debut,
                    "end": debut + len(mot_clean),
                })
    return resultats


@pytest.fixture
def pipeline_ner_mock():
    """Pipeline NER mocké — aucun téléchargement requis."""
    mock = MagicMock(side_effect=_pipeline_mock)
    return mock


@pytest.fixture
def dataset_simple(tmp_path):
    """Petit dataset de test."""
    data = [
        {
            "line_id": "l1",
            "transcription": "le roi Philippe alla vers Paris",
            "manuscrit": "bnf_ms",
            "has_abbreviation": False,
        },
        {
            "line_id": "l2",
            "transcription": "Dieu est en France et a Rome",
            "manuscrit": "bodmer_ms",
            "has_abbreviation": False,
        },
        {
            "line_id": "l3",
            "transcription": "ꝯmence le liure de Marie",
            "manuscrit": "bnf_ms",
            "has_abbreviation": True,
        },
        {
            "line_id": "l4",
            "transcription": "",
            "manuscrit": "bnf_ms",
            "has_abbreviation": False,
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p, data


# ─── Tests annoter_entites ────────────────────────────────────────────────────

class TestAnnoterEntites:
    def test_per_detecte(self, pipeline_ner_mock):
        entites = annoter_entites("le roi Philippe alla vers Paris", pipeline_ner_mock)
        types = {e["type"] for e in entites}
        assert "PER" in types

    def test_loc_detecte(self, pipeline_ner_mock):
        entites = annoter_entites("le roi Philippe alla vers Paris", pipeline_ner_mock)
        types = {e["type"] for e in entites}
        assert "LOC" in types

    def test_texte_vide(self, pipeline_ner_mock):
        assert annoter_entites("", pipeline_ner_mock) == []

    def test_texte_espace(self, pipeline_ner_mock):
        assert annoter_entites("   ", pipeline_ner_mock) == []

    def test_scores_valides(self, pipeline_ner_mock):
        entites = annoter_entites("Philippe alla vers Paris", pipeline_ner_mock)
        for e in entites:
            assert 0.0 <= e["score"] <= 1.0

    def test_types_valides(self, pipeline_ner_mock):
        entites = annoter_entites("Philippe alla vers Paris", pipeline_ner_mock)
        for e in entites:
            assert e["type"] in TYPES_ENTITES

    def test_champs_presents(self, pipeline_ner_mock):
        entites = annoter_entites("Philippe alla vers Paris", pipeline_ner_mock)
        for e in entites:
            assert {"mot", "type", "score", "debut", "fin"} <= e.keys()

    def test_sans_entite(self, pipeline_ner_mock):
        entites = annoter_entites("le liure est bone", pipeline_ner_mock)
        assert entites == []


# ─── Tests agregger_entites ───────────────────────────────────────────────────

class TestAggreggerEntites:
    def _make_lignes(self, entites_par_ligne_raw):
        return [
            {"line_id": str(i), "manuscrit": "ms_A", "entites": ents}
            for i, ents in enumerate(entites_par_ligne_raw)
        ]

    def test_comptage_correct(self):
        lignes = self._make_lignes([
            [{"mot": "philippe", "type": "PER", "score": 0.9, "debut": 0, "fin": 8}],
            [{"mot": "paris", "type": "LOC", "score": 0.95, "debut": 0, "fin": 5}],
            [{"mot": "philippe", "type": "PER", "score": 0.88, "debut": 0, "fin": 8}],
        ])
        stats = agregger_entites(lignes)
        assert stats["par_type"]["PER"]["nb_total"] == 2
        assert stats["par_type"]["LOC"]["nb_total"] == 1

    def test_top20_trie_par_freq(self):
        lignes = self._make_lignes([
            [{"mot": "dieu", "type": "PER", "score": 0.9, "debut": 0, "fin": 4}],
            [{"mot": "dieu", "type": "PER", "score": 0.9, "debut": 0, "fin": 4}],
            [{"mot": "philippe", "type": "PER", "score": 0.9, "debut": 0, "fin": 8}],
        ])
        stats = agregger_entites(lignes)
        top = stats["par_type"]["PER"]["top20"]
        assert top[0]["entite"] == "dieu"
        assert top[0]["freq"] == 2

    def test_repartition_manuscrits(self):
        lignes = [
            {"line_id": "l1", "manuscrit": "bnf",
             "entites": [{"mot": "paris", "type": "LOC", "score": 0.9, "debut": 0, "fin": 5}]},
            {"line_id": "l2", "manuscrit": "bodmer",
             "entites": [{"mot": "rome", "type": "LOC", "score": 0.9, "debut": 0, "fin": 4}]},
        ]
        stats = agregger_entites(lignes)
        assert "bnf" in stats["par_manuscrit"]
        assert "bodmer" in stats["par_manuscrit"]
        assert stats["par_manuscrit"]["bnf"]["LOC"] == 1

    def test_tous_types_presents(self):
        lignes = self._make_lignes([])
        stats = agregger_entites(lignes)
        for t in TYPES_ENTITES:
            assert t in stats["par_type"]

    def test_nb_formes_uniques(self):
        lignes = self._make_lignes([
            [{"mot": "dieu", "type": "PER", "score": 0.9, "debut": 0, "fin": 4}],
            [{"mot": "dieu", "type": "PER", "score": 0.9, "debut": 0, "fin": 4}],
            [{"mot": "marie", "type": "PER", "score": 0.9, "debut": 0, "fin": 5}],
        ])
        stats = agregger_entites(lignes)
        assert stats["par_type"]["PER"]["nb_formes_uniques"] == 2


# ─── Tests annoter_dataset ────────────────────────────────────────────────────

class TestAnnoterDataset:
    def test_cree_fichier_json(self, dataset_simple, pipeline_ner_mock, tmp_path):
        _, data = dataset_simple
        from src.nlp import charger_regles_nettoyage
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        rapport = annoter_dataset(
            data, pipeline_ner_mock, diplo, norm,
            out_path=tmp_path / "nlp" / "entites.json"
        )
        nlp_mod.NLP_DIR = old

        assert (tmp_path / "nlp" / "entites.json").exists()

    def test_nb_lignes_correct(self, dataset_simple, pipeline_ner_mock, tmp_path):
        _, data = dataset_simple
        from src.nlp import charger_regles_nettoyage
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        rapport = annoter_dataset(
            data, pipeline_ner_mock, diplo, norm,
            out_path=tmp_path / "nlp" / "entites.json"
        )
        nlp_mod.NLP_DIR = old

        assert rapport["meta"]["nb_lignes_annotees"] == len(data)

    def test_entites_par_ligne_contient_champs(self, dataset_simple, pipeline_ner_mock, tmp_path):
        _, data = dataset_simple
        from src.nlp import charger_regles_nettoyage
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        rapport = annoter_dataset(
            data, pipeline_ner_mock, diplo, norm,
            out_path=tmp_path / "nlp" / "entites.json"
        )
        nlp_mod.NLP_DIR = old

        for ligne in rapport["entites_par_ligne"]:
            assert "line_id" in ligne
            assert "transcription_originale" in ligne
            assert "transcription_nettoyee" in ligne
            assert "entites" in ligne

    def test_ligne_vide_gere(self, dataset_simple, pipeline_ner_mock, tmp_path):
        _, data = dataset_simple
        from src.nlp import charger_regles_nettoyage
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        rapport = annoter_dataset(
            data, pipeline_ner_mock, diplo, norm,
            out_path=tmp_path / "nlp" / "entites.json"
        )
        nlp_mod.NLP_DIR = old

        # La ligne vide (l4) doit être dans le résultat avec entites=[]
        lignes_vides = [l for l in rapport["entites_par_ligne"] if l["transcription_originale"] == ""]
        assert len(lignes_vides) == 1
        assert lignes_vides[0]["entites"] == []

    def test_statistiques_tous_types(self, dataset_simple, pipeline_ner_mock, tmp_path):
        _, data = dataset_simple
        from src.nlp import charger_regles_nettoyage
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        rapport = annoter_dataset(
            data, pipeline_ner_mock, diplo, norm,
            out_path=tmp_path / "nlp" / "entites.json"
        )
        nlp_mod.NLP_DIR = old

        for t in TYPES_ENTITES:
            assert t in rapport["statistiques"]
