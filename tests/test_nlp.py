"""Tests pytest pour src/nlp.py — pipeline NLP Volet 2."""

import json
import os
import sys
from pathlib import Path

import pytest

# Permet d'importer src.nlp depuis la racine du projet
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.nlp import (
    _classifier_char,
    _charger_cremmalab,
    construire_alphabet,
    charger_regles_nettoyage,
    nettoyer_ligne,
    nettoyer_dataset,
    tokeniser,
    construire_lexique,
    construire_dict_depuis_oracle,
    charger_dictionnaire,
    comparer_lexique_dictionnaire,
    calculer_precision_tokens,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_nlp(tmp_path):
    """Dossier nlp/ temporaire pour les outputs."""
    (tmp_path / "nlp").mkdir()
    return tmp_path


@pytest.fixture
def dataset_pred(tmp_path):
    """Faux dataset de prédictions."""
    data = [
        {"transcription": "que ce liure est bone", "manuscrit": "ms_A", "has_abbreviation": False},
        {"transcription": "ꝯmence par purte de pansee ⁊", "manuscrit": "ms_B", "has_abbreviation": True},
        {"transcription": "abituance de cuer cors", "manuscrit": "ms_A", "has_abbreviation": False},
    ]
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def dataset_oracle(tmp_path):
    """Faux dataset oracle (vérité terrain) avec plus de chars médiévaux."""
    data = [
        {"transcription": "que ce liure est bone ͣ", "manuscrit": "ms_A", "has_abbreviation": True},
        {"transcription": "ꝯmence par purte de pansee ⁊ ꝑ", "manuscrit": "ms_B", "has_abbreviation": True},
        {"transcription": "abituance de cuer cors ͤ", "manuscrit": "ms_A", "has_abbreviation": True},
    ]
    p = tmp_path / "oracle.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def regles(tmp_path):
    """Règles de nettoyage chargées depuis la vraie table.csv."""
    import src.nlp as nlp_mod
    old_nlp_dir = nlp_mod.NLP_DIR
    nlp_mod.NLP_DIR = tmp_path / "nlp"
    (tmp_path / "nlp").mkdir(exist_ok=True)
    diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "nlp" / "regex_rules.json")
    nlp_mod.NLP_DIR = old_nlp_dir
    return diplo, norm


# ─── Étape 1 : _classifier_char ─────────────────────────────────────────────

class TestClassifierChar:
    def test_latin_standard(self):
        assert _classifier_char("a") == "latin_standard"
        assert _classifier_char("Z") == "latin_standard"
        assert _classifier_char("é") == "latin_standard"

    def test_abreviation_medievale(self):
        assert _classifier_char("ꝯ") == "abreviation_medievale"   # U+A76F
        assert _classifier_char("ꝑ") == "abreviation_medievale"   # U+A751

    def test_combining_mark(self):
        assert _classifier_char("̃") == "combining_mark"  # tilde combinant
        assert _classifier_char("̅") == "combining_mark"  # macron combinant
        assert _classifier_char("ͣ") == "combining_mark"  # ͣ (a suscrit)
        assert _classifier_char("ͤ") == "combining_mark"  # ͤ (e suscrit)

    def test_ponctuation_speciale(self):
        assert _classifier_char("⁊") == "ponctuation_speciale"   # Tironian et
        assert _classifier_char("¶") == "ponctuation_speciale"   # pilcrow

    def test_autre(self):
        # L'espace est catégorie Zs dans unicodedata → ponctuation_speciale ou autre
        assert _classifier_char("0") == "autre"


# ─── Étape 1 : construire_alphabet ───────────────────────────────────────────

class TestConstruireAlphabet:
    def test_cree_fichier(self, dataset_pred, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        result = construire_alphabet(
            dataset_pred, dataset_oracle, out_path=tmp_path / "nlp" / "alphabet.json"
        )
        nlp_mod.NLP_DIR = old

        assert (tmp_path / "nlp" / "alphabet.json").exists()

    def test_meta_coherent(self, dataset_pred, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        result = construire_alphabet(
            dataset_pred, dataset_oracle, out_path=tmp_path / "nlp" / "alphabet.json"
        )
        nlp_mod.NLP_DIR = old

        meta = result["meta"]
        assert meta["nb_chars_total"] >= meta["nb_chars_predictions"]
        assert meta["nb_chars_total"] >= meta["nb_chars_oracle"]

    def test_manquants_detectes(self, dataset_pred, dataset_oracle, tmp_path):
        """ͣ et ͤ sont dans oracle mais pas dans pred → doivent être manquants."""
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        result = construire_alphabet(
            dataset_pred, dataset_oracle, out_path=tmp_path / "nlp" / "alphabet.json"
        )
        nlp_mod.NLP_DIR = old

        manquants = set(result["manquants_modele"])
        assert "ͣ" in manquants or "ͤ" in manquants or result["meta"]["nb_manquants_modele"] >= 1

    def test_categories_valides(self, dataset_pred, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        result = construire_alphabet(
            dataset_pred, dataset_oracle, out_path=tmp_path / "nlp" / "alphabet.json"
        )
        nlp_mod.NLP_DIR = old

        cats_valides = {"latin_standard", "abreviation_medievale", "combining_mark", "ponctuation_speciale", "autre"}
        for ch_entry in result["chars"]:
            assert ch_entry["categorie"] in cats_valides


# ─── Étape 2 : charger_regles_nettoyage ─────────────────────────────────────

class TestChargerReglesNettoyage:
    def test_retourne_deux_listes(self, tmp_path):
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        assert isinstance(diplo, list)
        assert isinstance(norm, list)

    def test_fichier_json_cree(self, tmp_path):
        charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        assert (tmp_path / "regex_rules.json").exists()

    def test_regles_diplo_sont_abreviations(self, tmp_path):
        from src.data_loader import est_char_abreviation
        diplo, _ = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        for src, _ in diplo:
            # Le pattern peut être multi-char (base + combining) : au moins un char doit être une abréviation
            assert any(est_char_abreviation(c) for c in src), f"{src!r} ne contient pas d'abréviation médiévale"

    def test_regles_norm_non_vides_si_table_existe(self, tmp_path):
        """Si table.csv est présent, il doit y avoir des règles norm."""
        if not Path("data/cremma-medieval/table.csv").exists():
            pytest.skip("table.csv absent")
        _, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        assert len(norm) > 0


# ─── Étape 2 : nettoyer_ligne ────────────────────────────────────────────────

class TestNettoyerLigne:
    def setup_method(self):
        self.diplo = [("ꝯ", "con"), ("⁊", "et"), ("ꝑ", "per")]
        self.norm = [("v", "u"), ("j", "i")]

    def test_expansion_abreviation_diplo(self):
        # ꝯ = "con" → conmence (forme médiévale correcte de "commence")
        r = nettoyer_ligne("ꝯmence", self.diplo, self.norm, mode="diplomatique")
        assert r == "conmence"

    def test_suppression_gap(self):
        r = nettoyer_ligne("le [---] liure", self.diplo, self.norm)
        assert "[---]" not in r
        assert "liure" in r

    def test_suppression_illisible(self):
        r = nettoyer_ligne("de [ill.] bone", self.diplo, self.norm)
        assert "[ill.]" not in r

    def test_suppression_tc(self):
        r = nettoyer_ligne("[TC] chapitre premier", self.diplo, self.norm)
        assert "[TC]" not in r

    def test_normalise_applique_v(self):
        r = nettoyer_ligne("venir", self.diplo, self.norm, mode="normalise")
        assert r == "uenir"

    def test_diplomatique_conserve_v(self):
        r = nettoyer_ligne("venir", self.diplo, self.norm, mode="diplomatique")
        assert r == "venir"

    def test_tironian_expand(self):
        r = nettoyer_ligne("cors ⁊ ame", self.diplo, self.norm, mode="diplomatique")
        assert "et" in r
        assert "⁊" not in r

    def test_espaces_multiples(self):
        r = nettoyer_ligne("le   liure   est", self.diplo, self.norm)
        assert "  " not in r

    def test_zero_width_supprime(self):
        r = nettoyer_ligne("le﻿liure", self.diplo, self.norm)
        assert "﻿" not in r


# ─── Étape 2 : nettoyer_dataset ─────────────────────────────────────────────

class TestNettoyerDataset:
    def test_ajoute_champ(self):
        items = [{"transcription": "ꝯmence", "manuscrit": "ms_A", "has_abbreviation": True}]
        diplo = [("ꝯ", "con")]
        result = nettoyer_dataset(items, diplo, [], mode="diplomatique")
        assert "transcription_nettoyee" in result[0]
        assert result[0]["transcription_nettoyee"] == "conmence"

    def test_conserve_champs_originaux(self):
        items = [{"transcription": "bone", "manuscrit": "ms_A", "has_abbreviation": False, "extra": 42}]
        result = nettoyer_dataset(items, [], [], mode="diplomatique")
        assert result[0]["extra"] == 42
        assert result[0]["transcription"] == "bone"


# ─── Étape 3 : tokeniser ─────────────────────────────────────────────────────

class TestTokeniser:
    def test_split_basic(self):
        assert tokeniser("que ce liure") == ["que", "ce", "liure"]

    def test_filtre_vide(self):
        assert tokeniser("  ") == []

    def test_filtre_ponctuation_pure(self):
        tokens = tokeniser("bone . est")
        assert "." not in tokens
        assert "bone" in tokens

    def test_medieval_text(self):
        tokens = tokeniser("commence par purte de pansee")
        assert "commence" in tokens
        assert len(tokens) == 5


# ─── Étape 3 : construire_lexique ────────────────────────────────────────────

class TestConstruireLexique:
    def test_cree_fichiers(self, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        lexique = construire_lexique(
            [(dataset_oracle, "oracle")], [], [],
            out_path=tmp_path / "nlp" / "lexique.json",
            out_txt=tmp_path / "nlp" / "lexique.txt",
        )
        nlp_mod.NLP_DIR = old

        assert (tmp_path / "nlp" / "lexique.json").exists()
        assert (tmp_path / "nlp" / "lexique.txt").exists()

    def test_mots_frequents_en_tete(self, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        lexique = construire_lexique(
            [(dataset_oracle, "oracle")], [], [],
            out_path=tmp_path / "nlp" / "lexique.json",
            out_txt=tmp_path / "nlp" / "lexique.txt",
        )
        nlp_mod.NLP_DIR = old

        # Le premier mot doit avoir la fréquence la plus haute
        formes = lexique["formes"]
        assert formes[0]["freq"] >= formes[-1]["freq"]

    def test_hapax_identifie(self, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        lexique = construire_lexique(
            [(dataset_oracle, "oracle")], [], [],
            out_path=tmp_path / "nlp" / "lexique.json",
            out_txt=tmp_path / "nlp" / "lexique.txt",
        )
        nlp_mod.NLP_DIR = old

        hapax = [f for f in lexique["formes"] if f["hapax"]]
        non_hapax = [f for f in lexique["formes"] if not f["hapax"]]
        assert all(f["freq"] == 1 for f in hapax)
        assert all(f["freq"] > 1 for f in non_hapax)

    def test_meta_coherent(self, dataset_oracle, tmp_path):
        import src.nlp as nlp_mod
        old = nlp_mod.NLP_DIR
        nlp_mod.NLP_DIR = tmp_path / "nlp"
        (tmp_path / "nlp").mkdir(exist_ok=True)

        lexique = construire_lexique(
            [(dataset_oracle, "oracle")], [], [],
            out_path=tmp_path / "nlp" / "lexique.json",
            out_txt=tmp_path / "nlp" / "lexique.txt",
        )
        nlp_mod.NLP_DIR = old

        assert lexique["meta"]["nb_formes_total"] == len(lexique["formes"])
        assert lexique["meta"]["nb_hapax"] == sum(1 for f in lexique["formes"] if f["hapax"])


# ─── Étape 4 : comparer_lexique_dictionnaire ────────────────────────────────

class TestComparerLexique:
    def _make_lexique(self, formes_freq: dict) -> dict:
        formes = [
            {
                "forme": f,
                "freq": n,
                "manuscrits": ["ms_A"],
                "has_abbreviation_chars": any(est_char for est_char in f if ord(est_char) > 0x024F),
                "hapax": n == 1,
            }
            for f, n in formes_freq.items()
        ]
        formes.sort(key=lambda x: -x["freq"])
        return {"meta": {"nb_formes_total": len(formes), "mode_nettoyage": "diplomatique", "corpus": []}, "formes": formes}

    def test_taux_couverture_plein(self, tmp_path):
        lexique = self._make_lexique({"que": 10, "de": 8, "et": 6})
        dico = {"que", "de", "et", "la", "le"}
        rapport = comparer_lexique_dictionnaire(
            lexique, dico, out_path=tmp_path / "comp.json", freq_min=1
        )
        assert rapport["metriques"]["taux_couverture"] == 1.0

    def test_taux_couverture_partiel(self, tmp_path):
        lexique = self._make_lexique({"que": 10, "ꝯrot": 5, "de": 8})
        dico = {"que", "de"}
        rapport = comparer_lexique_dictionnaire(
            lexique, dico, out_path=tmp_path / "comp.json", freq_min=1
        )
        assert 0 < rapport["metriques"]["taux_couverture"] < 1.0

    def test_classification_abreviation_non_developpee(self, tmp_path):
        lexique = self._make_lexique({"ꝯrot": 5})
        dico: set = set()
        rapport = comparer_lexique_dictionnaire(
            lexique, dico, out_path=tmp_path / "comp.json", freq_min=1
        )
        hors = rapport["formes_hors_dict"]
        assert any(f["categorie"] == "abreviation_non_developpee" for f in hors)

    def test_classification_fragment(self, tmp_path):
        lexique = self._make_lexique({"de": 10})
        dico: set = set()
        rapport = comparer_lexique_dictionnaire(
            lexique, dico, out_path=tmp_path / "comp.json", freq_min=1
        )
        hors = rapport["formes_hors_dict"]
        assert any(f["categorie"] == "fragment" for f in hors)

    def test_fichier_cree(self, tmp_path):
        lexique = self._make_lexique({"que": 5})
        dico = {"que"}
        comparer_lexique_dictionnaire(lexique, dico, out_path=tmp_path / "comp.json", freq_min=1)
        assert (tmp_path / "comp.json").exists()

    def test_freq_min_filtre(self, tmp_path):
        lexique = self._make_lexique({"que": 5, "rare": 1})
        dico: set = set()
        rapport = comparer_lexique_dictionnaire(
            lexique, dico, out_path=tmp_path / "comp.json", freq_min=2
        )
        # "rare" (freq=1) ne doit pas apparaître dans les résultats
        formes_verifiees = [f["forme"] for f in rapport["formes_hors_dict"]]
        assert "rare" not in formes_verifiees


# ─── Étape 4b : construire_dict_depuis_oracle ────────────────────────────────

class TestConstruireDictDepuisOracle:
    def test_retourne_set(self, dataset_oracle, tmp_path):
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        dico = construire_dict_depuis_oracle(
            dataset_oracle, diplo, norm, out_path=tmp_path / "dict.json"
        )
        assert isinstance(dico, set)
        assert len(dico) > 0

    def test_mots_reels_presents(self, dataset_oracle, tmp_path):
        """Les formes oracle nettoyées doivent apparaitre dans le dictionnaire."""
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        dico = construire_dict_depuis_oracle(
            dataset_oracle, diplo, norm, out_path=tmp_path / "dict.json"
        )
        # "de", "ce", "liure", "est", "bone" sont dans l'oracle fixture
        assert "de" in dico
        assert "liure" in dico

    def test_cree_fichier_json(self, dataset_oracle, tmp_path):
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        construire_dict_depuis_oracle(
            dataset_oracle, diplo, norm, out_path=tmp_path / "dict.json"
        )
        assert (tmp_path / "dict.json").exists()
        data = json.loads((tmp_path / "dict.json").read_text(encoding="utf-8"))
        assert "formes" in data
        assert data["nb_formes"] > 0

    def test_inclut_fallback(self, dataset_oracle, tmp_path):
        """Le fallback étendu doit être fusionné, donc 'tout' doit être présent."""
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "regex_rules.json")
        dico = construire_dict_depuis_oracle(
            dataset_oracle, diplo, norm, out_path=tmp_path / "dict.json"
        )
        # 'tout' est dans le fallback médiéval
        assert "tout" in dico


# ─── Étape 5 : calculer_precision_tokens ─────────────────────────────────────

class TestCalculerPrecisionTokens:
    def _make_paired(self, tmp_path, pred_texts, oracle_texts, has_abrev=None):
        """Crée deux fichiers JSON appariés par line_id."""
        if has_abrev is None:
            has_abrev = [False] * len(pred_texts)
        preds = [
            {"line_id": str(i), "transcription": t, "has_abbreviation": a}
            for i, (t, a) in enumerate(zip(pred_texts, has_abrev))
        ]
        oracles = [
            {"line_id": str(i), "transcription": t, "has_abbreviation": a}
            for i, (t, a) in enumerate(zip(oracle_texts, has_abrev))
        ]
        p = tmp_path / "pred.json"
        o = tmp_path / "oracle.json"
        p.write_text(json.dumps(preds), encoding="utf-8")
        o.write_text(json.dumps(oracles), encoding="utf-8")
        return p, o

    def test_precision_parfaite(self, tmp_path):
        """Prediction identique a l'oracle => P=R=F1=1.0."""
        p, o = self._make_paired(tmp_path, ["le roi est bon"], ["le roi est bon"])
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        r = calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert r["metriques"]["precision"] == 1.0
        assert r["metriques"]["rappel"] == 1.0
        assert r["metriques"]["f1"] == 1.0

    def test_precision_nulle(self, tmp_path):
        """Aucun mot en commun => P=R=F1=0.0."""
        p, o = self._make_paired(tmp_path, ["aaaa bbbb"], ["xxxx yyyy"])
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        r = calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert r["metriques"]["precision"] == 0.0
        assert r["metriques"]["f1"] == 0.0

    def test_precision_partielle(self, tmp_path):
        """Moitié des mots corrects => métriques intermédiaires."""
        p, o = self._make_paired(tmp_path, ["le roi mal bon"], ["le roi est bon"])
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        r = calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert 0.0 < r["metriques"]["f1"] < 1.0

    def test_breakdown_par_abreviation(self, tmp_path):
        """Les métriques sont bien séparées avec/sans abréviation."""
        p, o = self._make_paired(
            tmp_path,
            ["le roi", "ꝯmence"],
            ["le roi", "conmence"],
            has_abrev=[False, True],
        )
        diplo = [("ꝯ", "con")]
        norm: list = []
        r = calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert "avec_abreviation" in r["par_abreviation"]
        assert "sans_abreviation" in r["par_abreviation"]
        assert r["par_abreviation"]["sans_abreviation"]["nb_lignes"] == 1

    def test_cree_fichier(self, tmp_path):
        p, o = self._make_paired(tmp_path, ["le roi"], ["le roi"])
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert (tmp_path / "prec.json").exists()

    def test_pires_lignes_capturees(self, tmp_path):
        """Les lignes avec F1 < 0.5 doivent apparaitre dans pires_lignes."""
        p, o = self._make_paired(
            tmp_path,
            ["aaaa bbbb cccc"],
            ["xxxx yyyy zzzz"],
        )
        diplo, norm = charger_regles_nettoyage(out_path=tmp_path / "rr.json")
        r = calculer_precision_tokens(p, o, diplo, norm, out_path=tmp_path / "prec.json")
        assert len(r["pires_lignes"]) >= 1
        assert r["pires_lignes"][0]["f1"] < 0.5
