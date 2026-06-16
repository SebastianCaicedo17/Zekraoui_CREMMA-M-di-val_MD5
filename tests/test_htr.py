"""Tests du pipeline HTR et de l'analyse des abréviations médiévales."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data_loader import (
    contient_abreviation,
    detecter_abreviations,
    est_char_abreviation,
)
from src.htr import (
    calculer_cer,
    calculer_cer_moyen,
    evaluer_stratifie,
    extraire_lignes_avec_images,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIGNE_AVEC_ABREV = "Ci ꝯm̃ce le liure"   # ꝯ + combining tilde
_LIGNE_SANS_ABREV = "le liure du roi de France"


def _alto_xml_minimal(tmp_path: Path, text: str, hpos=10, vpos=5, w=200, h=30) -> Path:
    """Crée un fichier ALTO XML minimal avec une seule TextLine."""
    alto_ns = "http://www.loc.gov/standards/alto/ns-v4#"
    xml_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{alto_ns}">
  <Layout>
    <Page WIDTH="300" HEIGHT="100" ID="p1">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="300" HEIGHT="100">
        <TextBlock HPOS="0" VPOS="0" WIDTH="300" HEIGHT="100" ID="b1">
          <TextLine ID="l1" HPOS="{hpos}" VPOS="{vpos}" WIDTH="{w}" HEIGHT="{h}">
            <String CONTENT="{text}" HPOS="{hpos}" VPOS="{vpos}" WIDTH="{w}" HEIGHT="{h}"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "page.xml"
    path.write_text(xml_str, encoding="utf-8")
    return path


def _image_blanche(tmp_path: Path, w=300, h=100) -> Path:
    """Crée une image blanche de test."""
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    path = tmp_path / "page.jpg"
    img.save(str(path))
    return path


# ---------------------------------------------------------------------------
# Tests détection d'abréviations (fonctions dans data_loader)
# ---------------------------------------------------------------------------

def test_est_char_abreviation_latin_extended_d():
    """Les caractères Latin Extended-D (ꝯ, ꝓ, ꝑ) sont détectés comme abréviations."""
    assert est_char_abreviation("ꝯ")   # ꝯ LATIN SMALL LETTER CON
    assert est_char_abreviation("ꝓ")   # ꝓ LATIN SMALL LETTER P WITH FLOURISH
    assert est_char_abreviation("ꝑ")   # ꝑ LATIN SMALL LETTER P WITH STROKE


def test_est_char_abreviation_combining_tilde():
    """U+0303 (combining tilde — barre nasale) est détecté comme abréviation."""
    assert est_char_abreviation("̃")


def test_est_char_abreviation_combining_superscript():
    """Les lettres combinantes en exposant (U+0363–U+036F) sont des abréviations."""
    assert est_char_abreviation("ͥ")   # combining latin small letter i (ͥ)
    assert est_char_abreviation("ͣ")   # combining latin small letter a


def test_est_char_abreviation_lettre_normale():
    """Les lettres latines ordinaires ne sont pas des abréviations."""
    for c in "abcdefghijklmnopqrstuvwxyz":
        assert not est_char_abreviation(c), f"'{c}' ne devrait pas être une abréviation"


def test_detecter_abreviations_ligne_abreviee():
    """detecter_abreviations retourne les caractères d'abréviation présents."""
    abrevs = detecter_abreviations(_LIGNE_AVEC_ABREV)
    assert len(abrevs) >= 2
    assert "ꝯ" in abrevs   # ꝯ
    assert "̃" in abrevs   # combining tilde


def test_detecter_abreviations_ligne_normale():
    """detecter_abreviations retourne une liste vide sur du texte ordinaire."""
    assert detecter_abreviations(_LIGNE_SANS_ABREV) == []


def test_contient_abreviation_vrai():
    assert contient_abreviation(_LIGNE_AVEC_ABREV) is True


def test_contient_abreviation_faux():
    assert contient_abreviation(_LIGNE_SANS_ABREV) is False


def test_contient_abreviation_chaine_vide():
    assert contient_abreviation("") is False


# ---------------------------------------------------------------------------
# Tests CER
# ---------------------------------------------------------------------------

def test_calculer_cer_identique():
    """CER = 0.0 pour deux chaînes identiques."""
    assert calculer_cer("bonjour", "bonjour") == pytest.approx(0.0)


def test_calculer_cer_completement_different():
    """CER = 1.0 quand tous les caractères diffèrent (même longueur)."""
    assert calculer_cer("aaaa", "bbbb") == pytest.approx(1.0)


def test_calculer_cer_reference_vide_prediction_vide():
    """CER = 0.0 quand les deux chaînes sont vides."""
    assert calculer_cer("", "") == pytest.approx(0.0)


def test_calculer_cer_reference_vide_prediction_non_vide():
    """CER = 1.0 quand la référence est vide mais la prédiction non."""
    assert calculer_cer("texte", "") == pytest.approx(1.0)


def test_calculer_cer_valeur_connue():
    """CER = 1/7 pour 'bnjour' vs 'bonjour' (1 substitution sur 7 chars)."""
    assert calculer_cer("bnjour", "bonjour") == pytest.approx(1 / 7, rel=0.01)


def test_calculer_cer_moyen_parfait():
    """CER moyen = 0.0 quand toutes les prédictions sont correctes."""
    preds = ["bonjour", "monde"]
    refs = ["bonjour", "monde"]
    assert calculer_cer_moyen(preds, refs) == pytest.approx(0.0)


def test_calculer_cer_moyen_liste_vide():
    """CER moyen = 0.0 sur des listes vides (pas d'erreur)."""
    assert calculer_cer_moyen([], []) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests évaluation stratifiée (cœur de la problématique)
# ---------------------------------------------------------------------------

def test_evaluer_stratifie_cles_presentes():
    """evaluer_stratifie retourne toutes les clés attendues."""
    preds = ["texte"] * 4
    refs = ["texte"] * 4
    flags = [True, True, False, False]
    result = evaluer_stratifie(preds, refs, flags)
    for cle in ("CER_global", "CER_abbrev", "CER_no_abbrev", "delta_CER",
                "nb_lignes_abrev", "nb_lignes_no_abbrev"):
        assert cle in result, f"Clé manquante : {cle}"


def test_evaluer_stratifie_predictions_parfaites():
    """delta_CER = 0 quand toutes les prédictions sont parfaites."""
    refs = ["bonjour", "monde", "ꝯm̃ce"]
    flags = [False, False, True]
    result = evaluer_stratifie(refs.copy(), refs, flags)
    assert result["CER_global"] == pytest.approx(0.0)
    assert result["delta_CER"] == pytest.approx(0.0)


def test_evaluer_stratifie_pire_sur_abreviations():
    """CER_abbrev > CER_no_abbrev quand les erreurs se concentrent sur les abréviations."""
    preds = ["bonjour", "monde", "???", "???"]          # lignes 2-3 mal prédites
    refs = ["bonjour", "monde", "ꝯm̃ce", "ꝑres"]
    flags = [False, False, True, True]
    result = evaluer_stratifie(preds, refs, flags)
    assert result["CER_abbrev"] > result["CER_no_abbrev"]
    assert result["delta_CER"] > 0


def test_evaluer_stratifie_pas_de_lignes_abreviees():
    """CER_abbrev = None quand il n'y a aucune ligne abrégée."""
    preds = ["a", "b"]
    refs = ["a", "b"]
    flags = [False, False]
    result = evaluer_stratifie(preds, refs, flags)
    assert result["CER_abbrev"] is None
    assert result["delta_CER"] is None
    assert result["nb_lignes_abrev"] == 0


def test_evaluer_stratifie_compte_correct():
    """Les compteurs nb_lignes_abrev et nb_lignes_no_abbrev sont justes."""
    flags = [True, False, True, False, True]
    preds = refs = ["x"] * 5
    result = evaluer_stratifie(preds, refs, flags)
    assert result["nb_lignes_abrev"] == 3
    assert result["nb_lignes_no_abbrev"] == 2


# ---------------------------------------------------------------------------
# Tests extraction lignes depuis ALTO XML
# ---------------------------------------------------------------------------

def test_extraire_lignes_alto_xml(tmp_path):
    """extraire_lignes_avec_images extrait le texte et le crop depuis un ALTO XML."""
    xml_path = _alto_xml_minimal(tmp_path, _LIGNE_SANS_ABREV)
    img_path = _image_blanche(tmp_path)

    lignes = extraire_lignes_avec_images(xml_path, img_path, "train", "ms_test")

    assert len(lignes) == 1
    assert lignes[0]["text"] == _LIGNE_SANS_ABREV
    assert lignes[0]["split"] == "train"
    assert lignes[0]["manuscrit"] == "ms_test"
    assert isinstance(lignes[0]["image"], Image.Image)


def test_extraire_lignes_flag_abreviation(tmp_path):
    """Le flag has_abbreviation est correctement positionné selon le texte."""
    xml_abrev = _alto_xml_minimal(tmp_path / "abrev", _LIGNE_AVEC_ABREV)
    (tmp_path / "abrev").mkdir(exist_ok=True)
    xml_abrev = _alto_xml_minimal(tmp_path / "abrev", _LIGNE_AVEC_ABREV)
    img_abrev = _image_blanche(tmp_path / "abrev")
    lignes = extraire_lignes_avec_images(xml_abrev, img_abrev, "train", "ms_test")
    assert lignes[0]["has_abbreviation"] is True

    xml_norm = _alto_xml_minimal(tmp_path / "norm", _LIGNE_SANS_ABREV)
    (tmp_path / "norm").mkdir(exist_ok=True)
    xml_norm = _alto_xml_minimal(tmp_path / "norm", _LIGNE_SANS_ABREV)
    img_norm = _image_blanche(tmp_path / "norm")
    lignes = extraire_lignes_avec_images(xml_norm, img_norm, "train", "ms_test")
    assert lignes[0]["has_abbreviation"] is False


def test_extraire_lignes_image_introuvable(tmp_path):
    """extraire_lignes_avec_images retourne [] si l'image est absente."""
    xml_path = _alto_xml_minimal(tmp_path, "texte")
    lignes = extraire_lignes_avec_images(xml_path, tmp_path / "absent.jpg", "train", "ms")
    assert lignes == []


def test_extraire_lignes_xml_corrompu(tmp_path):
    """extraire_lignes_avec_images retourne [] si le XML est invalide."""
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text("<broken>", encoding="utf-8")
    img_path = _image_blanche(tmp_path)
    lignes = extraire_lignes_avec_images(xml_path, img_path, "train", "ms")
    assert lignes == []