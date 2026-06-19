"""Tests de la segmentation layout et lignes de texte."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.segmentation import (
    calculer_iou,
    evaluer_segmentation,
    exporter_page_xml,
)

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _carre(cx: int, cy: int, taille: int) -> list[tuple]:
    """Carré centré en (cx, cy) de côté ``taille``."""
    h = taille // 2
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lignes_test() -> list[dict]:
    return [
        {
            "id": "line_0000",
            "baseline": [(10, 50), (200, 50)],
            "polygon": [(10, 40), (200, 40), (200, 60), (10, 60)],
        },
        {
            "id": "line_0001",
            "baseline": [(10, 100), (200, 100)],
            "polygon": [(10, 90), (200, 90), (200, 110), (10, 110)],
        },
    ]


# ---------------------------------------------------------------------------
# Tests IoU
# ---------------------------------------------------------------------------

def test_iou_polygones_identiques():
    """IoU de deux polygones identiques vaut 1.0."""
    poly = _carre(500, 500, 200)
    assert calculer_iou(poly, poly) == pytest.approx(1.0, abs=0.01)


def test_iou_polygones_disjoints():
    """IoU de deux polygones sans chevauchement vaut 0.0."""
    poly1 = _carre(100, 100, 50)
    poly2 = _carre(900, 900, 50)
    assert calculer_iou(poly1, poly2) == pytest.approx(0.0)


def test_iou_polygones_chevauchants():
    """IoU de deux polygones partiellement superposés est dans ]0, 1[."""
    poly1 = _carre(500, 500, 200)
    poly2 = _carre(560, 500, 200)
    iou = calculer_iou(poly1, poly2)
    assert 0.0 < iou < 1.0


def test_iou_polygone_vide_retourne_zero():
    """IoU avec un polygone vide retourne 0.0 sans lever d'exception."""
    poly = _carre(500, 500, 200)
    assert calculer_iou(poly, []) == 0.0
    assert calculer_iou([], poly) == 0.0
    assert calculer_iou([], []) == 0.0


def test_iou_inclusion_totale():
    """IoU d'un grand polygone contenant un petit est strictement entre 0 et 1."""
    grand = _carre(500, 500, 400)
    petit = _carre(500, 500, 100)
    iou = calculer_iou(grand, petit)
    # intersection = petit, union = grand → IoU = aire_petit / aire_grand ≈ 0.0625
    assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# Tests evaluer_segmentation
# ---------------------------------------------------------------------------

def test_evaluer_segmentation_cles_presentes():
    """evaluer_segmentation retourne toutes les clés attendues."""
    poly = _carre(500, 500, 200)
    stats = evaluer_segmentation([poly, poly], [poly, poly])
    for cle in ("iou_moyen", "iou_min", "iou_max", "nb_pairs", "taux_valide", "taux_excellent"):
        assert cle in stats, f"Clé manquante : {cle}"


def test_evaluer_segmentation_valeurs_coherentes():
    """Les statistiques IoU sont dans les plages attendues."""
    poly_ref = _carre(500, 500, 200)
    poly_decale = _carre(560, 500, 200)
    stats = evaluer_segmentation([poly_ref, poly_decale], [poly_ref, poly_ref])
    assert 0.0 <= stats["iou_moyen"] <= 1.0
    assert stats["iou_min"] <= stats["iou_moyen"] <= stats["iou_max"]
    assert stats["nb_pairs"] == 2


def test_evaluer_segmentation_liste_vide():
    """evaluer_segmentation avec des listes vides retourne nb_pairs = 0."""
    stats = evaluer_segmentation([], [])
    assert stats["nb_pairs"] == 0


# ---------------------------------------------------------------------------
# Tests export PAGE XML
# ---------------------------------------------------------------------------

def test_page_xml_parseable(tmp_path, lignes_test):
    """Le PAGE XML généré est parseable sans erreur."""
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=300, image_height=400,
    )
    tree = ET.parse(str(out))
    assert tree.getroot() is not None


def test_page_xml_contient_toutes_lignes(tmp_path, lignes_test):
    """Le PAGE XML contient exactement autant de TextLine que de lignes passées."""
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=300, image_height=400,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    text_lines = root.findall(".//page:TextLine", ns)
    assert len(text_lines) == len(lignes_test)


def test_page_xml_chaque_ligne_a_coords(tmp_path, lignes_test):
    """Chaque TextLine du PAGE XML possède un élément Coords."""
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=300, image_height=400,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    for line_el in root.findall(".//page:TextLine", ns):
        coords = line_el.findall("page:Coords", ns)
        assert len(coords) >= 1, f"TextLine {line_el.get('id')} sans Coords"


def test_page_xml_chaque_ligne_a_baseline(tmp_path, lignes_test):
    """Chaque TextLine du PAGE XML possède un élément Baseline."""
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=300, image_height=400,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    for line_el in root.findall(".//page:TextLine", ns):
        baseline = line_el.findall("page:Baseline", ns)
        assert len(baseline) == 1, f"TextLine {line_el.get('id')} sans Baseline"


def test_page_xml_coordonnees_dans_limites(tmp_path, lignes_test):
    """Toutes les coordonnées Coords du PAGE XML restent dans les dimensions de l'image."""
    W, H = 300, 400
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=W, image_height=H,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    for coords_el in root.findall(".//page:TextLine/page:Coords", ns):
        for pt in coords_el.get("points", "").split():
            x, y = map(int, pt.split(","))
            assert 0 <= x <= W, f"x={x} dépasse la largeur W={W}"
            assert 0 <= y <= H, f"y={y} dépasse la hauteur H={H}"


def test_page_xml_ids_uniques(tmp_path, lignes_test):
    """Chaque TextLine a un attribut id unique."""
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test,
        output_path=out, image_width=300, image_height=400,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    ids = [el.get("id") for el in root.findall(".//page:TextLine", ns)]
    assert len(ids) == len(set(ids)), "Des ids de TextLine sont en double"


def test_page_xml_avec_regions(tmp_path, lignes_test):
    """Quand des régions SAM sont fournies, le PAGE XML les inclut comme TextRegion."""
    regions = [
        {"id": "region_0000", "type": "paragraph", "polygon": [(0, 0), (300, 0), (300, 400), (0, 400)]},
    ]
    out = tmp_path / "page.xml"
    exporter_page_xml(
        Path("page.jpg"), lignes_test, regions=regions,
        output_path=out, image_width=300, image_height=400,
    )
    ns = {"page": PAGE_NS}
    root = ET.parse(str(out)).getroot()
    text_regions = root.findall(".//page:TextRegion", ns)
    assert len(text_regions) >= 1
    region_ids = [r.get("id") for r in text_regions]
    assert "region_0000" in region_ids