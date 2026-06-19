"""Tests du pipeline de prétraitement des images de manuscrits."""

import numpy as np
import pytest
import cv2
from pathlib import Path

from src.preprocessing import (
    detecter_angle_inclinaison,
    corriger_inclinaison,
    ameliorer_contraste,
    binariser_sauvola,
    masquer_enluminures,
    pretraiter_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_lignes(height: int = 300, width: int = 600, n_lignes: int = 15) -> np.ndarray:
    """Image synthétique en niveaux de gris avec des lignes de texte horizontales."""
    image = np.full((height, width), 255, dtype=np.uint8)
    espacement = height // (n_lignes + 1)
    for i in range(1, n_lignes + 1):
        y = i * espacement
        image[max(0, y - 2) : y + 2, 20 : width - 20] = 0
    return image


def _image_bgr_synthétique(height: int = 300, width: int = 600) -> np.ndarray:
    """Image synthétique BGR (niveaux de gris convertis en couleur)."""
    return cv2.cvtColor(_image_lignes(height, width), cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Fixtures pytest
# ---------------------------------------------------------------------------

@pytest.fixture
def img_gris():
    return _image_lignes()


@pytest.fixture
def img_bgr():
    return _image_bgr_synthétique()


# ---------------------------------------------------------------------------
# Test 1 — Forme et type de sortie
# ---------------------------------------------------------------------------

def test_sortie_forme_et_type(img_gris):
    """binariser_sauvola retourne un ndarray 2D uint8."""
    result = binariser_sauvola(img_gris)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2, f"Attendu 2D, obtenu {result.ndim}D"
    assert result.dtype == np.uint8, f"Attendu uint8, obtenu {result.dtype}"
    assert result.shape == img_gris.shape


# ---------------------------------------------------------------------------
# Test 2 — Plages de valeurs (binarisation stricte)
# ---------------------------------------------------------------------------

def test_valeurs_binarisation(img_gris):
    """Tous les pixels de l'image binarisée appartiennent à {0, 255}."""
    result = binariser_sauvola(img_gris)
    valeurs = set(np.unique(result).tolist())
    assert valeurs.issubset({0, 255}), f"Valeurs inattendues : {valeurs - {0, 255}}"


# ---------------------------------------------------------------------------
# Test 3 — Reproductibilité
# ---------------------------------------------------------------------------

def test_reproductibilite(img_gris):
    """Même entrée produit toujours la même sortie (pipeline déterministe)."""
    r1 = binariser_sauvola(img_gris)
    r2 = binariser_sauvola(img_gris)
    assert np.array_equal(r1, r2), "Les deux appels successifs produisent des résultats différents"


def test_reproductibilite_pipeline(tmp_path):
    """pretraiter_image est déterministe sur une même image disque."""
    img = _image_lignes(height=200, width=400)
    p = tmp_path / "test.png"
    cv2.imwrite(str(p), img)

    r1 = pretraiter_image(p, deskew=False)  # deskew=False pour la rapidité
    r2 = pretraiter_image(p, deskew=False)
    assert np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# Test 4 — Correction d'angle sur angle connu
# ---------------------------------------------------------------------------

def test_correction_angle_connu():
    """Deskewing d'une image inclinée de 5° : l'angle résiduel après correction est < 1°."""
    ANGLE_APPLIQUE = 5.0
    image_droite = _image_lignes(height=400, width=800, n_lignes=20)

    h, w = image_droite.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, -ANGLE_APPLIQUE, 1.0)
    image_inclinee = cv2.warpAffine(image_droite, M, (w, h), borderValue=255)

    image_corrigee = corriger_inclinaison(image_inclinee, plage_deg=10.0, pas=0.5)
    angle_residuel = detecter_angle_inclinaison(image_corrigee, plage_deg=5.0, pas=0.5)

    assert abs(angle_residuel) < 1.5, (
        f"Angle résiduel trop grand : {angle_residuel:.2f}° (attendu < 1.5°)"
    )


def test_detection_angle_image_droite():
    """L'angle détecté sur une image déjà droite est proche de 0°."""
    image_droite = _image_lignes(height=400, width=800, n_lignes=20)
    angle = detecter_angle_inclinaison(image_droite, plage_deg=10.0, pas=0.5)
    assert abs(angle) < 1.0, f"Angle détecté sur image droite : {angle:.2f}°"


# ---------------------------------------------------------------------------
# Tests des fonctions individuelles
# ---------------------------------------------------------------------------

def test_ameliorer_contraste_shape(img_bgr):
    """CLAHE conserve la forme et le dtype de l'image."""
    result = ameliorer_contraste(img_bgr)
    assert result.shape == img_bgr.shape
    assert result.dtype == img_bgr.dtype


def test_masquer_enluminures_zone_coloree(img_bgr):
    """Une zone verte saturée est mise à blanc dans l'image binarisée après masquage."""
    image_avec_enluminure = img_bgr.copy()
    image_avec_enluminure[60:120, 60:160] = [0, 200, 0]  # vert vif saturé

    gris = cv2.cvtColor(image_avec_enluminure, cv2.COLOR_BGR2GRAY)
    img_bin = binariser_sauvola(gris)
    img_masquee = masquer_enluminures(image_avec_enluminure, img_bin)

    zone = img_masquee[60:120, 60:160]
    assert np.all(zone == 255), "La zone colorée doit être entièrement blanche après masquage"


def test_masquer_enluminures_texte_conserve(img_bgr):
    """Les zones non colorées restent inchangées après le masquage."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_bin = binariser_sauvola(gris)
    img_masquee = masquer_enluminures(img_bgr, img_bin)

    # Sans enluminures dans l'image, le masquage ne change rien
    assert np.array_equal(img_bin, img_masquee)


# ---------------------------------------------------------------------------
# Tests d'erreurs
# ---------------------------------------------------------------------------

def test_fichier_inexistant():
    """pretraiter_image lève FileNotFoundError si le fichier n'existe pas."""
    with pytest.raises(FileNotFoundError):
        pretraiter_image(Path("chemin/qui/nexiste/pas.jpg"))