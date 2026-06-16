"""Pipeline de prétraitement des images de manuscrits médiévaux."""

import cv2
import numpy as np
from pathlib import Path
from skimage.filters import threshold_sauvola


def detecter_angle_inclinaison(
    image_gris: np.ndarray,
    plage_deg: float = 10.0,
    pas: float = 0.5,
) -> float:
    """Détecte l'angle d'inclinaison dominant par la méthode des projections horizontales.

    L'angle retourné est celui qu'il faut appliquer pour redresser l'image :
    la rotation par cet angle maximise la variance des sommes de pixels par
    ligne (les lignes de texte horizontales créent des pics nets).

    Args:
        image_gris: Image en niveaux de gris (uint8, 2D).
        plage_deg: Plage de recherche en degrés (±plage_deg). Défaut ±10°.
        pas: Pas angulaire en degrés. Défaut 0.5°.

    Returns:
        Angle d'inclinaison en degrés à appliquer pour corriger l'image.

    Example:
        >>> angle = detecter_angle_inclinaison(img_gris)
    """
    _, binaire = cv2.threshold(image_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binaire.shape
    centre = (w // 2, h // 2)

    meilleur_angle = 0.0
    meilleure_variance = -1.0

    for angle in np.arange(-plage_deg, plage_deg + pas, pas):
        M = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotee = cv2.warpAffine(binaire, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        projection = np.sum(rotee, axis=1, dtype=np.float64)
        variance = float(np.var(projection))
        if variance > meilleure_variance:
            meilleure_variance = variance
            meilleur_angle = float(angle)

    return meilleur_angle


def corriger_inclinaison(
    image: np.ndarray,
    plage_deg: float = 10.0,
    pas: float = 0.5,
) -> np.ndarray:
    """Détecte et corrige l'angle d'inclinaison d'une image.

    Fonctionne sur les images en niveaux de gris (2D) et en couleur BGR (3D).
    Le fond introduit par la rotation est blanc.

    Args:
        image: Image uint8, shape (H, W) pour gris ou (H, W, 3) pour BGR.
        plage_deg: Plage de recherche en degrés (±plage_deg). Défaut ±10°.
        pas: Pas angulaire en degrés. Défaut 0.5°.

    Returns:
        Image redressée, même shape et dtype que l'entrée, fond blanc.

    Example:
        >>> img_droite = corriger_inclinaison(img_gris)
    """
    if image.ndim == 3:
        gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        border_value: int | tuple = (255, 255, 255)
    else:
        gris = image
        border_value = 255

    angle = detecter_angle_inclinaison(gris, plage_deg, pas)

    h, w = gris.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=border_value)


def ameliorer_contraste(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """Améliore le contraste local avec CLAHE appliqué au canal luminance (espace LAB).

    Laisse les canaux couleur intacts, ce qui préserve la détectabilité des
    enluminures dans l'étape suivante.

    Args:
        image_bgr: Image couleur BGR (uint8).
        clip_limit: Limite de contraste adaptatif pour éviter la sur-amplification.
        tile_size: Taille des tuiles pour l'histogramme local (tile_size × tile_size).

    Returns:
        Image BGR avec contraste amélioré (uint8).

    Example:
        >>> img_contraste = ameliorer_contraste(img_bgr)
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def binariser_sauvola(
    image_gris: np.ndarray,
    window_size: int = 25,
    k: float = 0.2,
) -> np.ndarray:
    """Binarise une image en niveaux de gris par la méthode de Sauvola.

    Calcule un seuil local T(x,y) = mean * (1 + k * (std/R - 1)) pour chaque
    pixel, robuste aux variations d'illumination inhomogène des manuscrits.

    Args:
        image_gris: Image en niveaux de gris (uint8, 2D).
        window_size: Taille de la fenêtre locale en pixels (doit être impair).
        k: Paramètre de sensibilité Sauvola (défaut 0.2).

    Returns:
        Image binarisée uint8, 2D, pixels ∈ {0, 255}.

    Example:
        >>> img_bin = binariser_sauvola(img_gris)
    """
    seuil = threshold_sauvola(image_gris, window_size=window_size, k=k)
    return (image_gris < seuil).astype(np.uint8) * 255


def masquer_enluminures(
    image_bgr: np.ndarray,
    image_bin: np.ndarray,
    seuil_saturation: int = 60,
    dilatation: int = 15,
) -> np.ndarray:
    """Masque les zones enluminées (très colorées) dans l'image binarisée.

    Détecte les régions à forte saturation en espace HSV — miniatures, lettrines
    décorées, initiales peintes — et les remplace par du blanc pour qu'elles ne
    soient pas interprétées comme du texte par le modèle HTR.

    Args:
        image_bgr: Image couleur BGR originale (uint8).
        image_bin: Image binarisée correspondante (uint8, pixels ∈ {0, 255}).
        seuil_saturation: Seuil de saturation HSV (0–255) au-dessus duquel un
            pixel est considéré comme enluminé. Défaut 60.
        dilatation: Rayon de dilatation du masque en pixels pour couvrir les
            bords de transition. Défaut 15.

    Returns:
        Image binarisée avec zones enluminées mises à blanc (uint8).

    Example:
        >>> img_masquee = masquer_enluminures(img_bgr, img_bin)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    masque = (saturation > seuil_saturation).astype(np.uint8) * 255

    if dilatation > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilatation + 1, 2 * dilatation + 1)
        )
        masque = cv2.dilate(masque, kernel)

    resultat = image_bin.copy()
    resultat[masque > 0] = 255
    return resultat


def pretraiter_image(
    chemin: Path,
    deskew: bool = True,
    clahe: bool = True,
    sauvola: bool = True,
    masquer: bool = True,
) -> np.ndarray:
    """Pipeline complet de prétraitement d'une image de manuscrit médiéval.

    Enchaîne dans l'ordre : correction d'inclinaison → amélioration CLAHE →
    binarisation Sauvola → masquage des enluminures. Chaque étape est
    activable/désactivable indépendamment.

    Args:
        chemin: Chemin vers l'image source (JPEG, PNG, TIFF).
        deskew: Activer la correction d'inclinaison.
        clahe: Activer l'amélioration de contraste CLAHE.
        sauvola: Activer la binarisation Sauvola (sinon seuillage Otsu global).
        masquer: Activer le masquage des zones enluminées.

    Returns:
        Image binarisée uint8, shape (H, W), pixels ∈ {0, 255}.

    Raises:
        FileNotFoundError: Si le fichier image n'existe pas.
        ValueError: Si l'image ne peut pas être décodée par OpenCV.

    Example:
        >>> img = pretraiter_image(Path("data/cremma-medieval/data/bnf_fr_1728/page.jpg"))
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Image introuvable : {chemin}")

    image_bgr = cv2.imread(str(chemin))
    if image_bgr is None:
        raise ValueError(f"Impossible de décoder : {chemin}")

    # 1. Correction d'inclinaison (détection sur gris, correction sur BGR)
    if deskew:
        image_bgr = corriger_inclinaison(image_bgr)

    # 2. Amélioration du contraste CLAHE
    if clahe:
        image_bgr = ameliorer_contraste(image_bgr)

    # 3. Binarisation
    image_gris = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if sauvola:
        image_bin = binariser_sauvola(image_gris)
    else:
        _, image_bin = cv2.threshold(image_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Masquage des enluminures
    if masquer:
        image_bin = masquer_enluminures(image_bgr, image_bin)

    return image_bin


if __name__ == "__main__":
    import sys
    from src.utils import fixer_seeds

    fixer_seeds(42)

    data_dir = Path("data/cremma-medieval/data")
    if not data_dir.exists():
        print("[erreur] Corpus introuvable. Lancez d'abord : python -m src.data_loader")
        sys.exit(1)

    # Prendre la première image trouvée dans le corpus
    images = list(data_dir.rglob("*.jpg"))
    if not images:
        print("[erreur] Aucune image .jpg trouvée dans le corpus.")
        sys.exit(1)

    chemin = images[0]
    print(f"[info] Prétraitement de : {chemin.name}")
    resultat = pretraiter_image(chemin)
    print(f"[info] Image prétraitée : shape={resultat.shape}, dtype={resultat.dtype}")
    print(f"[info] Valeurs uniques : {sorted(set(resultat.flatten().tolist()[:1000]))}")

    out = Path("data/pretraitement_demo.png")
    cv2.imwrite(str(out), resultat)
    print(f"[info] Sauvegardé -> {out}")