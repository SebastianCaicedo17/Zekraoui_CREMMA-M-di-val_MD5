"""Segmentation layout et lignes de texte pour les manuscrits médiévaux."""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# SAM est optionnel (modèle lourd, ~375 MB)
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINTS_DIR = Path("checkpoints")


# ---------------------------------------------------------------------------
# Téléchargement des poids SAM
# ---------------------------------------------------------------------------

def telecharger_poids_sam(dest: Path = CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth") -> Path:
    """Télécharge les poids SAM ViT-B (~375 MB) si absents.

    Args:
        dest: Chemin de destination du fichier .pth.

    Returns:
        Chemin vers le fichier de poids téléchargé.

    Example:
        >>> checkpoint = telecharger_poids_sam()
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[info] Téléchargement SAM ViT-B -> {dest} (~375 MB) ...")
    urllib.request.urlretrieve(SAM_VIT_B_URL, str(dest))
    print("[info] Téléchargement terminé.")
    return dest


# ---------------------------------------------------------------------------
# Segmentation layout (SAM)
# ---------------------------------------------------------------------------

def segmenter_layout_sam(
    image_bgr: np.ndarray,
    checkpoint: Optional[Path] = None,
    model_type: str = "vit_b",
    min_area_ratio: float = 0.005,
) -> list[dict]:
    """Détecte les régions de mise en page avec SAM (Segment Anything Model).

    Génère automatiquement des masques, filtre par taille minimale et retourne
    les polygones englobants de chaque région candidate.

    Args:
        image_bgr: Image couleur BGR (uint8, ndarray).
        checkpoint: Chemin vers les poids SAM. Téléchargé automatiquement si absent.
        model_type: Variante SAM à utiliser (``vit_b``, ``vit_l``, ``vit_h``).
        min_area_ratio: Fraction minimale de l'image pour qu'une région soit retenue.

    Returns:
        Liste de dicts ``{id, type, polygon, score}`` triés par score décroissant.

    Raises:
        ImportError: Si ``segment_anything`` n'est pas installé.

    Example:
        >>> regions = segmenter_layout_sam(image_bgr)
    """
    if not _SAM_AVAILABLE:
        raise ImportError(
            "Le paquet segment-anything n'est pas disponible. "
            "Installez-le avec : pip install segment-anything"
        )

    if checkpoint is None:
        checkpoint = telecharger_poids_sam()

    import torch  # importé ici pour ne pas planter si absent lors de l'import du module

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device="cpu")

    h, w = image_bgr.shape[:2]
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=int(h * w * min_area_ratio),
    )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masques = generator.generate(image_rgb)

    regions = []
    for i, masque in enumerate(sorted(masques, key=lambda m: m["predicted_iou"], reverse=True)):
        x, y, bw, bh = masque["bbox"]  # format XYWH
        polygon = [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh)]
        regions.append({
            "id": f"region_{i:04d}",
            "type": "paragraph",
            "polygon": polygon,
            "score": float(masque["predicted_iou"]),
        })

    return regions


# ---------------------------------------------------------------------------
# Segmentation des lignes (Kraken BLLA)
# ---------------------------------------------------------------------------

def segmenter_lignes_kraken(image_path: Path) -> list[dict]:
    """Extrait les lignes de texte et leurs polygones avec Kraken BLLA.

    Kraken BLLA (Baseline Layout Analysis) détecte les lignes de base et
    calcule les polygones de contour de chaque ligne de texte.

    Args:
        image_path: Chemin vers l'image source (JPEG, PNG, TIFF).

    Returns:
        Liste de dicts ``{id, baseline, polygon}`` pour chaque ligne détectée.
        ``baseline`` et ``polygon`` sont des listes de tuples ``(x, y)``.

    Example:
        >>> lignes = segmenter_lignes_kraken(Path("data/.../page.jpg"))
    """
    from kraken import blla  # import différé : Kraken est lourd à charger

    img = Image.open(str(image_path)).convert("RGB")
    seg = blla.segment(img)

    lignes = []
    for i, line in enumerate(seg.lines):
        baseline = [tuple(pt) for pt in (line.baseline or [])]
        polygon = [tuple(pt) for pt in (line.boundary or [])]
        lignes.append({
            "id": f"line_{i:04d}",
            "baseline": baseline,
            "polygon": polygon,
        })

    return lignes


# ---------------------------------------------------------------------------
# Export PAGE XML
# ---------------------------------------------------------------------------

def _points_en_chaine(points: list[tuple]) -> str:
    """Convertit une liste de (x, y) en chaîne PAGE XML ``x1,y1 x2,y2 ...``."""
    return " ".join(f"{int(x)},{int(y)}" for x, y in points)


def exporter_page_xml(
    image_path: Path,
    lignes: list[dict],
    regions: Optional[list[dict]] = None,
    output_path: Optional[Path] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Path:
    """Exporte la segmentation au format PAGE XML standard.

    Crée un fichier ``.xml`` compatible avec eScriptorium / Transkribus contenant
    les régions de mise en page, les polygones et les lignes de base.

    Args:
        image_path: Chemin vers l'image source (utilisé pour le nom de fichier).
        lignes: Liste de dicts ``{id, baseline, polygon}`` produite par Kraken.
        regions: Liste optionnelle de dicts ``{id, type, polygon}`` depuis SAM.
            Si absent, une région unique couvre la page entière.
        output_path: Chemin de sortie. Défaut : ``segmentations/<stem>.xml``.
        image_width: Largeur en pixels (lu dans l'image si omis).
        image_height: Hauteur en pixels (lu dans l'image si omis).

    Returns:
        Chemin du fichier PAGE XML créé.

    Example:
        >>> path = exporter_page_xml(Path("page.jpg"), lignes)
    """
    if output_path is None:
        output_path = Path("segmentations") / f"{image_path.stem}.xml"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image_width is None or image_height is None:
        with Image.open(str(image_path)) as img:
            image_width, image_height = img.size

    ET.register_namespace("", PAGE_NS)
    root = ET.Element(
        f"{{{PAGE_NS}}}PcGts",
        attrib={
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                f"{PAGE_NS} {PAGE_NS}/pagecontent.xsd"
            ),
        },
    )

    meta = ET.SubElement(root, f"{{{PAGE_NS}}}Metadata")
    ET.SubElement(meta, f"{{{PAGE_NS}}}Creator").text = "HTR-Pipeline-MD5"
    ET.SubElement(meta, f"{{{PAGE_NS}}}Created").text = datetime.now(timezone.utc).isoformat()

    page_el = ET.SubElement(
        root,
        f"{{{PAGE_NS}}}Page",
        attrib={
            "imageFilename": image_path.name,
            "imageWidth": str(image_width),
            "imageHeight": str(image_height),
        },
    )

    page_poly = [(0, 0), (image_width, 0), (image_width, image_height), (0, image_height)]

    if regions:
        for region in regions:
            reg_el = ET.SubElement(
                page_el,
                f"{{{PAGE_NS}}}TextRegion",
                attrib={"id": region["id"], "type": region.get("type", "paragraph")},
            )
            ET.SubElement(
                reg_el, f"{{{PAGE_NS}}}Coords",
                attrib={"points": _points_en_chaine(region["polygon"])},
            )
    else:
        reg_el = ET.SubElement(
            page_el, f"{{{PAGE_NS}}}TextRegion",
            attrib={"id": "region_main", "type": "paragraph"},
        )
        ET.SubElement(
            reg_el, f"{{{PAGE_NS}}}Coords",
            attrib={"points": _points_en_chaine(page_poly)},
        )

    # Toutes les lignes dans la dernière région (ou région principale)
    for ligne in lignes:
        line_el = ET.SubElement(
            reg_el, f"{{{PAGE_NS}}}TextLine", attrib={"id": ligne["id"]}
        )
        if ligne.get("polygon"):
            ET.SubElement(
                line_el, f"{{{PAGE_NS}}}Coords",
                attrib={"points": _points_en_chaine(ligne["polygon"])},
            )
        if ligne.get("baseline"):
            ET.SubElement(
                line_el, f"{{{PAGE_NS}}}Baseline",
                attrib={"points": _points_en_chaine(ligne["baseline"])},
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), encoding="UTF-8", xml_declaration=True)

    return output_path


# ---------------------------------------------------------------------------
# Calcul IoU
# ---------------------------------------------------------------------------

def calculer_iou(
    poly1: list[tuple],
    poly2: list[tuple],
    canvas_size: tuple[int, int] = (1000, 1000),
) -> float:
    """Calcule l'IoU (Intersection over Union) entre deux polygones.

    Rasterise les polygones sur un canvas et calcule le ratio pixel à pixel.

    Args:
        poly1: Premier polygone — liste de tuples ``(x, y)``.
        poly2: Second polygone — liste de tuples ``(x, y)``.
        canvas_size: ``(height, width)`` du canvas de rasterisation (défaut 1000×1000).

    Returns:
        Score IoU entre 0.0 et 1.0.

    Example:
        >>> iou = calculer_iou([(0,0),(100,0),(100,50),(0,50)], [(50,0),(150,0),(150,50),(50,50)])
    """
    if not poly1 or not poly2:
        return 0.0

    h, w = canvas_size
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(mask1, [np.array(poly1, dtype=np.int32).reshape(-1, 1, 2)], 1)
    cv2.fillPoly(mask2, [np.array(poly2, dtype=np.int32).reshape(-1, 1, 2)], 1)

    intersection = int(np.logical_and(mask1, mask2).sum())
    union = int(np.logical_or(mask1, mask2).sum())

    return float(intersection / union) if union > 0 else 0.0


def evaluer_segmentation(
    predictions: list[list[tuple]],
    references: list[list[tuple]],
    canvas_size: tuple[int, int] = (1000, 1000),
) -> dict:
    """Calcule les statistiques IoU sur un ensemble de prédictions vs références.

    Apparie prédictions et références dans l'ordre (même index = même ligne).

    Args:
        predictions: Liste de polygones prédits (un par ligne).
        references: Liste de polygones de référence (même longueur).
        canvas_size: Taille du canvas de rasterisation pour :func:`calculer_iou`.

    Returns:
        Dict avec ``iou_moyen``, ``iou_min``, ``iou_max``, ``nb_pairs``,
        ``taux_valide`` (fraction IoU ≥ 0.75) et ``taux_excellent`` (IoU ≥ 0.85).

    Example:
        >>> stats = evaluer_segmentation(preds, refs)
        >>> print(stats["iou_moyen"])
    """
    if not predictions or not references:
        return {"iou_moyen": 0.0, "nb_pairs": 0}

    scores = [
        calculer_iou(p, r, canvas_size)
        for p, r in zip(predictions, references)
    ]

    return {
        "iou_moyen": round(float(np.mean(scores)), 4),
        "iou_min": round(float(np.min(scores)), 4),
        "iou_max": round(float(np.max(scores)), 4),
        "nb_pairs": len(scores),
        "taux_valide": round(sum(s >= 0.75 for s in scores) / len(scores), 4),
        "taux_excellent": round(sum(s >= 0.85 for s in scores) / len(scores), 4),
    }


# ---------------------------------------------------------------------------
# Point d'entrée : démo sur une image réelle du corpus
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from src.utils import fixer_seeds

    fixer_seeds(42)

    data_dir = Path("data/cremma-medieval/data")
    if not data_dir.exists():
        print("[erreur] Corpus introuvable. Lancez d'abord : python -m src.data_loader")
        sys.exit(1)

    images = sorted(data_dir.rglob("*.jpg"))
    if not images:
        print("[erreur] Aucune image trouvée.")
        sys.exit(1)

    image_path = images[0]
    print(f"[info] Segmentation de : {image_path.name}")

    # Segmentation Kraken BLLA
    print("[info] Kraken BLLA en cours ...")
    lignes = segmenter_lignes_kraken(image_path)
    print(f"[info] {len(lignes)} lignes détectées")

    if lignes:
        print(f"       Exemple ligne 0 — baseline : {lignes[0]['baseline'][:3]} ...")
        print(f"       Exemple ligne 0 — polygone : {lignes[0]['polygon'][:3]} ...")

    # Export PAGE XML
    xml_path = exporter_page_xml(image_path, lignes)
    print(f"[info] PAGE XML exporté -> {xml_path}")

    # Statistiques rapides
    stats = {
        "image": image_path.name,
        "nb_lignes": len(lignes),
        "nb_lignes_avec_polygone": sum(1 for l in lignes if l["polygon"]),
        "nb_lignes_avec_baseline": sum(1 for l in lignes if l["baseline"]),
    }
    print(f"[info] Stats : {json.dumps(stats, indent=2)}")