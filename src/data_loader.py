"""Téléchargement et analyse exploratoire du corpus CREMMA Médiéval."""

import hashlib
import json
import random
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from src.utils import fixer_seeds

# Namespaces PAGE XML
PAGE_NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
# Namespaces ALTO XML
ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

DATA_DIR = Path("data/cremma-medieval")
REPO_URL = "https://github.com/HTR-United/cremma-medieval.git"


def telecharger_corpus(dest: Path = DATA_DIR, shallow: bool = True) -> None:
    """Clone le dépôt CREMMA Médiéval dans le dossier data/.

    Args:
        dest: Chemin de destination du clone.
        shallow: Si True, clone superficiel (--depth 1) pour économiser de l'espace.

    Raises:
        subprocess.CalledProcessError: Si le clone git échoue.

    Example:
        >>> telecharger_corpus()
    """
    if dest.exists():
        print(f"[info] Corpus déjà présent dans {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", REPO_URL, str(dest)] if shallow else ["git", "clone", REPO_URL, str(dest)]
    print(f"[info] Clonage de {REPO_URL} -> {dest} ...")
    subprocess.run(cmd, check=True)
    print("[info] Clonage terminé.")


def lister_manuscrits(data_dir: Path = DATA_DIR) -> list[Path]:
    """Retourne la liste des dossiers de manuscrits dans data/.

    Args:
        data_dir: Racine du corpus cloné.

    Returns:
        Liste de chemins vers les dossiers de manuscrits.

    Example:
        >>> manuscrits = lister_manuscrits()
    """
    return sorted([p for p in (data_dir / "data").iterdir() if p.is_dir()])


def extraire_lignes_xml(xml_path: Path) -> list[str]:
    """Extrait les transcriptions de lignes depuis un fichier ALTO ou PAGE XML.

    Détecte automatiquement le format selon le namespace de la racine.

    Args:
        xml_path: Chemin vers le fichier .xml (ALTO v4 ou PAGE).

    Returns:
        Liste des textes de lignes transcrites (vides exclues).

    Example:
        >>> lignes = extraire_lignes_xml(Path("data/.../page.xml"))
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lignes = []
        tag = root.tag
        if "loc.gov/standards/alto" in tag:
            # ALTO XML : String[@CONTENT] dans chaque TextLine
            for string_el in root.findall(".//alto:TextLine/alto:String", ALTO_NS):
                content = string_el.get("CONTENT", "").strip()
                if content:
                    lignes.append(content)
        else:
            # PAGE XML : TextLine/TextEquiv/Unicode
            for text_equiv in root.findall(".//page:TextLine/page:TextEquiv/page:Unicode", PAGE_NS):
                if text_equiv.text and text_equiv.text.strip():
                    lignes.append(text_equiv.text.strip())
        return lignes
    except ET.ParseError:
        return []


# Alias rétrocompatible
extraire_lignes_page_xml = extraire_lignes_xml


def analyser_manuscrit(manuscrit_dir: Path) -> dict:
    """Analyse les statistiques d'un dossier de manuscrit.

    Args:
        manuscrit_dir: Chemin vers le dossier du manuscrit.

    Returns:
        Dictionnaire avec : nom, nb_pages, nb_lignes, nb_images,
        longueur_moy_ligne, taux_lignes_vides.

    Example:
        >>> stats = analyser_manuscrit(Path("data/cremma-medieval/data/BNF-ms1"))
    """
    # Exclure les fichiers .chocomufin.xml (doublons normalisés des .xml)
    xml_files = [f for f in manuscrit_dir.rglob("*.xml") if ".chocomufin" not in f.name]
    img_files = list(manuscrit_dir.rglob("*.jpg")) + list(manuscrit_dir.rglob("*.png")) + list(manuscrit_dir.rglob("*.tif"))

    toutes_lignes = []
    for xml_path in xml_files:
        toutes_lignes.extend(extraire_lignes_xml(xml_path))

    longueurs = [len(l) for l in toutes_lignes] if toutes_lignes else [0]

    return {
        "nom": manuscrit_dir.name,
        "nb_pages": len(xml_files),
        "nb_images": len(img_files),
        "nb_lignes": len(toutes_lignes),
        "longueur_moy_ligne": round(sum(longueurs) / len(longueurs), 1),
        "longueur_min_ligne": min(longueurs),
        "longueur_max_ligne": max(longueurs),
    }


def analyser_corpus(data_dir: Path = DATA_DIR) -> dict:
    """Analyse complète du corpus : stats globales et par manuscrit.

    Args:
        data_dir: Racine du corpus cloné.

    Returns:
        Dictionnaire avec statistiques globales et liste par manuscrit.

    Example:
        >>> stats = analyser_corpus()
        >>> print(stats["total_lignes"])
    """
    manuscrits = lister_manuscrits(data_dir)
    stats_par_manuscrit = [analyser_manuscrit(m) for m in manuscrits]

    total_lignes = sum(s["nb_lignes"] for s in stats_par_manuscrit)
    total_pages = sum(s["nb_pages"] for s in stats_par_manuscrit)

    return {
        "nb_manuscrits": len(manuscrits),
        "total_pages": total_pages,
        "total_lignes": total_lignes,
        "manuscrits": stats_par_manuscrit,
    }


def sha256_fichier(path: Path) -> str:
    """Calcule le hash SHA-256 d'un fichier.

    Args:
        path: Chemin vers le fichier.

    Returns:
        Hash SHA-256 en hexadécimal.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.

    Example:
        >>> h = sha256_fichier(Path("data/split/test.json"))
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dossier(path: Path, extensions: tuple[str, ...] = (".xml",)) -> str:
    """Calcule un hash SHA-256 canonique sur tous les fichiers d'un dossier.

    Utile pour sceller un split train/val/test de manière reproductible.

    Args:
        path: Chemin vers le dossier.
        extensions: Extensions de fichiers à inclure dans le hash.

    Returns:
        Hash SHA-256 agrégé en hexadécimal.

    Example:
        >>> h = sha256_dossier(Path("data/splits/test"))
    """
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in extensions:
            h.update(f.name.encode())
            h.update(sha256_fichier(f).encode())
    return h.hexdigest()


def creer_split(
    data_dir: Path = DATA_DIR,
    ratio_val: float = 0.15,
    ratio_test: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Répartit les manuscrits en train / val / test sans chevauchement.

    Le split est fait au niveau du manuscrit entier : aucune page d'un même
    manuscrit ne peut se retrouver dans deux ensembles différents.

    Algorithme : mélange aléatoire puis réservation des ``n_val`` manuscrits
    les plus proches de la cible pour val, idem pour test. Le reste va en
    train. Cette approche garantit qu'aucun ensemble n'est vide, même avec
    un petit nombre de manuscrits (ici 14).

    Args:
        data_dir: Racine du corpus cloné.
        ratio_val: Part cible des lignes pour la validation (défaut 0.15).
        ratio_test: Part cible des lignes pour le test (défaut 0.15).
        seed: Graine pour la reproductibilité.

    Returns:
        Dictionnaire avec clés ``train``, ``val``, ``test``, chaque valeur
        étant la liste des chemins de dossiers de manuscrits correspondants.

    Example:
        >>> split = creer_split()
        >>> print(len(split["train"]), "manuscrits en train")
    """
    manuscrits = lister_manuscrits(data_dir)
    # Nombre de manuscrits à réserver pour val et test (minimum 1 chacun)
    n = len(manuscrits)
    n_val = max(1, round(n * ratio_val))
    n_test = max(1, round(n * ratio_test))

    rng = random.Random(seed)
    ordre = list(manuscrits)
    rng.shuffle(ordre)

    # Les derniers manuscrits du mélange vont en test, ceux d'avant en val
    test = sorted(ordre[-n_test:])
    val = sorted(ordre[-(n_val + n_test):-n_test])
    train = sorted(ordre[:-(n_val + n_test)])

    return {"train": train, "val": val, "test": test}


def sauvegarder_split(split: dict[str, list[Path]], out_path: Path) -> dict[str, str]:
    """Sauvegarde le split dans un JSON et calcule les hashes SHA-256 de chaque ensemble.

    Args:
        split: Dictionnaire train/val/test renvoyé par :func:`creer_split`.
        out_path: Chemin du fichier JSON de sortie.

    Returns:
        Dictionnaire ``{ensemble: hash_sha256}``.

    Example:
        >>> hashes = sauvegarder_split(split, Path("data/split.json"))
    """
    serialisable = {k: [str(p) for p in v] for k, v in split.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serialisable, ensure_ascii=False, indent=2), encoding="utf-8")

    hashes: dict[str, str] = {}
    for ensemble, chemins in split.items():
        h = hashlib.sha256()
        for p in sorted(chemins):
            for f in sorted(p.rglob("*.xml")):
                if ".chocomufin" not in f.name:
                    h.update(f.name.encode())
                    h.update(sha256_fichier(f).encode())
        hashes[ensemble] = h.hexdigest()

    return hashes


if __name__ == "__main__":
    fixer_seeds(42)

    # 1. Téléchargement
    telecharger_corpus()

    # 2. Analyse du corpus
    print("\n=== Analyse du corpus CREMMA Médiéval ===")
    stats = analyser_corpus()

    print(f"Manuscrits    : {stats['nb_manuscrits']}")
    print(f"Pages totales : {stats['total_pages']}")
    print(f"Lignes totales: {stats['total_lignes']}")
    print("\nDétail par manuscrit :")
    print(f"{'Nom':<40} {'Pages':>6} {'Lignes':>7} {'Moy.car':>8}")
    print("-" * 65)
    for m in stats["manuscrits"]:
        print(f"{m['nom']:<40} {m['nb_pages']:>6} {m['nb_lignes']:>7} {m['longueur_moy_ligne']:>8}")

    # 3. Export JSON stats
    out = Path("data/corpus_stats.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[info] Stats exportees -> {out}")

    # 4. Split train / val / test
    print("\n=== Split train / val / test ===")
    split = creer_split()
    hashes = sauvegarder_split(split, Path("data/split.json"))

    for ensemble in ("train", "val", "test"):
        noms = [p.name for p in split[ensemble]]
        lignes_ens = sum(
            analyser_manuscrit(p)["nb_lignes"] for p in split[ensemble]
        )
        print(f"{ensemble:5s} ({len(noms):2d} manuscrits, {lignes_ens:6d} lignes) : {', '.join(noms)}")
        print(f"       SHA-256 : {hashes[ensemble]}")

    print("\n[info] Split sauvegarde -> data/split.json")