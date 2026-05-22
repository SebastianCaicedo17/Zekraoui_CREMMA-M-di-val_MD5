"""Téléchargement et analyse exploratoire du corpus CREMMA Médiéval."""

import hashlib
import json
import random
import subprocess
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
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


# ---------------------------------------------------------------------------
# Détection des abréviations médiévales (Unicode)
# ---------------------------------------------------------------------------

# Latin Extended-D : formes abrégées spécialisées (ꝑ, ꝓ, ꝯ, ꝰ…)
_ABREV_RANGES = [
    (0xA720, 0xA7FF),   # Latin Extended-D
    (0x0363, 0x036F),   # Lettres combinantes en exposant (ͣ ͤ ͥ ͦ ͧ…)
]
# Signes diacritiques combinants utilisés comme marques d'abréviation
_ABREV_COMBINING = {
    0x0303,  # Tilde combinant — barre nasale (m̃, ñ médiéval)
    0x0305,  # Macron combinant — barre de suspension
    0x033E,  # Tilde vertical combinant — marque d'abréviation
    0x0360,  # Double tilde combinant
    0x0361,  # Double brève renversée combinante
}


def est_char_abreviation(c: str) -> bool:
    """Retourne True si le caractère est un marqueur d'abréviation médiévale.

    Args:
        c: Caractère Unicode à tester (longueur 1).

    Returns:
        True si c'est un caractère d'abréviation médiévale.

    Example:
        >>> est_char_abreviation('ꝯ')
        True
        >>> est_char_abreviation('a')
        False
    """
    code = ord(c)
    if code in _ABREV_COMBINING:
        return True
    return any(start <= code <= end for start, end in _ABREV_RANGES)


def detecter_abreviations(ligne: str) -> list[str]:
    """Retourne la liste des caractères d'abréviation présents dans une ligne.

    Args:
        ligne: Texte transcrit d'une ligne de manuscrit.

    Returns:
        Liste des caractères d'abréviation trouvés (peut contenir des doublons).

    Example:
        >>> detecter_abreviations("Ci ꝯm̃ce le liure")
        ['ꝯ', '̃']
    """
    return [c for c in ligne if est_char_abreviation(c)]


def contient_abreviation(ligne: str) -> bool:
    """Retourne True si la ligne contient au moins un caractère d'abréviation.

    Args:
        ligne: Texte transcrit d'une ligne de manuscrit.

    Returns:
        True si la ligne contient une ou plusieurs abréviations médiévales.

    Example:
        >>> contient_abreviation("Ci ꝯm̃ce le liure")
        True
        >>> contient_abreviation("le liure du roi")
        False
    """
    return any(est_char_abreviation(c) for c in ligne)


def statistiques_abreviations(data_dir: Path = DATA_DIR) -> dict:
    """Calcule les statistiques d'abréviations dans l'ensemble du corpus.

    Pour chaque manuscrit et globalement, mesure la proportion de lignes
    contenant des abréviations et les caractères les plus fréquents.

    Args:
        data_dir: Racine du corpus cloné.

    Returns:
        Dictionnaire avec stats globales et par manuscrit :
        ``nb_lignes``, ``nb_lignes_abrev``, ``taux_abrev``,
        ``top_chars`` (liste des 10 caractères les plus fréquents avec nom Unicode).

    Example:
        >>> stats = statistiques_abreviations()
        >>> print(stats["taux_abrev_global"])
    """
    manuscrits = lister_manuscrits(data_dir)
    compteur_global: Counter = Counter()
    total_lignes = total_avec_abrev = 0
    stats_par_manuscrit = []

    for manuscrit_dir in manuscrits:
        xml_files = [
            f for f in manuscrit_dir.rglob("*.xml") if ".chocomufin" not in f.name
        ]
        compteur_ms: Counter = Counter()
        nb_lignes = nb_avec_abrev = 0

        for xml_path in xml_files:
            for ligne in extraire_lignes_xml(xml_path):
                nb_lignes += 1
                chars_abrev = detecter_abreviations(ligne)
                if chars_abrev:
                    nb_avec_abrev += 1
                    compteur_ms.update(chars_abrev)
                    compteur_global.update(chars_abrev)

        total_lignes += nb_lignes
        total_avec_abrev += nb_avec_abrev
        taux = round(nb_avec_abrev / nb_lignes, 4) if nb_lignes else 0.0

        top = [
            {
                "char": c,
                "code": f"U+{ord(c):04X}",
                "nom": _nom_unicode(c),
                "count": n,
            }
            for c, n in compteur_ms.most_common(10)
        ]
        stats_par_manuscrit.append({
            "nom": manuscrit_dir.name,
            "nb_lignes": nb_lignes,
            "nb_lignes_abrev": nb_avec_abrev,
            "taux_abrev": taux,
            "top_chars": top,
        })

    taux_global = round(total_avec_abrev / total_lignes, 4) if total_lignes else 0.0
    top_global = [
        {
            "char": c,
            "code": f"U+{ord(c):04X}",
            "nom": _nom_unicode(c),
            "count": n,
        }
        for c, n in compteur_global.most_common(10)
    ]

    return {
        "nb_lignes_total": total_lignes,
        "nb_lignes_abrev": total_avec_abrev,
        "taux_abrev_global": taux_global,
        "top_chars_global": top_global,
        "par_manuscrit": stats_par_manuscrit,
    }


def _nom_unicode(c: str) -> str:
    """Retourne le nom Unicode d'un caractère, ou 'INCONNU' si absent."""
    try:
        return unicodedata.name(c)
    except ValueError:
        return "INCONNU"


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

    # 5. Statistiques d'abréviations
    print("\n=== Statistiques abréviations ===")
    stats_abrev = statistiques_abreviations()
    print(f"Lignes totales     : {stats_abrev['nb_lignes_total']}")
    print(f"Lignes avec abrev. : {stats_abrev['nb_lignes_abrev']}")
    print(f"Taux global        : {stats_abrev['taux_abrev_global']:.1%}")
    print("\nTop 10 caractères d'abréviation :")
    for entry in stats_abrev["top_chars_global"]:
        print(f"  {entry['code']}  {entry['nom']:<45}  x{entry['count']}")
    print("\nTaux par manuscrit :")
    print(f"{'Nom':<50} {'Taux':>6}")
    print("-" * 58)
    for m in stats_abrev["par_manuscrit"]:
        print(f"{m['nom']:<50} {m['taux_abrev']:>6.1%}")

    out_abrev = Path("data/abreviations_stats.json")
    out_abrev.write_text(json.dumps(stats_abrev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[info] Stats abreviations -> {out_abrev}")