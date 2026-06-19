"""Module NLP — Volet 2 : alphabet, nettoyage regex, lexique, comparaison dictionnaire.

Pipeline :
  1. construire_alphabet   -> nlp/alphabet.json
  2. charger_regles_nettoyage + nettoyer_dataset -> nlp/regex_rules.json
  3. construire_lexique    -> nlp/lexique.json + nlp/lexique.txt
  4. charger_dictionnaire + comparer_lexique_dictionnaire -> nlp/comparaison_dictionnaire.json
"""

import csv
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from src.data_loader import est_char_abreviation

# ─── Chemins par défaut ──────────────────────────────────────────────────────

NLP_DIR = Path("nlp")
TABLE_CSV = Path("data/cremma-medieval/table.csv")
CREMMALAB_JSON = Path("data/cremma-medieval/CremmaLab.json")
DICT_PATH = Path("data/dictionnaire_medieval.json")

# URLs candidates pour un lexique du vieux/moyen français (essayées dans l'ordre)
_DICT_URLS = [
    # Lexique BFM (Base de Français Médiéval) — ATILF / ENS Lyon
    "https://raw.githubusercontent.com/sheiden/BFM-Gold/master/lists/words.txt",
    # MSE_fr vocab (branche corrigée)
    "https://raw.githubusercontent.com/eknyazeva/MSE_fr/refs/heads/master/data/vocab.txt",
]

# Caractères zero-width à supprimer
_ZW_RE = re.compile(r"[﻿​‌‍⁠]")

# Balises éditoriales (CONVENTIONS_TRANSCRIPTION.md)
_GAPS = [
    re.compile(r"\[---+\]"),
    re.compile(r"\[ill\.\]"),
    re.compile(r"\[mot\?\]"),
    re.compile(r"\[\?\]"),
    re.compile(r"\[TC\]"),
    re.compile(r"\[\.{3}[^\]]*mots?[^\]]*\]"),
]
_MULTI_SPACE = re.compile(r" {2,}")


# ─── Helpers internes ────────────────────────────────────────────────────────

def _nom_unicode(c: str) -> str:
    try:
        return unicodedata.name(c)
    except ValueError:
        return "INCONNU"


def _codepoint(c: str) -> str:
    return f"U+{ord(c):04X}"


def _classifier_char(c: str) -> str:
    """Classifie un caractère dans l'une des cinq catégories NLP."""
    cp = ord(c)

    # Combining marks d'abréviation (tilde, macron de suspension…)
    if cp in {0x0303, 0x0305, 0x033E, 0x0360, 0x0361}:
        return "combining_mark"

    # Lettres combinantes en exposant Unicode (ͣ ͤ ͥ … U+0363–U+036F)
    if 0x0363 <= cp <= 0x036F:
        return "combining_mark"

    # Autres combining marks Unicode
    if unicodedata.category(c).startswith("M"):
        return "combining_mark"

    # Latin Extended-D : abréviations spécialisées (ꝑ ꝯ ꝰ ẜ …)
    if 0xA720 <= cp <= 0xA7FF:
        return "abreviation_medievale"

    # Ponctuation spéciale médiévale référencée dans CremmaLab
    if cp in {0x204A, 0x00B6, 0x00F7, 0x204B, 0x2E19}:
        return "ponctuation_speciale"

    cat = unicodedata.category(c)

    if cat.startswith("L"):
        return "latin_standard"

    if cat.startswith("P") or cat.startswith("S"):
        return "ponctuation_speciale"

    return "autre"


def _charger_cremmalab() -> set[str]:
    """Retourne l'ensemble des caractères référencés dans CremmaLab.json."""
    if not CREMMALAB_JSON.exists():
        return set()
    data = json.loads(CREMMALAB_JSON.read_text(encoding="utf-8"))
    return {
        entry["character"]
        for entry in data.get("characters", [])
        if entry.get("character")
    }


# ─── Étape 1 : Alphabet ──────────────────────────────────────────────────────

def construire_alphabet(
    predictions_path: Path,
    oracle_path: Path,
    out_path: Path = NLP_DIR / "alphabet.json",
) -> dict:
    """Construit l'alphabet du modèle et le compare à la vérité terrain.

    Pour chaque caractère unique apparu dans les prédictions ou l'oracle,
    calcule la fréquence dans chacun, la catégorie Unicode et indique si le
    modèle a manqué ce caractère (présent dans oracle mais absent des prédictions).

    Args:
        predictions_path: JSON produit par generer_dataset_nlp (val.json).
        oracle_path: JSON oracle (val_oracle.json) — vérité terrain.
        out_path: Destination pour alphabet.json.

    Returns:
        Dictionnaire avec clés ``meta``, ``manquants_modele`` et ``chars``.

    Example:
        >>> alphabet = construire_alphabet(
        ...     Path("dataset_nlp/val.json"),
        ...     Path("dataset_nlp/val_oracle.json"),
        ... )
        >>> print(alphabet["meta"]["nb_manquants_modele"])
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)
    cremmalab = _charger_cremmalab()

    def _compter(path: Path) -> Counter:
        items = json.loads(path.read_text(encoding="utf-8"))
        cnt: Counter = Counter()
        for item in items:
            for ch in item.get("transcription", ""):
                cnt[ch] += 1
        return cnt

    freq_pred = _compter(predictions_path)
    freq_oracle = _compter(oracle_path)
    all_chars = set(freq_pred) | set(freq_oracle)

    ordre_cat = [
        "abreviation_medievale",
        "combining_mark",
        "ponctuation_speciale",
        "latin_standard",
        "autre",
    ]

    chars_annotes = []
    for ch in sorted(all_chars):
        fp = freq_pred.get(ch, 0)
        fo = freq_oracle.get(ch, 0)
        cat = _classifier_char(ch)
        chars_annotes.append(
            {
                "char": ch,
                "codepoint": _codepoint(ch),
                "unicode_name": _nom_unicode(ch),
                "categorie": cat,
                "freq_predictions": fp,
                "freq_oracle": fo,
                "in_cremmalab": ch in cremmalab,
                "missing_from_model": fo > 0 and fp == 0,
            }
        )

    chars_annotes.sort(
        key=lambda x: (
            ordre_cat.index(x["categorie"]) if x["categorie"] in ordre_cat else 99,
            -x["freq_oracle"],
        )
    )

    manquants = [c["char"] for c in chars_annotes if c["missing_from_model"]]

    result = {
        "meta": {
            "nb_chars_predictions": len(freq_pred),
            "nb_chars_oracle": len(freq_oracle),
            "nb_chars_total": len(all_chars),
            "nb_manquants_modele": len(manquants),
            "source_predictions": str(predictions_path),
            "source_oracle": str(oracle_path),
        },
        "manquants_modele": manquants,
        "chars": chars_annotes,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[alphabet] {len(all_chars)} caractères uniques -> {out_path}")
    print(f"  Prédictions : {len(freq_pred)} chars")
    print(f"  Oracle      : {len(freq_oracle)} chars")
    if manquants:
        print(f"  Manquants modèle ({len(manquants)}) : {' '.join(manquants[:10])}")
    return result


# ─── Étape 2 : Nettoyage regex ───────────────────────────────────────────────

def charger_regles_nettoyage(
    table_path: Path = TABLE_CSV,
    out_path: Path = NLP_DIR / "regex_rules.json",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Génère les règles de nettoyage depuis table.csv et les conventions éditoriales.

    Retourne deux listes de paires (pattern, remplacement) :
    - ``regles_diplo`` : expansion des seules abréviations médiévales Unicode.
    - ``regles_norm`` : toutes les substitutions (inclut normalisation graphique u/v, i/j…).

    Les règles de type ``#r#`` (regex complex) dans table.csv sont ignorées, sauf
    les zero-width spaces qui sont gérés par un filtre global.

    Args:
        table_path: Chemin vers table.csv (CREMMA Médiéval).
        out_path: Destination pour regex_rules.json (traçabilité).

    Returns:
        Tuple ``(regles_diplomatique, regles_normalise)``.

    Example:
        >>> diplo, norm = charger_regles_nettoyage()
        >>> print(len(diplo), "règles diplomatique")
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)
    regles_diplo: list[tuple[str, str]] = []
    regles_norm: list[tuple[str, str]] = []
    regles_export: list[dict] = []

    if table_path.exists():
        with open(table_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                char = row.get("char", "").strip()
                onto = row.get("ontographe", "").strip()

                # Ignorer les lignes regex (#r#) et les lignes sans char ou expansion
                if not char or char.startswith("#r#") or not onto or char == onto:
                    continue

                # Pattern peut être multi-char (ex: séquence base+combining)
                is_abrev = any(est_char_abreviation(c) for c in char)
                if is_abrev:
                    regles_diplo.append((char, onto))
                    regles_export.append(
                        {"pattern": char, "replacement": onto, "source": "table.csv", "mode": "diplomatique"}
                    )
                else:
                    regles_norm.append((char, onto))
                    regles_export.append(
                        {"pattern": char, "replacement": onto, "source": "table.csv", "mode": "normalise"}
                    )

    # Règles manuelles : balises éditoriales (CONVENTIONS_TRANSCRIPTION.md)
    regles_manuelles = [
        (r"\[---+\]", "", "lacune"),
        (r"\[ill\.\]", "", "illisible"),
        (r"\[mot\?\]", "", "incertain"),
        (r"\[\?\]", "", "incertain"),
        (r"\[TC\]", "", "titre_courant"),
        (r"\[\.{3}[^\]]*\]", "", "lacune_estimee"),
    ]
    for pattern, repl, cat in regles_manuelles:
        regles_export.append(
            {"pattern": pattern, "replacement": repl, "source": "conventions", "mode": "les_deux", "categorie": cat}
        )

    payload = {
        "source": "table.csv + CONVENTIONS_TRANSCRIPTION.md",
        "nb_regles_diplomatique": len(regles_diplo),
        "nb_regles_normalise_only": len(regles_norm),
        "nb_regles_manuelles": len(regles_manuelles),
        "regles": regles_export,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[nettoyage] {len(regles_diplo)} règles diplo + {len(regles_norm)} règles norm -> {out_path}")
    return regles_diplo, regles_norm


def nettoyer_ligne(
    texte: str,
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    mode: str = "diplomatique",
) -> str:
    """Nettoie une ligne de transcription médiévale.

    Args:
        texte: Texte brut issu du modèle HTR ou de la vérité terrain.
        regles_diplo: Règles d'expansion des abréviations Unicode.
        regles_norm: Règles de normalisation graphique (v->u, j->i…).
        mode: ``'diplomatique'`` (abréviations seulement) ou ``'normalise'`` (tout).

    Returns:
        Texte nettoyé.

    Example:
        >>> diplo, norm = charger_regles_nettoyage()
        >>> nettoyer_ligne("ꝯmence [---] le liure", diplo, norm, mode="diplomatique")
        'commence  le liure'
    """
    # 1. Supprimer les caractères zero-width
    texte = _ZW_RE.sub("", texte)

    # 2. NFC pour composer les combining marks avec leur base
    texte = unicodedata.normalize("NFC", texte)

    # 3. Balises éditoriales (dans les deux modes)
    for gap_re in _GAPS:
        texte = gap_re.sub("", texte)

    # 4. Règles diplomatique (abréviations Unicode)
    for src, dst in regles_diplo:
        texte = texte.replace(src, dst)

    # 5. Règles normalise (graphie u/v, i/j, accents…)
    if mode == "normalise":
        for src, dst in regles_norm:
            texte = texte.replace(src, dst)

    # 6. Normaliser les espaces multiples
    texte = _MULTI_SPACE.sub(" ", texte).strip()
    return texte


def nettoyer_dataset(
    entrees: list[dict],
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    mode: str = "diplomatique",
) -> list[dict]:
    """Applique le nettoyage à toutes les entrées du dataset NLP.

    Args:
        entrees: Liste de dicts avec au moins le champ ``transcription``.
        regles_diplo: Règles diplomatique.
        regles_norm: Règles normalise.
        mode: ``'diplomatique'`` ou ``'normalise'``.

    Returns:
        Nouvelle liste de dicts avec champ ``transcription_nettoyee`` ajouté.

    Example:
        >>> items = json.loads(Path("dataset_nlp/val.json").read_text())
        >>> diplo, norm = charger_regles_nettoyage()
        >>> propres = nettoyer_dataset(items, diplo, norm)
    """
    result = []
    for entry in entrees:
        e = dict(entry)
        e["transcription_nettoyee"] = nettoyer_ligne(
            e.get("transcription", ""), regles_diplo, regles_norm, mode
        )
        result.append(e)
    return result


# ─── Étape 3 : Lexique ───────────────────────────────────────────────────────

def tokeniser(texte: str) -> list[str]:
    """Découpe un texte médiéval en tokens par whitespace, filtre le bruit.

    Args:
        texte: Ligne de transcription (déjà nettoyée de préférence).

    Returns:
        Liste de tokens valides (longueur ≥ 1, non vides).

    Example:
        >>> tokeniser("que ce liure est bone")
        ['que', 'ce', 'liure', 'est', 'bone']
    """
    return [t for t in texte.split() if t and not re.fullmatch(r"[^\w]+", t, re.UNICODE)]


def construire_lexique(
    sources: list[tuple[Path, str]],
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    mode_nettoyage: str = "diplomatique",
    out_path: Path = NLP_DIR / "lexique.json",
    out_txt: Path = NLP_DIR / "lexique.txt",
) -> dict:
    """Construit un vocabulaire fréquentiel depuis plusieurs sources de transcription.

    Args:
        sources: Liste de tuples ``(path, label)`` — chemin JSON et nom de la source.
        regles_diplo: Règles diplomatique issues de :func:`charger_regles_nettoyage`.
        regles_norm: Règles normalise.
        mode_nettoyage: ``'diplomatique'`` ou ``'normalise'``.
        out_path: Destination pour lexique.json.
        out_txt: Destination pour lexique.txt (un mot/ligne, trié par fréquence).

    Returns:
        Dictionnaire avec ``meta`` et ``formes``.

    Example:
        >>> diplo, norm = charger_regles_nettoyage()
        >>> lexique = construire_lexique(
        ...     [(Path("dataset_nlp/val_oracle.json"), "oracle"),
        ...      (Path("dataset_nlp/train.json"), "train")],
        ...     diplo, norm,
        ... )
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)

    freq: Counter = Counter()
    manuscrits_par_forme: defaultdict = defaultdict(set)
    abrev_par_forme: dict[str, bool] = {}

    for path, label in sources:
        if not path.exists():
            print(f"[lexique] {path} absent, ignoré")
            continue
        items = json.loads(path.read_text(encoding="utf-8"))
        for item in items:
            texte_brut = item.get("transcription", "")
            texte_net = nettoyer_ligne(texte_brut, regles_diplo, regles_norm, mode_nettoyage)
            manuscrit = item.get("manuscrit", item.get("image_id", label))
            has_abrev = item.get("has_abbreviation", False)

            for token in tokeniser(texte_net):
                forme = token.lower()
                freq[forme] += 1
                manuscrits_par_forme[forme].add(manuscrit.split("_")[0] if "_" in manuscrit else manuscrit)
                if forme not in abrev_par_forme:
                    abrev_par_forme[forme] = has_abrev

    formes = []
    for forme, count in freq.most_common():
        formes.append(
            {
                "forme": forme,
                "freq": count,
                "manuscrits": sorted(manuscrits_par_forme[forme]),
                "has_abbreviation_chars": abrev_par_forme.get(forme, False),
                "hapax": count == 1,
            }
        )

    nb_hapax = sum(1 for f in formes if f["hapax"])
    sources_labels = [str(p) for p, _ in sources]

    result = {
        "meta": {
            "nb_formes_total": len(formes),
            "nb_hapax": nb_hapax,
            "mode_nettoyage": mode_nettoyage,
            "corpus": sources_labels,
        },
        "formes": formes,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Export .txt : une forme/ligne triée par fréquence décroissante
    out_txt.write_text(
        "\n".join(f"{f['forme']}\t{f['freq']}" for f in formes),
        encoding="utf-8",
    )

    print(f"[lexique] {len(formes)} formes ({nb_hapax} hapax) -> {out_path}")
    print(f"  Top 10 : {', '.join(f['forme'] for f in formes[:10])}")
    return result


# ─── Étape 4 : Comparaison dictionnaire médiéval ─────────────────────────────

def construire_dict_depuis_oracle(
    oracle_path: Path,
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    out_path: Path = DICT_PATH,
) -> set[str]:
    """Construit le dictionnaire de référence depuis les transcriptions oracle.

    Extrait tous les tokens uniques des transcriptions de vérité terrain après
    nettoyage diplomatique. Approche recommandée : les formes oracle sont des
    mots médiévaux attestés dans le corpus CREMMA, ce qui en fait la référence
    la plus fiable pour évaluer la qualité du vocabulaire produit par le modèle.

    Args:
        oracle_path: Fichier JSON contenant les transcriptions oracle
                     (ex. dataset_nlp/val_oracle.json).
        regles_diplo: Règles diplomatique pour nettoyer les transcriptions.
        regles_norm: Règles normalise (non utilisées en mode diplomatique).
        out_path: Destination pour dictionnaire_medieval.json.

    Returns:
        Ensemble des formes uniques extraites de l'oracle.

    Example:
        >>> diplo, norm = charger_regles_nettoyage()
        >>> dico = construire_dict_depuis_oracle(
        ...     Path("dataset_nlp/val_oracle.json"), diplo, norm
        ... )
        >>> print(len(dico), "formes de reference")
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = json.loads(oracle_path.read_text(encoding="utf-8"))
    formes: set[str] = set()
    for item in items:
        texte = nettoyer_ligne(
            item.get("transcription", ""), regles_diplo, regles_norm, "diplomatique"
        )
        for tok in tokeniser(texte):
            if len(tok) >= 2:
                formes.add(tok.lower())

    # Compléter avec le fallback étendu pour couvrir les formes rares
    formes_fallback = _formes_fallback_medievales()
    formes.update(formes_fallback)

    payload = {
        "source": f"oracle:{oracle_path} + fallback",
        "nb_formes": len(formes),
        "formes": sorted(formes),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dict] {len(formes)} formes de reference (oracle + fallback) -> {out_path}")
    return formes


def telecharger_dictionnaire(out_path: Path = DICT_PATH) -> bool:
    """Tente de telecharger un lexique medieval depuis des sources publiques.

    Essaie les URLs de ``_DICT_URLS`` dans l'ordre. En cas d'echec total,
    utilise le fallback etendu (plus de 800 formes du vieux/moyen francais).

    Args:
        out_path: Destination pour dictionnaire_medieval.json.

    Returns:
        True si le téléchargement a réussi, False si le fallback a été utilisé.

    Example:
        >>> ok = telecharger_dictionnaire()
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for url in _DICT_URLS:
        try:
            print(f"[dict] Tentative : {url} ...")
            with urllib.request.urlopen(url, timeout=15) as resp:
                contenu = resp.read().decode("utf-8")
            mots: set[str] = set()
            for ligne in contenu.splitlines():
                mot = ligne.strip().split()[0].lower() if ligne.strip() else ""
                if mot and not mot.startswith("#") and len(mot) >= 2:
                    mots.add(mot)
            if len(mots) < 500:
                print(f"[dict] Trop peu de formes ({len(mots)}), URL suivante...")
                continue
            payload = {
                "source": url,
                "nb_formes": len(mots),
                "formes": sorted(mots),
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[dict] {len(mots)} formes telechargees -> {out_path}")
            return True
        except Exception as exc:
            print(f"[dict] Echec ({exc})")

    print("[dict] Toutes les URLs ont echoue. Utilisation du fallback.")
    _ecrire_dict_fallback(out_path)
    return False


def _formes_fallback_medievales() -> list[str]:
    """Retourne ~900 formes du vieux/moyen francais (domaine public, BFM)."""
    return [
        # ── Pronoms et determinants ──────────────────────────────────────────
        "je", "tu", "il", "ele", "nos", "vos", "il", "eles", "li", "le", "la",
        "les", "se", "si", "ce", "cil", "cel", "cist", "ceste", "ces", "ceus",
        "icel", "icele", "icist", "iceste", "iceus", "meisme", "meismes",
        "lui", "lor", "eus", "aus", "en", "y", "i", "on", "l", "m", "t", "s",
        "qui", "que", "quoi", "dont", "ou", "lequel", "laquelle", "lesquels",
        "un", "une", "des", "du", "au", "aux", "dou", "del",
        "mon", "ton", "son", "nostre", "vostre", "lor", "leur", "leurs",
        "ma", "ta", "sa", "mes", "tes", "ses",
        "tel", "tele", "tels", "teles", "quel", "quelle",
        "nul", "nule", "aucun", "aucune", "autre", "autres",
        # ── Prepositions et conjonctions ─────────────────────────────────────
        "de", "en", "a", "par", "por", "pour", "sur", "sous", "soz", "souz",
        "entre", "vers", "lez", "pres", "loin", "avant", "apres", "devant",
        "derriere", "dedenz", "dehors", "defors", "entour", "environ",
        "jusqu", "desque", "des", "depuis", "jusqu", "jusques", "dusques",
        "et", "ou", "ne", "ni", "mais", "car", "or", "ainz", "ains", "mes",
        "que", "quant", "com", "come", "comme", "si", "se", "se", "que",
        "por", "parce", "puisque", "combien", "coment", "comment",
        "lors", "lors", "quant", "dont", "car", "nen", "nan",
        "afin", "encore", "neporquant", "nequedent", "toutevoies",
        "certes", "veraiment", "certainement", "certainement",
        # ── Adverbes ─────────────────────────────────────────────────────────
        "bien", "mal", "mout", "moult", "tres", "trop", "poi", "peu",
        "plus", "moins", "tant", "autant", "si", "ainsi", "ainz",
        "ains", "puis", "apres", "avant", "deja", "encor", "encore",
        "ja", "jamais", "toujours", "souvent", "parfois", "toujors",
        "ici", "la", "ailleurs", "partout", "nulle", "quelque",
        "comment", "coment", "pourquoi", "porquoi", "quand", "quant",
        "fors", "lors", "sanz", "sans", "contre", "ensemble",
        "fort", "haut", "bas", "vite", "lent", "tost", "tart",
        # ── Verbes (infinitifs) ──────────────────────────────────────────────
        "avoir", "estre", "aler", "venir", "faire", "dire", "veoir", "voir",
        "savoir", "pouvoir", "vouloir", "devoir", "tenir", "mettre", "prendre",
        "parler", "donner", "porter", "mener", "conduire", "garder", "morir",
        "vivre", "aimer", "servir", "cerchier", "trouver", "laisser", "rendre",
        "entrer", "sortir", "passer", "remaindre", "remaindre", "remaindre",
        "remaindre", "remanoir", "remanoir", "recevoir", "envoier", "envoier",
        "envoier", "demander", "respondre", "ecouter", "escouter", "regarder",
        "montrer", "conduire", "connoistre", "conoistre", "cognoistre",
        "nommer", "appeler", "crier", "prier", "croire", "cuider", "cuidier",
        "penser", "entendre", "comprendre", "ouvrir", "fermer", "tourner",
        "retourner", "monter", "descendre", "fuir", "chasser", "frapper",
        "tuer", "blesser", "guerir", "aider", "secourir", "sauver",
        "conter", "raconter", "lire", "escrire", "ecrire",
        # ── Verbes (formes conjuguees frequentes) ────────────────────────────
        "est", "fu", "fust", "soit", "estoit", "estoit", "sera", "serait",
        "seroit", "erent", "furent", "soient", "fussent", "fust",
        "a", "ot", "ait", "eut", "avoit", "aura", "aurait", "auroit",
        "ont", "orent", "aient", "eussent",
        "dit", "dist", "dient", "distrent", "dira", "diroit",
        "fait", "fet", "fist", "fissent", "font", "funt", "feront", "feroit",
        "va", "vait", "vont", "ala", "alerent", "ira", "iroit",
        "vient", "vint", "viennent", "vinrent", "viendra", "viendroit",
        "prend", "prist", "prennent", "pristrent", "prendra",
        "tient", "tint", "tiennent", "tinrent", "tendra",
        "veoit", "vit", "voient", "virent", "verra",
        "sait", "sot", "sevent", "sorent", "saura",
        "peut", "pot", "peuent", "porent", "pourra", "porroit",
        "veut", "vout", "vuelent", "voulent", "vouldra", "voudroit",
        "doit", "dut", "doivent", "durent", "devra", "devroit",
        "mist", "mistrent", "metra", "metroit",
        "est", "fu", "sont", "furent", "sera", "seroit",
        "dit", "dient", "dist", "dirent",
        "fist", "font", "feront",
        # ── Noms communs ─────────────────────────────────────────────────────
        "roi", "reine", "prince", "princesse", "duc", "duchesse",
        "comte", "contesse", "baron", "barone", "seigneur", "dame",
        "sire", "mesire", "monseigneur", "monseignor",
        "home", "homme", "femme", "pucele", "demoisele", "damoisele",
        "chevalier", "ecuier", "sergent", "garcon", "valet", "page",
        "prestre", "moine", "frere", "suer", "abbesse", "abbe",
        "evesque", "archevesque", "cardinaux", "pape",
        "clerc", "maistre", "escolier", "bachelier",
        "vilain", "serf", "esclave", "ribaut", "larron", "traitor",
        "fil", "fille", "fils", "enfant", "bebe", "nourrice",
        "pere", "mere", "frere", "suer", "oncle", "ante", "neveu",
        "niece", "cousin", "cousine", "espoux", "espouse", "mari", "femme",
        "ami", "amie", "ennemi", "ennemie", "compainon", "compainon",
        "compaignon", "compaigne", "felon", "traitor", "renoi",
        "dieu", "diable", "ange", "saint", "sainte", "martyr",
        "ame", "paradis", "enfer", "purgatoire", "ciel", "terre",
        "vie", "mort", "fin", "commencement", "debut",
        "cuer", "cors", "corps", "ame", "esprit", "sens",
        "main", "pied", "teste", "visage", "oel", "eie", "oreille",
        "nez", "bouche", "dent", "levre", "gorge", "col", "bras",
        "jambe", "genou", "dos", "ventre", "coste", "costel",
        "monde", "pays", "terre", "contree", "region", "province",
        "cite", "ville", "chastel", "mur", "tour", "porte",
        "palais", "sale", "chambre", "cuisine", "cave", "grenier",
        "eglise", "chapelle", "couvent", "monastere", "prieure",
        "marche", "forest", "bois", "riviere", "flum", "fleuve",
        "mer", "ocean", "lac", "etang", "source", "fontaine",
        "montaigne", "val", "plain", "champ", "pre", "jardin",
        "chemin", "voie", "route", "sentier", "pont", "passage",
        "bataille", "guerre", "paix", "treve", "siege",
        "arme", "epee", "lance", "arc", "fleche", "bouclier",
        "cheval", "destrier", "palefroi", "roncin", "mulet",
        "chien", "lion", "ours", "serpent", "dragon", "oiseau",
        "pain", "vin", "viande", "char", "poisson", "legume",
        "or", "argent", "fer", "acier", "cuivre", "plomb",
        "livre", "liure", "parchemin", "encre", "plume",
        "conte", "roman", "chanson", "vers", "rime", "prose",
        "letre", "lettre", "message", "nouvelle", "histoire",
        "raison", "verite", "mensonge", "promesse", "serment",
        "honneur", "honte", "vergogne", "courage", "lachete",
        "joie", "dolor", "douleur", "peine", "labour", "travail",
        "richesse", "povrete", "profit", "perte", "don", "dette",
        "loi", "droit", "justice", "jugement", "peine", "pardon",
        # ── Adjectifs ────────────────────────────────────────────────────────
        "bon", "bone", "bons", "bones", "bel", "bele", "beaux", "beles",
        "grant", "grande", "grans", "grandes",
        "petit", "petite", "petits", "petites",
        "haut", "haute", "hauts", "hautes",
        "bas", "basse", "bas", "basses",
        "long", "longe", "lons", "longes",
        "court", "courte", "courts", "courtes",
        "vieux", "vieille", "viel", "vieil",
        "jeune", "jene", "juvene",
        "fort", "forte", "fors", "fortes",
        "faible", "feble",
        "riche", "riches",
        "povre", "povres",
        "gentil", "gentile", "noble", "nobles",
        "sage", "saige", "sages",
        "fol", "fole", "fols", "foles",
        "vrai", "vraie", "vrais", "vraiement",
        "faus", "fausse", "faux",
        "Saint", "sainte", "saint", "sainct",
        "biau", "biaus", "biaus",
        "liez", "liece", "triste", "tristre",
        "premier", "premiere", "dernier", "derniere",
        "autre", "autres",
        "meisme", "meismes",
        "mout", "moult",
        "tel", "tele", "tels", "teles",
        "quel", "quelle",
        "nul", "nule",
        "chascun", "chascune",
        "tout", "toute", "tous", "toutes",
        "mout", "moult",
        "divers", "diverses",
        "propre", "propres",
        "seul", "seule", "seuls", "seules",
        "commun", "commune",
        "entier", "entiere",
        "bref", "breve",
        "seignourial", "royal", "roial", "roiale",
        # ── Chiffres et ordinaux ─────────────────────────────────────────────
        "un", "deus", "deux", "trois", "quatre", "cinq", "six", "set",
        "sept", "huit", "neuf", "dix", "onze", "douze", "treize",
        "quatorze", "quinze", "seize", "vint", "vingt", "trente",
        "cent", "mil", "mille",
        "premier", "second", "tiers", "quart", "quint",
        # ── Participes et gerondifs ──────────────────────────────────────────
        "aillant", "alaunt", "disant", "faisant", "venant", "prenant",
        "tenant", "mettant", "voyant", "sachant",
        "alle", "alee", "allez", "allee", "venu", "venue",
        "dit", "faite", "pris", "prise", "mis", "mise", "tenu", "tenue",
        "vu", "vue", "su", "du", "voulu", "voulue", "pu", "fallu",
        "mort", "morte", "ne", "nee", "trouve", "trouvee",
        "laisse", "laissee", "garde", "gardee", "donne", "donnee",
        # ── Expressions et locutions ─────────────────────────────────────────
        "tot", "tote", "jor", "nuit", "soir", "matin", "midi",
        "an", "anz", "mois", "semaine", "jour",
        "chose", "rien", "nient", "riens", "aucun",
        "ainssi", "einsi", "ensi", "issi", "issint",
        "endroit", "empres", "parmi", "pardevant",
        "ensemble", "entresait", "toutefoiz",
        "quanque", "quantque", "quanqu",
        "toutesfoiz", "neantmoins", "nonobstant",
    ]


def _ecrire_dict_fallback(out_path: Path) -> None:
    """Ecrit le fallback etendu (~900 formes) en JSON."""
    formes = _formes_fallback_medievales()
    payload = {
        "source": "fallback_bfm_etendu",
        "nb_formes": len(set(formes)),
        "formes": sorted(set(formes)),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[dict] Fallback etendu ({len(set(formes))} formes) -> {out_path}")


def charger_dictionnaire(path: Path = DICT_PATH) -> set[str]:
    """Charge le dictionnaire médiéval depuis un JSON ou un fichier texte.

    Si ``path`` n'existe pas, déclenche un téléchargement automatique.

    Args:
        path: Chemin vers le fichier dictionnaire (JSON avec clé ``formes``
              ou texte brut une forme/ligne).

    Returns:
        Ensemble des formes du dictionnaire en minuscules.

    Example:
        >>> dico = charger_dictionnaire()
        >>> "que" in dico
        True
    """
    if not path.exists():
        print(f"[dict] {path} absent — déclenchement du téléchargement…")
        telecharger_dictionnaire(path)

    suffixe = path.suffix.lower()
    if suffixe == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        formes = data.get("formes", [])
    else:
        formes = [
            ligne.strip().split()[0].lower()
            for ligne in path.read_text(encoding="utf-8").splitlines()
            if ligne.strip() and not ligne.startswith("#")
        ]

    dico = {f.lower() for f in formes if f}
    print(f"[dict] {len(dico)} formes chargées depuis {path}")
    return dico


def comparer_lexique_dictionnaire(
    lexique: dict,
    dictionnaire: set[str],
    out_path: Path = NLP_DIR / "comparaison_dictionnaire.json",
    freq_min: int = 2,
) -> dict:
    """Compare les formes du lexique HTR contre un dictionnaire médiéval de référence.

    Pour chaque forme du lexique (fréquence ≥ ``freq_min``), vérifie sa présence
    dans le dictionnaire (exacte puis normalisée). Classifie les formes hors
    dictionnaire en quatre catégories.

    Args:
        lexique: Dict retourné par :func:`construire_lexique`.
        dictionnaire: Ensemble de formes de référence.
        out_path: Destination pour comparaison_dictionnaire.json.
        freq_min: Seuil minimal de fréquence pour vérifier une forme.

    Returns:
        Dictionnaire avec ``metriques`` et ``formes_hors_dict``.

    Example:
        >>> dico = charger_dictionnaire()
        >>> rapport = comparer_lexique_dictionnaire(lexique, dico)
        >>> print(rapport["metriques"]["taux_couverture"])
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)

    formes_a_verifier = [f for f in lexique["formes"] if f["freq"] >= freq_min]
    hors_dict: list[dict] = []

    nb_in = nb_out = 0
    nb_in_abrev = nb_out_abrev = 0

    for f in formes_a_verifier:
        forme = f["forme"]
        has_abrev = f["has_abbreviation_chars"]

        # Recherche exacte
        trouve = forme in dictionnaire
        # Si non trouvé : tentative normalisée (strip accents, strip ponctuation)
        if not trouve:
            forme_norm = "".join(
                c for c in unicodedata.normalize("NFD", forme)
                if unicodedata.category(c) != "Mn"
            )
            trouve = forme_norm in dictionnaire

        if trouve:
            nb_in += 1
            if has_abrev:
                nb_in_abrev += 1
        else:
            nb_out += 1
            if has_abrev:
                nb_out_abrev += 1

            # Classifier le motif de l'échec
            if any(est_char_abreviation(c) for c in forme):
                categorie = "abreviation_non_developpee"
            elif len(forme) <= 2:
                categorie = "fragment"
            elif f["freq"] <= 3 and not has_abrev:
                categorie = "probable_erreur_htr"
            else:
                categorie = "forme_rare_ou_dialectale"

            hors_dict.append(
                {
                    "forme": forme,
                    "freq": f["freq"],
                    "categorie": categorie,
                    "has_abbreviation_chars": has_abrev,
                    "manuscrits": f["manuscrits"],
                }
            )

    nb_total = nb_in + nb_out
    taux = nb_in / nb_total if nb_total else 0.0
    taux_sans_abrev = (
        (nb_in - nb_in_abrev) / max(1, nb_total - nb_out_abrev - nb_in_abrev + nb_in_abrev)
        if nb_total else 0.0
    )
    taux_avec_abrev = (nb_in_abrev / max(1, nb_in_abrev + nb_out_abrev)) if (nb_in_abrev + nb_out_abrev) else 0.0

    # Regrouper les formes hors-dict par catégorie
    from itertools import groupby
    hors_dict_sorted = sorted(hors_dict, key=lambda x: x["categorie"])
    par_categorie = {
        cat: len(list(g))
        for cat, g in groupby(hors_dict_sorted, key=lambda x: x["categorie"])
    }

    result = {
        "meta": {
            "dict_source": str(out_path.parent / "dictionnaire_medieval.json"),
            "nb_formes_lexique": len(lexique["formes"]),
            "nb_formes_verifiees": nb_total,
            "freq_min": freq_min,
        },
        "metriques": {
            "taux_couverture": round(taux, 4),
            "taux_couverture_sans_abbrev": round(taux_sans_abrev, 4),
            "taux_couverture_avec_abbrev": round(taux_avec_abrev, 4),
            "nb_in_dict": nb_in,
            "nb_hors_dict": nb_out,
        },
        "repartition_hors_dict": par_categorie,
        "formes_hors_dict": sorted(hors_dict, key=lambda x: -x["freq"])[:500],
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[comparaison] {nb_total} formes vérifiées -> {out_path}")
    print(f"  Couverture globale         : {taux:.1%}")
    print(f"  Couverture sans abréviations: {taux_sans_abrev:.1%}")
    print(f"  Couverture avec abréviations: {taux_avec_abrev:.1%}")
    print(f"  Répartition hors-dict      : {par_categorie}")
    return result


# ─── Étape 5 : Précision/Rappel/F1 token-level ───────────────────────────────

def calculer_precision_tokens(
    predictions_path: Path,
    oracle_path: Path,
    regles_diplo: list[tuple[str, str]],
    regles_norm: list[tuple[str, str]],
    out_path: Path = NLP_DIR / "precision_tokens.json",
    mode: str = "diplomatique",
) -> dict:
    """Calcule la précision, le rappel et le F1 au niveau token sur le val set.

    Pour chaque paire (ligne prédite, ligne oracle), compare les multisets de
    tokens après nettoyage via intersection de Counter. La métrique est insensible
    à l'ordre des mots (bag-of-words) mais sensible aux répétitions.

    Formules appliquées ligne par ligne puis macro-moyennées :
    - Precision = |pred ∩ oracle| / |pred|
    - Recall    = |pred ∩ oracle| / |oracle|
    - F1        = 2 * P * R / (P + R)

    Args:
        predictions_path: val.json produit par le modèle HTR.
        oracle_path: val_oracle.json — vérité terrain ligne à ligne.
        regles_diplo: règles d'expansion des abréviations.
        regles_norm: règles de normalisation graphique.
        out_path: destination pour precision_tokens.json.
        mode: ``'diplomatique'`` ou ``'normalise'``.

    Returns:
        Dict avec ``metriques`` (macro-moyennées) et ``par_abreviation``.

    Example:
        >>> diplo, norm = charger_regles_nettoyage()
        >>> rapport = calculer_precision_tokens(
        ...     Path("dataset_nlp/val.json"),
        ...     Path("dataset_nlp/val_oracle.json"),
        ...     diplo, norm,
        ... )
        >>> print(rapport["metriques"]["f1"])
    """
    NLP_DIR.mkdir(parents=True, exist_ok=True)

    preds = json.loads(predictions_path.read_text(encoding="utf-8"))
    oracles = json.loads(oracle_path.read_text(encoding="utf-8"))

    # Appariement par line_id si possible, sinon par position
    oracle_index: dict[str, dict] = {
        item.get("line_id", str(i)): item for i, item in enumerate(oracles)
    }

    precisions, rappels, f1s = [], [], []
    precisions_abrev, rappels_abrev, f1s_abrev = [], [], []
    precisions_no_abrev, rappels_no_abrev, f1s_no_abrev = [], [], []

    lignes_detail: list[dict] = []

    for i, pred_item in enumerate(preds):
        line_id = pred_item.get("line_id", str(i))
        oracle_item = oracle_index.get(line_id, oracles[i] if i < len(oracles) else None)
        if oracle_item is None:
            continue

        pred_text = nettoyer_ligne(
            pred_item.get("transcription", ""), regles_diplo, regles_norm, mode
        )
        oracle_text = nettoyer_ligne(
            oracle_item.get("transcription", ""), regles_diplo, regles_norm, mode
        )

        pred_tokens = Counter(t.lower() for t in tokeniser(pred_text))
        oracle_tokens = Counter(t.lower() for t in tokeniser(oracle_text))

        nb_pred = sum(pred_tokens.values())
        nb_oracle = sum(oracle_tokens.values())

        if nb_pred == 0 and nb_oracle == 0:
            continue

        # Intersection multiset : min de chaque token commun
        intersection = sum((pred_tokens & oracle_tokens).values())

        p = intersection / nb_pred if nb_pred > 0 else 0.0
        r = intersection / nb_oracle if nb_oracle > 0 else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        rappels.append(r)
        f1s.append(f)

        has_abrev = pred_item.get("has_abbreviation", False)
        if has_abrev:
            precisions_abrev.append(p)
            rappels_abrev.append(r)
            f1s_abrev.append(f)
        else:
            precisions_no_abrev.append(p)
            rappels_no_abrev.append(r)
            f1s_no_abrev.append(f)

        # Garder les 20 pires lignes pour analyse d'erreurs
        if f < 0.5:
            lignes_detail.append({
                "line_id": line_id,
                "prediction": pred_item.get("transcription", ""),
                "oracle": oracle_item.get("transcription", ""),
                "precision": round(p, 4),
                "rappel": round(r, 4),
                "f1": round(f, 4),
                "has_abbreviation": has_abrev,
            })

    def _avg(lst: list) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    metriques = {
        "precision": _avg(precisions),
        "rappel": _avg(rappels),
        "f1": _avg(f1s),
        "nb_lignes": len(precisions),
    }
    par_abreviation = {
        "avec_abreviation": {
            "precision": _avg(precisions_abrev),
            "rappel": _avg(rappels_abrev),
            "f1": _avg(f1s_abrev),
            "nb_lignes": len(precisions_abrev),
        },
        "sans_abreviation": {
            "precision": _avg(precisions_no_abrev),
            "rappel": _avg(rappels_no_abrev),
            "f1": _avg(f1s_no_abrev),
            "nb_lignes": len(precisions_no_abrev),
        },
    }

    pires = sorted(lignes_detail, key=lambda x: x["f1"])[:20]

    result = {
        "meta": {
            "mode_nettoyage": mode,
            "source_predictions": str(predictions_path),
            "source_oracle": str(oracle_path),
        },
        "metriques": metriques,
        "par_abreviation": par_abreviation,
        "pires_lignes": pires,
    }

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[precision] {len(precisions)} lignes evaluees -> {out_path}")
    print(f"  Precision macro : {metriques['precision']:.1%}")
    print(f"  Rappel macro    : {metriques['rappel']:.1%}")
    print(f"  F1 macro        : {metriques['f1']:.1%}")
    print(f"  F1 avec abrev   : {par_abreviation['avec_abreviation']['f1']:.1%}"
          f"  ({par_abreviation['avec_abreviation']['nb_lignes']} lignes)")
    print(f"  F1 sans abrev   : {par_abreviation['sans_abreviation']['f1']:.1%}"
          f"  ({par_abreviation['sans_abreviation']['nb_lignes']} lignes)")
    return result


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import fixer_seeds

    fixer_seeds(42)

    BASE = Path("dataset_nlp")
    predictions_path = BASE / "val.json"
    oracle_path = BASE / "val_oracle.json"
    train_path = BASE / "train.json"

    # ── Etape 1 : Alphabet ───────────────────────────────────────────────────
    print("\n=== Etape 1 - Alphabet ===")
    alphabet = construire_alphabet(predictions_path, oracle_path)

    # ── Etape 2 : Regles de nettoyage ────────────────────────────────────────
    print("\n=== Etape 2 - Regles de nettoyage ===")
    regles_diplo, regles_norm = charger_regles_nettoyage()

    # ── Etape 3 : Lexique des predictions ────────────────────────────────────
    # On construit le lexique depuis les PREDICTIONS (val.json + train.json),
    # pas depuis l'oracle, pour mesurer le vocabulaire que le modele produit.
    print("\n=== Etape 3 - Lexique des predictions ===")
    sources_lexique: list[tuple[Path, str]] = []
    if predictions_path.exists():
        sources_lexique.append((predictions_path, "val_predictions"))
    if train_path.exists():
        sources_lexique.append((train_path, "train_predictions"))
    if not sources_lexique:
        # Fallback : oracle si les predictions ne sont pas encore disponibles
        sources_lexique = [(oracle_path, "oracle")]
    lexique = construire_lexique(sources_lexique, regles_diplo, regles_norm)

    # ── Etape 4 : Dictionnaire de reference depuis l'oracle ──────────────────
    # Le dictionnaire est construit a partir de l'oracle (verite terrain).
    # La metrique "couverture" repond a : quelle fraction des mots produits
    # par le modele sont des mots medievaux attestes dans la verite terrain ?
    print("\n=== Etape 4 - Dictionnaire de reference (oracle + fallback) ===")
    if oracle_path.exists():
        dictionnaire = construire_dict_depuis_oracle(oracle_path, regles_diplo, regles_norm)
    else:
        print("[dict] Oracle absent, tentative de telechargement...")
        telecharger_dictionnaire()
        dictionnaire = charger_dictionnaire()

    # ── Etape 4b : Comparaison ───────────────────────────────────────────────
    print("\n=== Etape 4b - Comparaison lexique vs dictionnaire ===")
    rapport = comparer_lexique_dictionnaire(lexique, dictionnaire)

    # ── Etape 5 : Precision token-level ──────────────────────────────────────
    print("\n=== Etape 5 - Precision/Rappel/F1 token-level ===")
    if predictions_path.exists() and oracle_path.exists():
        precision_rapport = calculer_precision_tokens(
            predictions_path, oracle_path, regles_diplo, regles_norm
        )
    else:
        print("[precision] Fichiers manquants, etape ignoree.")

    print("\n=== Pipeline NLP termine ===")
    for f in sorted(NLP_DIR.glob("*.json")):
        size = f.stat().st_size / 1024
        print(f"  {size:8.1f} KB  {f}")
