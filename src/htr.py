"""Fine-tuning et inférence HTR pour manuscrits médiévaux.

Axe de recherche : impact des abréviations médiévales sur les performances HTR.
Pipeline : extraction des lignes → baseline TrOCR → fine-tuning LoRA →
évaluation stratifiée CER (global / abrévié / non-abrévié).
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import editdistance
import numpy as np
from PIL import Image

from src.data_loader import contient_abreviation, detecter_abreviations
from src.utils import fixer_seeds, log_experience

# Namespace ALTO XML
_ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v4#"}

MODELE_TROCR = "microsoft/trocr-base-handwritten"
CHECKPOINTS_DIR = Path("checkpoints")
LIGNES_DIR = Path("data/lignes")


# ---------------------------------------------------------------------------
# Extraction des lignes depuis les fichiers ALTO XML
# ---------------------------------------------------------------------------

def extraire_lignes_avec_images(
    xml_path: Path,
    image_path: Path,
    split: str,
    manuscrit: str,
) -> list[dict]:
    """Extrait les crops de lignes + transcriptions depuis un fichier ALTO XML.

    Pour chaque TextLine du fichier, découpe l'image selon la bounding box
    (HPOS, VPOS, WIDTH, HEIGHT) et retourne les métadonnées associées.

    Args:
        xml_path: Chemin vers le fichier ALTO XML.
        image_path: Chemin vers l'image correspondante (.jpg).
        split: Nom de l'ensemble (``train``, ``val``, ``test``).
        manuscrit: Nom du dossier manuscrit.

    Returns:
        Liste de dicts avec clés : ``image`` (PIL.Image), ``text``,
        ``has_abbreviation``, ``abreviations``, ``manuscrit``, ``split``,
        ``line_id``, ``source_xml``.

    Example:
        >>> lignes = extraire_lignes_avec_images(xml, img, "train", "bnf_fr_1728")
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    try:
        img = Image.open(str(image_path)).convert("RGB")
    except (OSError, FileNotFoundError):
        return []

    img_w, img_h = img.size
    lignes = []

    for text_line in root.findall(".//alto:TextLine", _ALTO_NS):
        try:
            hpos = int(text_line.get("HPOS", 0))
            vpos = int(text_line.get("VPOS", 0))
            width = int(text_line.get("WIDTH", 0))
            height = int(text_line.get("HEIGHT", 0))
        except (ValueError, TypeError):
            continue

        if width <= 0 or height <= 0:
            continue

        string_el = text_line.find("alto:String", _ALTO_NS)
        if string_el is None:
            continue
        text = string_el.get("CONTENT", "").strip()
        if not text:
            continue

        x1 = max(0, hpos)
        y1 = max(0, vpos)
        x2 = min(img_w, hpos + width)
        y2 = min(img_h, vpos + height)
        if x2 <= x1 or y2 <= y1:
            continue

        lignes.append({
            "image": img.crop((x1, y1, x2, y2)),
            "text": text,
            "has_abbreviation": contient_abreviation(text),
            "abreviations": detecter_abreviations(text),
            "manuscrit": manuscrit,
            "split": split,
            "line_id": text_line.get("ID", ""),
            "source_xml": xml_path.name,
        })

    return lignes


def construire_dataset(
    split_path: Path = Path("data/split.json"),
    out_dir: Path = LIGNES_DIR,
    max_lignes_par_split: Optional[int] = None,
    forcer: bool = False,
) -> dict[str, list[dict]]:
    """Construit le dataset HTR ligne par ligne à partir du split.

    Découpe les images en crops de lignes, sauvegarde sur disque et crée un
    manifest JSON par ensemble. Si les manifests existent déjà et que
    ``forcer=False``, les recharge sans reconstruire.

    Args:
        split_path: Chemin vers ``data/split.json``.
        out_dir: Dossier de sortie des crops et manifests.
        max_lignes_par_split: Limite optionnelle (utile pour les tests).
        forcer: Si True, reconstruit même si les manifests existent.

    Returns:
        Dict ``{split: [metadata_dicts]}`` — métadonnées de chaque ligne.

    Example:
        >>> dataset = construire_dataset()
        >>> print(len(dataset["train"]), "lignes en train")
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vérifier si les manifests existent déjà
    if not forcer:
        manifests = {
            s: out_dir / f"manifest_{s}.json"
            for s in ("train", "val", "test")
        }
        if all(p.exists() for p in manifests.values()):
            print("[info] Manifests déjà présents, rechargement ...")
            return {
                s: json.loads(p.read_text(encoding="utf-8"))
                for s, p in manifests.items()
            }

    with open(split_path, "r", encoding="utf-8") as f:
        split_json = json.load(f)

    dataset: dict[str, list[dict]] = {}

    for split_name, chemins in split_json.items():
        out_split = out_dir / split_name
        out_split.mkdir(parents=True, exist_ok=True)
        lignes_split: list[dict] = []

        for chemin_manuscrit in chemins:
            manuscrit_dir = Path(chemin_manuscrit)
            manuscrit_nom = manuscrit_dir.name
            xml_files = [
                f for f in manuscrit_dir.rglob("*.xml")
                if ".chocomufin" not in f.name
            ]

            for xml_path in sorted(xml_files):
                image_path = xml_path.with_suffix(".jpg")
                if not image_path.exists():
                    continue

                lignes = extraire_lignes_avec_images(
                    xml_path, image_path, split_name, manuscrit_nom
                )
                for i, ligne in enumerate(lignes):
                    img_name = f"{manuscrit_nom}_{xml_path.stem}_l{i:04d}.png"
                    img_out = out_split / img_name
                    ligne["image"].save(str(img_out))

                    meta = {k: v for k, v in ligne.items() if k != "image"}
                    meta["image_path"] = str(img_out)
                    lignes_split.append(meta)

        if max_lignes_par_split:
            lignes_split = lignes_split[:max_lignes_par_split]

        manifest_path = out_dir / f"manifest_{split_name}.json"
        manifest_path.write_text(
            json.dumps(lignes_split, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        dataset[split_name] = lignes_split
        nb_abrev = sum(1 for l in lignes_split if l["has_abbreviation"])
        print(f"[info] {split_name:5s}: {len(lignes_split):6d} lignes  "
              f"({nb_abrev} avec abréviations, {nb_abrev/max(1, len(lignes_split)):.1%})")

    return dataset


# ---------------------------------------------------------------------------
# Métriques CER et évaluation stratifiée
# ---------------------------------------------------------------------------

def calculer_cer(prediction: str, reference: str) -> float:
    """Calcule le Character Error Rate (CER) entre une prédiction et une référence.

    CER = distance_edition(pred, ref) / len(ref).
    Cas limites : référence vide → 0.0 si pred vide, 1.0 sinon.

    Args:
        prediction: Texte prédit par le modèle HTR.
        reference: Transcription de référence.

    Returns:
        CER ∈ [0, +∞[. Les valeurs > 1.0 sont possibles si la prédiction
        est beaucoup plus longue que la référence.

    Example:
        >>> calculer_cer("bnjour", "bonjour")
        0.142857...
    """
    if not reference:
        return 0.0 if not prediction else 1.0
    return editdistance.eval(prediction, reference) / len(reference)


def calculer_cer_moyen(predictions: list[str], references: list[str]) -> float:
    """Calcule le CER moyen sur une liste de paires (prédiction, référence).

    Args:
        predictions: Liste de textes prédits.
        references: Liste de transcriptions de référence (même longueur).

    Returns:
        CER moyen. Retourne 0.0 si les listes sont vides.

    Example:
        >>> calculer_cer_moyen(["bonjour"], ["bonjour"])
        0.0
    """
    if not predictions or not references:
        return 0.0
    scores = [calculer_cer(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(scores))


def evaluer_stratifie(
    predictions: list[str],
    references: list[str],
    has_abbreviations: list[bool],
) -> dict:
    """Évalue les performances HTR en distinguant lignes abrégées et non-abrégées.

    C'est la métrique centrale pour répondre à la problématique du projet :
    « dans quelle mesure les abréviations influencent-elles le CER ? »

    Args:
        predictions: Transcriptions prédites par le modèle.
        references: Transcriptions de référence (vérité terrain).
        has_abbreviations: Booléen par ligne — True si la référence contient
            au moins un caractère d'abréviation médiévale.

    Returns:
        Dict avec ``CER_global``, ``CER_abbrev``, ``CER_no_abbrev``,
        ``delta_CER`` (CER_abbrev − CER_no_abbrev), ``nb_lignes_abrev``,
        ``nb_lignes_no_abbrev``.

    Example:
        >>> stats = evaluer_stratifie(preds, refs, flags)
        >>> print(f"Impact abréviations : +{stats['delta_CER']:.1%}")
    """
    idx_abrev = [i for i, h in enumerate(has_abbreviations) if h]
    idx_no_abrev = [i for i, h in enumerate(has_abbreviations) if not h]

    cer_global = calculer_cer_moyen(predictions, references)

    cer_abrev = calculer_cer_moyen(
        [predictions[i] for i in idx_abrev],
        [references[i] for i in idx_abrev],
    ) if idx_abrev else None

    cer_no_abrev = calculer_cer_moyen(
        [predictions[i] for i in idx_no_abrev],
        [references[i] for i in idx_no_abrev],
    ) if idx_no_abrev else None

    delta = (cer_abrev - cer_no_abrev) if (cer_abrev is not None and cer_no_abrev is not None) else None

    return {
        "CER_global": round(cer_global, 4),
        "CER_abbrev": round(cer_abrev, 4) if cer_abrev is not None else None,
        "CER_no_abbrev": round(cer_no_abrev, 4) if cer_no_abrev is not None else None,
        "delta_CER": round(delta, 4) if delta is not None else None,
        "nb_lignes_abrev": len(idx_abrev),
        "nb_lignes_no_abbrev": len(idx_no_abrev),
    }


# ---------------------------------------------------------------------------
# Inférence TrOCR (baseline et modèle fine-tuné)
# ---------------------------------------------------------------------------

def inferer_trocr(
    images: list[Image.Image],
    model_path: str = MODELE_TROCR,
    batch_size: int = 4,
) -> list[str]:
    """Transcrit une liste d'images de lignes avec TrOCR.

    Args:
        images: Images PIL (lignes de texte) à transcrire.
        model_path: Identifiant HuggingFace ou chemin local vers le modèle.
        batch_size: Nombre d'images traitées en parallèle.

    Returns:
        Liste de transcriptions dans le même ordre que les images.

    Raises:
        ImportError: Si ``transformers`` n'est pas installé.

    Example:
        >>> texts = inferer_trocr([img1, img2])
    """
    from pathlib import Path as _Path
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch

    is_peft_ckpt = _Path(str(model_path)).is_dir() and (
        _Path(str(model_path)) / "adapter_config.json"
    ).exists()

    if is_peft_ckpt:
        from peft import PeftModel
        processor = TrOCRProcessor.from_pretrained(MODELE_TROCR)
        base = VisionEncoderDecoderModel.from_pretrained(MODELE_TROCR)
        base.config.pad_token_id = processor.tokenizer.pad_token_id
        base.config.decoder_start_token_id = processor.tokenizer.bos_token_id
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    transcriptions = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pixel_values = processor(
            images=batch, return_tensors="pt"
        ).pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixel_values=pixel_values)
        transcriptions.extend(
            processor.batch_decode(ids, skip_special_tokens=True)
        )

    return transcriptions


def evaluer_baseline_trocr(
    manifest_path: Path,
    max_lignes: Optional[int] = 200,
    batch_size: int = 4,
    journal_path: Path = Path("experiments/journal.jsonl"),
) -> dict:
    """Évalue TrOCR sans fine-tuning (baseline zéro-shot) sur un manifest.

    Calcule CER_global, CER_abbrev et CER_no_abbrev pour quantifier l'impact
    des abréviations avant tout entraînement.

    Args:
        manifest_path: Chemin vers un fichier manifest JSON (val ou test).
        max_lignes: Nombre max de lignes à évaluer (défaut 200 pour rapidité).
        batch_size: Taille des batchs d'inférence.
        journal_path: Chemin du journal JSONL pour enregistrer le run.

    Returns:
        Dict de résultats : ``CER_global``, ``CER_abbrev``, ``CER_no_abbrev``,
        ``delta_CER``, ``nb_lignes``.

    Example:
        >>> results = evaluer_baseline_trocr(Path("data/lignes/manifest_val.json"))
    """
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if max_lignes:
        items = items[:max_lignes]

    images = [Image.open(item["image_path"]).convert("RGB") for item in items]
    references = [item["text"] for item in items]
    has_abbreviations = [item["has_abbreviation"] for item in items]

    print(f"[info] Inférence TrOCR baseline sur {len(images)} lignes ...")
    predictions = inferer_trocr(images, batch_size=batch_size)

    results = evaluer_stratifie(predictions, references, has_abbreviations)
    results["nb_lignes"] = len(items)
    results["modele"] = MODELE_TROCR
    results["type"] = "baseline_zero_shot"

    log_experience(
        journal_path=journal_path,
        run_name="trocr_baseline",
        hyperparams={"model": MODELE_TROCR, "max_lignes": max_lignes},
        metrics={
            "CER_global": results["CER_global"],
            "CER_abbrev": results["CER_abbrev"],
            "CER_no_abbrev": results["CER_no_abbrev"],
        },
    )
    return results


# ---------------------------------------------------------------------------
# Fine-tuning TrOCR + LoRA
# ---------------------------------------------------------------------------

class _DatasetLignesHTR:
    """Dataset PyTorch pour le fine-tuning TrOCR sur les lignes de manuscrits."""

    def __init__(self, manifest_path: Path, processor, max_target_length: int = 128):
        import torch
        self._torch = torch
        self.items = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze()

        labels = self.processor.tokenizer(
            item["text"],
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze()
        # Remplacer le padding par -100 pour ignorer dans la loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def fine_tuner_trocr_lora(
    train_manifest: Path,
    val_manifest: Path,
    lora_r: int = 8,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    patience: int = 3,
    max_val_lignes: Optional[int] = None,
    checkpoint_dir: Path = CHECKPOINTS_DIR,
    journal_path: Path = Path("experiments/journal.jsonl"),
    seed: int = 42,
) -> dict:
    """Fine-tune TrOCR avec LoRA sur le corpus CREMMA Médiéval.

    Implémente l'early stopping sur CER val. Log toutes les epochs dans le
    journal JSONL pour traçabilité complète de l'expérience.

    Args:
        train_manifest: Manifest JSON du train set.
        val_manifest: Manifest JSON du val set.
        lora_r: Rang LoRA (8 ou 16 selon le plan). Contrôle le nombre de
            paramètres entraînables.
        epochs: Nombre maximum d'epochs (early stopping avant si patience atteinte).
        batch_size: Taille des batchs.
        learning_rate: Taux d'apprentissage initial.
        patience: Nombre d'epochs sans amélioration avant arrêt.
        checkpoint_dir: Dossier de sauvegarde des checkpoints.
        journal_path: Chemin du journal JSONL.
        seed: Graine de reproductibilité.

    Returns:
        Dict avec ``meilleur_cer_val``, ``epoch_arret``, ``checkpoint_path``.

    Raises:
        ImportError: Si ``torch``, ``transformers`` ou ``peft`` ne sont pas installés.

    Example:
        >>> results = fine_tuner_trocr_lora(
        ...     Path("data/lignes/manifest_train.json"),
        ...     Path("data/lignes/manifest_val.json"),
        ...     lora_r=8,
        ... )
    """
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    fixer_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    print(f"[info] Fine-tuning TrOCR LoRA r={lora_r} sur {device} | fp16={use_fp16}")

    processor = TrOCRProcessor.from_pretrained(MODELE_TROCR)
    model = VisionEncoderDecoderModel.from_pretrained(MODELE_TROCR)

    # Requis par les nouvelles versions de transformers
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id

    # Configuration LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_r * 4,
        target_modules="all-linear",
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.to(device)
    model.print_trainable_parameters()

    # Datasets
    train_ds = _DatasetLignesHTR(train_manifest, processor)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    val_items = json.loads(val_manifest.read_text(encoding="utf-8"))
    if max_val_lignes:
        val_items = val_items[:max_val_lignes]
    val_images = [Image.open(it["image_path"]).convert("RGB") for it in val_items]
    val_refs = [it["text"] for it in val_items]
    val_flags = [it["has_abbreviation"] for it in val_items]

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    epochs_sans_amelioration = 0
    meilleur_chemin = checkpoint_dir / f"trocr_lora_r{lora_r}_best"

    for epoch in range(1, epochs + 1):
        # -- Train --
        model.train()
        pertes = []
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pertes.append(loss.item())

        perte_moy = float(np.mean(pertes))

        # -- Val CER (inférence avec le modèle courant, sans charger un second modèle) --
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(val_images), batch_size):
                batch_imgs = val_images[i : i + batch_size]
                pixel_values = processor(
                    images=batch_imgs, return_tensors="pt"
                ).pixel_values.to(device)
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    generated_ids = model.generate(pixel_values=pixel_values)
                preds.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))

        stats_val = evaluer_stratifie(preds, val_refs, val_flags)
        cer_val = stats_val["CER_global"]

        print(f"  Epoch {epoch:2d} | loss={perte_moy:.4f} | "
              f"CER_val={cer_val:.4f} | "
              f"CER_abbrev={stats_val['CER_abbrev']} | "
              f"CER_no_abbrev={stats_val['CER_no_abbrev']}")

        log_experience(
            journal_path=journal_path,
            run_name=f"trocr_lora_r{lora_r}_epoch{epoch}",
            hyperparams={"lora_r": lora_r, "lr": learning_rate, "batch_size": batch_size},
            metrics={**stats_val, "loss_train": perte_moy},
        )

        if cer_val < best_cer:
            best_cer = cer_val
            epochs_sans_amelioration = 0
            model.save_pretrained(str(meilleur_chemin))
            processor.save_pretrained(str(meilleur_chemin))
        else:
            epochs_sans_amelioration += 1
            if epochs_sans_amelioration >= patience:
                print(f"[info] Early stopping à l'epoch {epoch} (patience={patience})")
                break

    return {
        "meilleur_cer_val": round(best_cer, 4),
        "epoch_arret": epoch,
        "checkpoint_path": str(meilleur_chemin),
        "lora_r": lora_r,
    }


# ---------------------------------------------------------------------------
# Point d'entrée : construction du dataset + baseline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    fixer_seeds(42)

    split_path = Path("data/split.json")
    if not split_path.exists():
        print("[erreur] data/split.json introuvable. Lancez d'abord : python -m src.data_loader")
        sys.exit(1)

    # 1. Construction du dataset
    print("=== Construction du dataset lignes ===")
    dataset = construire_dataset(split_path=split_path)

    # 2. Résumé axé problématique
    print("\n=== Répartition abréviations par split ===")
    print(f"{'Split':<8} {'Total':>7} {'Abrev.':>7} {'Taux':>7}")
    print("-" * 30)
    for split_name, lignes in dataset.items():
        nb_abrev = sum(1 for l in lignes if l["has_abbreviation"])
        taux = nb_abrev / max(1, len(lignes))
        print(f"{split_name:<8} {len(lignes):>7} {nb_abrev:>7} {taux:>7.1%}")

    # 3. Baseline TrOCR (optionnel — demande le téléchargement du modèle)
    manifest_val = LIGNES_DIR / "manifest_val.json"
    if manifest_val.exists() and "--baseline" in sys.argv:
        print("\n=== Baseline TrOCR (zero-shot) ===")
        results = evaluer_baseline_trocr(manifest_val, max_lignes=100)
        print(f"CER global    : {results['CER_global']:.4f}")
        print(f"CER abreviees : {results['CER_abbrev']}")
        print(f"CER normales  : {results['CER_no_abbrev']}")
        print(f"Delta CER     : {results['delta_CER']} (reponse a la problematique)")