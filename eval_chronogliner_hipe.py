# /// script
# dependencies = [
#   "torch",
#   "gliner",
#   "transformers",
#   "seqeval",
#   "tqdm"
# ]
# ///

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from tqdm import tqdm
from gliner import GLiNER
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

TARGET_TEST_FILE = "data/hipe2020/fr/HIPE-2022-v2.1-hipe2020-test-fr.tsv"
LABEL_COLUMN = "NE-COARSE-LIT"


# -------------------------
# Label sets
# -------------------------

LABEL_VARIANTS = {
    "simple": [
        "person",
        "location",
        "organization",
        "product",
        "time",
    ],
    "historical": [
        "person: named individual, historical person, family name, title plus name",
        "location: city, country, region, street, building, historical place",
        "organization: council, court, administration, institution, company, newspaper",
        "product: named work, book, newspaper, object, artifact, human-made product",
        "time: date, year, month, day, historical time expression",
    ],
    "historical_fr": [
        "personne: individu nommé, personne historique, nom de famille, titre suivi d’un nom",
        "lieu: ville, pays, région, rue, bâtiment, lieu historique",
        "organisation: conseil, tribunal, administration, institution, entreprise, journal",
        "produit: œuvre nommée, livre, journal, objet, artefact, produit humain",
        "temps: date, année, mois, jour, expression temporelle historique",
    ],
}

LABEL_TO_HIPE = {
    # English simple
    "person": "pers",
    "location": "loc",
    "organization": "org",
    "product": "prod",
    "time": "time",
    # English descriptive labels
    "person: named individual, historical person, family name, title plus name": "pers",
    "location: city, country, region, street, building, historical place": "loc",
    "organization: council, court, administration, institution, company, newspaper": "org",
    "product: named work, book, newspaper, object, artifact, human-made product": "prod",
    "time: date, year, month, day, historical time expression": "time",
    # French descriptive labels
    "personne: individu nommé, personne historique, nom de famille, titre suivi d’un nom": "pers",
    "lieu: ville, pays, région, rue, bâtiment, lieu historique": "loc",
    "organisation: conseil, tribunal, administration, institution, entreprise, journal": "org",
    "produit: œuvre nommée, livre, journal, objet, artefact, produit humain": "prod",
    "temps: date, année, mois, jour, expression temporelle historique": "time",
}


# -------------------------
# HIPE reading and normalization
# -------------------------


def normalize_ner_label(label: str) -> str:
    if label in ["O", "_", ""]:
        return "O"

    if "-" not in label:
        return label

    prefix, ent_type = label.split("-", 1)

    prefix = prefix.upper()
    ent_type_norm = ent_type.lower()

    type_map = {
        "per": "pers",
        "pers": "pers",
        "loc": "loc",
        "building": "loc",
        "street": "loc",
        "org": "org",
        "prod": "prod",
        "humanprod": "prod",
        "object": "prod",
        "work": "prod",
        "time": "time",
        "date": "time",
        # optional: ignore scope
        "scope": "scope",
    }

    ent_type_norm = type_map.get(ent_type_norm, ent_type_norm)

    return f"{prefix}-{ent_type_norm}"


def read_hipe_tsv(path: Path, label_column: str = LABEL_COLUMN) -> List[Dict]:
    examples = []

    current_tokens = []
    current_labels = []
    current_misc = []

    header = None
    label_idx = None
    misc_idx = None
    current_doc_id = None
    current_date = None

    def flush_sentence():
        nonlocal current_tokens, current_labels, current_misc

        if current_tokens:
            examples.append(
                {
                    "tokens": current_tokens,
                    "labels": current_labels,
                    "misc": current_misc,
                    "document_id": current_doc_id,
                    "date": current_date,
                    "source_file": str(path),
                }
            )

        current_tokens = []
        current_labels = []
        current_misc = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                flush_sentence()
                continue

            if line.startswith("#"):
                if line.startswith("# hipe2022:document_id ="):
                    current_doc_id = line.split("=", 1)[1].strip()
                elif line.startswith("# hipe2022:date ="):
                    current_date = line.split("=", 1)[1].strip()
                continue

            cols = line.split("\t")

            if header is None:
                header = cols

                if label_column not in header:
                    raise ValueError(f"{label_column} not found in header of {path}")

                label_idx = header.index(label_column)
                misc_idx = header.index("MISC") if "MISC" in header else None
                continue

            if len(cols) <= label_idx:
                continue

            token = cols[0]
            label = normalize_ner_label(cols[label_idx])
            misc = (
                cols[misc_idx] if misc_idx is not None and len(cols) > misc_idx else "_"
            )

            current_tokens.append(token)
            current_labels.append(label)
            current_misc.append(misc)

            if "EndOfSentence" in misc:
                flush_sentence()

    flush_sentence()

    return examples


# -------------------------
# Text reconstruction
# -------------------------


def tokens_to_text_and_offsets(tokens: List[str], misc: Optional[List[str]] = None):
    """
    Reconstruct text from HIPE tokens and keep char offsets per token.

    Returns:
      text: str
      offsets: List[(start, end)]
    """

    text_parts = []
    offsets = []
    cursor = 0

    for i, token in enumerate(tokens):
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        end = cursor
        offsets.append((start, end))

        no_space = False
        if misc is not None and i < len(misc):
            no_space = "NoSpaceAfter" in misc[i]

        if not no_space and i != len(tokens) - 1:
            text_parts.append(" ")
            cursor += 1

    text = "".join(text_parts)
    return text, offsets


def char_span_to_token_span(
    start_char: int,
    end_char: int,
    token_offsets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    Convert character span to inclusive token span.

    Returns:
      (start_token, end_token), both inclusive.
    """

    overlapping = []

    for i, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            continue
        overlapping.append(i)

    if not overlapping:
        return None

    return overlapping[0], overlapping[-1]


# -------------------------
# BIO conversion
# -------------------------


def empty_bio(length: int) -> List[str]:
    return ["O"] * length


def predicted_entities_to_bio(
    entities: List[Dict],
    tokens: List[str],
    token_offsets: List[Tuple[int, int]],
) -> List[str]:
    """
    Convert GLiNER character-span predictions to BIO labels.

    If spans overlap, keep the higher-score span first.
    """

    labels = empty_bio(len(tokens))
    occupied = set()

    entities = sorted(
        entities,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    for ent in entities:
        raw_label = ent.get("label", "")
        mapped_label = LABEL_TO_HIPE.get(raw_label)

        if mapped_label is None:
            # Sometimes GLiNER may return a normalized/capitalized label.
            mapped_label = LABEL_TO_HIPE.get(raw_label.lower())

        if mapped_label is None:
            continue

        start_char = ent.get("start")
        end_char = ent.get("end")

        if start_char is None or end_char is None:
            continue

        token_span = char_span_to_token_span(
            int(start_char),
            int(end_char),
            token_offsets,
        )

        if token_span is None:
            continue

        start_tok, end_tok = token_span

        span_indices = set(range(start_tok, end_tok + 1))

        if occupied.intersection(span_indices):
            continue

        labels[start_tok] = f"B-{mapped_label}"

        for i in range(start_tok + 1, end_tok + 1):
            labels[i] = f"I-{mapped_label}"

        occupied.update(span_indices)

    return labels


# -------------------------
# Evaluation
# -------------------------


def evaluate_predictions(gold_sequences, pred_sequences) -> Dict:
    return {
        "precision": precision_score(gold_sequences, pred_sequences),
        "recall": recall_score(gold_sequences, pred_sequences),
        "f1": f1_score(gold_sequences, pred_sequences),
        "classification_report": classification_report(gold_sequences, pred_sequences),
    }


def run_variant(
    model,
    examples: List[Dict],
    label_variant_name: str,
    threshold: float,
    max_chars: int,
):
    gliner_labels = LABEL_VARIANTS[label_variant_name]

    gold_sequences = []
    pred_sequences = []

    all_raw_predictions = []

    for ex in tqdm(examples, desc=f"{label_variant_name} threshold={threshold}"):
        tokens = ex["tokens"]
        gold = ex["labels"]
        misc = ex.get("misc")

        text, offsets = tokens_to_text_and_offsets(tokens, misc)

        # GLiNER has length limits. For this first version, skip very long text,
        # or truncate text safely. Since HIPE examples are sentence-like, this
        # should usually be okay.
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]

        entities = model.predict_entities(
            text,
            gliner_labels,
            threshold=threshold,
        )

        pred_bio = predicted_entities_to_bio(
            entities=entities,
            tokens=tokens,
            token_offsets=offsets,
        )

        gold_sequences.append(gold)
        pred_sequences.append(pred_bio)

        all_raw_predictions.append(
            {
                "text": text,
                "tokens": tokens,
                "gold": gold,
                "pred": pred_bio,
                "entities": entities,
                "label_variant": label_variant_name,
                "threshold": threshold,
                "document_id": ex.get("document_id"),
                "date": ex.get("date"),
            }
        )

    metrics = evaluate_predictions(gold_sequences, pred_sequences)

    return metrics, all_raw_predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_file",
        type=str,
        default=TARGET_TEST_FILE,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="urchade/gliner_medium-v2.1",
    )

    parser.add_argument(
        "--label_variants",
        type=str,
        default="simple,historical,historical_fr",
        help="Comma-separated list: simple,historical,historical_fr",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.25,0.30,0.35,0.40,0.50",
        help="Comma-separated thresholds",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/chronogliner_hipe",
    )

    parser.add_argument(
        "--max_chars",
        type=int,
        default=2000,
        help="Maximum reconstructed sentence length in characters.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    test_file = Path(args.test_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_variants = [x.strip() for x in args.label_variants.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    print("=" * 80)
    print("ChronoGLiNER evaluation")
    print("=" * 80)
    print("Test file:", test_file)
    print("Model:", args.model_name)
    print("Label variants:", label_variants)
    print("Thresholds:", thresholds)
    print("Device:", args.device)

    examples = read_hipe_tsv(test_file)
    print("Examples:", len(examples))

    print("Loading GLiNER model...")
    model = GLiNER.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    summary = {}

    for variant_name in label_variants:
        if variant_name not in LABEL_VARIANTS:
            raise ValueError(
                f"Unknown label variant: {variant_name}. "
                f"Available: {list(LABEL_VARIANTS.keys())}"
            )

        for threshold in thresholds:
            run_name = f"{variant_name}_thr_{threshold:.2f}".replace(".", "p")

            metrics, raw_predictions = run_variant(
                model=model,
                examples=examples,
                label_variant_name=variant_name,
                threshold=threshold,
                max_chars=args.max_chars,
            )

            summary[run_name] = {
                "model_name": args.model_name,
                "label_variant": variant_name,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "classification_report": metrics["classification_report"],
            }

            print("=" * 80)
            print(run_name)
            print("=" * 80)
            print("Precision:", metrics["precision"])
            print("Recall:", metrics["recall"])
            print("F1:", metrics["f1"])
            print()
            print(metrics["classification_report"])

            with (output_dir / f"{run_name}_predictions.jsonl").open(
                "w",
                encoding="utf-8",
            ) as f:
                for item in raw_predictions:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            with (output_dir / f"{run_name}_results.json").open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(summary[run_name], f, indent=2, ensure_ascii=False)

    with (output_dir / "summary_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("BEST RUNS BY F1")
    print("=" * 80)

    ranked = sorted(
        summary.items(),
        key=lambda x: x[1]["f1"],
        reverse=True,
    )

    for name, result in ranked:
        print(
            name,
            "P=",
            round(result["precision"], 4),
            "R=",
            round(result["recall"], 4),
            "F1=",
            round(result["f1"], 4),
        )

    print(f"Saved summary to {output_dir / 'summary_results.json'}")


if __name__ == "__main__":
    main()
