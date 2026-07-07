# /// script
# dependencies = [
#   "torch",
#   "gliner",
#   "transformers",
#   "seqeval",
#   "tqdm",
#   "scikit-learn",
#   "huggingface_hub"
# ]
# ///

import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from gliner import GLiNER
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

DEFAULT_LABEL_COLUMN = "NE-COARSE-LIT"

DEFAULT_LABELS = [
    "person",
    "location",
    "organization",
    "product",
    "time",
]


# -------------------------
# Label normalization
# -------------------------


def normalize_ner_label(label: str, keep_scope: bool = False) -> str:
    """
    Normalize mixed HIPE / historical NER labels.

    Examples:
      B-PER       -> B-pers
      I-ORG       -> I-org
      B-LOC       -> B-loc
      B-STREET    -> B-loc
      B-BUILDING  -> B-loc
      B-HumanProd -> B-prod
      B-object    -> B-prod
      B-work      -> B-prod
      B-date      -> B-time
    """

    if label in ["O", "_", ""]:
        return "O"

    if "-" not in label:
        return "O"

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
    }

    if keep_scope:
        type_map["scope"] = "scope"
    else:
        type_map["scope"] = "O"

    mapped = type_map.get(ent_type_norm, ent_type_norm)

    if mapped == "O":
        return "O"

    return f"{prefix}-{mapped}"


def hipe_to_gliner_label(label_type: str) -> Optional[str]:
    mapping = {
        "pers": "person",
        "loc": "location",
        "org": "organization",
        "prod": "product",
        "time": "time",
    }

    return mapping.get(label_type)


def gliner_to_hipe_label(gliner_label: str) -> Optional[str]:
    mapping = {
        "person": "pers",
        "location": "loc",
        "organization": "org",
        "product": "prod",
        "time": "time",
    }

    return mapping.get(gliner_label)


# -------------------------
# TSV reading
# -------------------------


def read_hipe_tsv(
    path: Path,
    label_column: str = DEFAULT_LABEL_COLUMN,
    keep_scope: bool = False,
) -> List[Dict]:
    """
    Reads a HIPE-style TSV file and returns sentence-level examples.

    Output example:
    {
      "tokens": [...],
      "labels": [...],
      "misc": [...],
      "document_id": "...",
      "date": "...",
      "source_file": "..."
    }
    """

    examples = []

    current_tokens = []
    current_labels = []
    current_misc = []

    current_doc_id = None
    current_date = None
    current_language = None
    current_dataset = None

    header = None
    label_idx = None
    misc_idx = None

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
                    "language": current_language,
                    "dataset": current_dataset,
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
                elif line.startswith("# hipe2022:language ="):
                    current_language = line.split("=", 1)[1].strip()
                elif line.startswith("# hipe2022:dataset ="):
                    current_dataset = line.split("=", 1)[1].strip()
                continue

            cols = line.split("\t")

            if header is None:
                header = cols

                if label_column not in header:
                    raise ValueError(
                        f"Column {label_column} not found in header of {path}. "
                        f"Available columns: {header}"
                    )

                label_idx = header.index(label_column)
                misc_idx = header.index("MISC") if "MISC" in header else None
                continue

            if len(cols) <= label_idx:
                continue

            token = cols[0]
            label = normalize_ner_label(cols[label_idx], keep_scope=keep_scope)
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


def parse_file_list(files_arg: Optional[str]) -> List[Path]:
    if not files_arg:
        return []

    files = []
    for item in files_arg.split(","):
        item = item.strip()
        if item:
            files.append(Path(item))

    return files


def resolve_train_test_files(
    data_dir: Optional[str],
    train_files_arg: Optional[str],
    test_file_arg: str,
) -> Tuple[List[Path], Path]:
    """
    Supports two modes:

    1. Explicit:
       --train_files file1.tsv,file2.tsv --test_file test.tsv

    2. Directory:
       --data_dir data --test_file data/hipe2020/fr/test.tsv
       Then all .tsv under data_dir except test_file are used for training.
    """

    explicit_train_files = parse_file_list(train_files_arg)

    test_file = Path(test_file_arg)

    if explicit_train_files:
        train_files = explicit_train_files

        if not test_file.exists():
            raise FileNotFoundError(f"Test file does not exist: {test_file}")

        for p in train_files:
            if not p.exists():
                raise FileNotFoundError(f"Train file does not exist: {p}")

        return train_files, test_file

    if not data_dir:
        raise ValueError("You must provide either --train_files or --data_dir.")

    data_dir_path = Path(data_dir)

    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir_path}")

    all_tsvs = sorted(data_dir_path.rglob("*.tsv"))

    if not all_tsvs:
        raise FileNotFoundError(f"No .tsv files found under {data_dir_path}")

    test_file_str = str(test_file_arg)

    matching_test_files = [
        p
        for p in all_tsvs
        if p.name == Path(test_file_str).name
        or str(p).endswith(test_file_str)
        or p.resolve() == Path(test_file_str).resolve()
    ]

    if not matching_test_files:
        raise FileNotFoundError(
            f"Could not find test file {test_file_arg} under {data_dir_path}"
        )

    resolved_test_file = matching_test_files[0]
    train_files = [p for p in all_tsvs if p != resolved_test_file]

    return train_files, resolved_test_file


def load_dataset(
    data_dir: Optional[str],
    train_files: Optional[str],
    test_file: str,
    label_column: str,
    keep_scope: bool = False,
):
    train_paths, test_path = resolve_train_test_files(
        data_dir=data_dir,
        train_files_arg=train_files,
        test_file_arg=test_file,
    )

    print("=" * 80)
    print("DATASET FILES")
    print("=" * 80)
    print("Test file:", test_path)
    print("Train files:")
    for p in train_paths:
        print(" -", p)

    train_examples = []

    for p in train_paths:
        train_examples.extend(
            read_hipe_tsv(
                p,
                label_column=label_column,
                keep_scope=keep_scope,
            )
        )

    test_examples = read_hipe_tsv(
        test_path,
        label_column=label_column,
        keep_scope=keep_scope,
    )

    print("=" * 80)
    print("DATASET SIZE")
    print("=" * 80)
    print("Train examples:", len(train_examples))
    print("Test examples:", len(test_examples))

    return train_examples, test_examples, train_paths, test_path


# -------------------------
# BIO <-> span conversion
# -------------------------


def bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert BIO labels to inclusive token spans:
      (start_token, end_token, entity_type)
    """

    spans = []
    start = None
    ent_type = None

    for i, label in enumerate(labels):
        if label == "O" or label == "_":
            if start is not None:
                spans.append((start, i - 1, ent_type))
                start = None
                ent_type = None
            continue

        if "-" not in label:
            if start is not None:
                spans.append((start, i - 1, ent_type))
                start = None
                ent_type = None
            continue

        prefix, current_type = label.split("-", 1)

        if prefix == "B":
            if start is not None:
                spans.append((start, i - 1, ent_type))

            start = i
            ent_type = current_type

        elif prefix == "I":
            if start is None:
                start = i
                ent_type = current_type
            elif current_type != ent_type:
                spans.append((start, i - 1, ent_type))
                start = i
                ent_type = current_type

    if start is not None:
        spans.append((start, len(labels) - 1, ent_type))

    return spans


def hipe_example_to_gliner(example: Dict) -> Optional[Dict]:
    """
    Convert sentence-level HIPE example to GLiNER training format:

    {
      "tokenized_text": ["Charlotte", "née", "Bourgoin"],
      "ner": [[0, 2, "person"]]
    }
    """

    tokens = example["tokens"]
    labels = example["labels"]

    if not tokens:
        return None

    spans = bio_to_spans(labels)

    ner = []

    for start, end, ent_type in spans:
        gliner_label = hipe_to_gliner_label(ent_type)

        if gliner_label is None:
            continue

        ner.append([start, end, gliner_label])

    return {
        "tokenized_text": tokens,
        "ner": ner,
    }


def convert_examples_to_gliner(examples: List[Dict]) -> List[Dict]:
    converted = []

    for ex in examples:
        item = hipe_example_to_gliner(ex)
        if item is not None:
            converted.append(item)

    return converted


def print_dataset_stats(examples: List[Dict], title: str):
    counter = Counter()

    for ex in examples:
        for start, end, label in ex["ner"]:
            counter[label] += 1

    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Examples:", len(examples))
    print("Entity counts:")
    for label, count in counter.most_common():
        print(f"{label}\t{count}")


# -------------------------
# Text reconstruction for evaluation
# -------------------------


def tokens_to_text_and_offsets(
    tokens: List[str],
    misc: Optional[List[str]] = None,
) -> Tuple[str, List[Tuple[int, int]]]:
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

    return "".join(text_parts), offsets


def char_span_to_token_span(
    start_char: int,
    end_char: int,
    token_offsets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
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


def predicted_entities_to_bio(
    entities: List[Dict],
    tokens: List[str],
    token_offsets: List[Tuple[int, int]],
) -> List[str]:
    labels = ["O"] * len(tokens)
    occupied = set()

    entities = sorted(
        entities,
        key=lambda x: float(x.get("score", 0.0)),
        reverse=True,
    )

    for ent in entities:
        raw_label = ent.get("label", "")
        hipe_type = gliner_to_hipe_label(raw_label)

        if hipe_type is None:
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

        labels[start_tok] = f"B-{hipe_type}"

        for i in range(start_tok + 1, end_tok + 1):
            labels[i] = f"I-{hipe_type}"

        occupied.update(span_indices)

    return labels


# -------------------------
# Evaluation
# -------------------------


@torch.no_grad()
def evaluate_gliner(
    model,
    examples: List[Dict],
    threshold: float,
    labels: List[str],
    output_predictions_path: Optional[Path] = None,
):
    gold_sequences = []
    pred_sequences = []
    prediction_rows = []

    for ex in tqdm(examples, desc=f"Evaluating threshold={threshold}"):
        tokens = ex["tokens"]
        gold = ex["labels"]
        misc = ex.get("misc")

        text, offsets = tokens_to_text_and_offsets(tokens, misc)

        entities = model.predict_entities(
            text,
            labels,
            threshold=threshold,
        )

        pred_bio = predicted_entities_to_bio(
            entities=entities,
            tokens=tokens,
            token_offsets=offsets,
        )

        gold_sequences.append(gold)
        pred_sequences.append(pred_bio)

        prediction_rows.append(
            {
                "text": text,
                "tokens": tokens,
                "gold": gold,
                "pred": pred_bio,
                "entities": entities,
                "document_id": ex.get("document_id"),
                "date": ex.get("date"),
                "source_file": ex.get("source_file"),
            }
        )

    metrics = {
        "precision": precision_score(gold_sequences, pred_sequences),
        "recall": recall_score(gold_sequences, pred_sequences),
        "f1": f1_score(gold_sequences, pred_sequences),
        "classification_report": classification_report(gold_sequences, pred_sequences),
    }

    if output_predictions_path is not None:
        with output_predictions_path.open("w", encoding="utf-8") as f:
            for row in prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics


# -------------------------
# Generic training wrapper
# -------------------------


def train_with_gliner_api(
    model,
    train_data: List[Dict],
    dev_data: List[Dict],
    output_dir: Path,
    args,
):
    """
    GLiNER's training API has changed across versions.

    This function first tries the common GLiNER Trainer API.
    If your installed GLiNER version differs, the script will fail here,
    but all dataset conversion files will already be saved.
    """

    try:
        from gliner.data_processing.collator import DataCollator
        from gliner.training import Trainer, TrainingArguments
    except Exception as e:
        raise ImportError(
            "Could not import GLiNER training classes. "
            "Your installed gliner version may have a different training API. "
            "The converted train/dev/test JSON files were saved, so you can still "
            "use them with the GLiNER official fine-tuning script."
        ) from e

    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        others_lr=args.learning_rate,
        others_weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(str(output_dir / "model"))

    try:
        model.save_pretrained(str(output_dir / "model"))
    except Exception:
        pass

    return output_dir / "model"


# -------------------------
# Main
# -------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ChronoGLiNER on a configurable HIPE-style TSV dataset."
    )

    # Dataset arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing .tsv files. Used if --train_files is not provided.",
    )

    parser.add_argument(
        "--train_files",
        type=str,
        default=None,
        help="Comma-separated list of training TSV files. Overrides --data_dir training discovery.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test TSV file.",
    )

    parser.add_argument(
        "--label_column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="TSV column to use as NER labels.",
    )

    parser.add_argument(
        "--keep_scope",
        action="store_true",
        help="Keep scope labels instead of mapping them to O. Not used by default labels.",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="urchade/gliner_medium-v2.1",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/chronogliner_ft",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="person,location,organization,product,time",
        help="Comma-separated GLiNER labels used at evaluation.",
    )

    # Training arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument(
        "--dev_size",
        type=float,
        default=0.1,
        help="Fraction of training examples used as dev if --dev_files is not provided.",
    )

    parser.add_argument(
        "--dev_files",
        type=str,
        default=None,
        help="Optional comma-separated dev TSV files.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Evaluation threshold after fine-tuning.",
    )

    parser.add_argument("--fp16", action="store_true")

    # Hub upload
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload final model folder to Hugging Face Hub.",
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Example: emanuelaboros/chrono-gliner-ft",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    print("=" * 80)
    print("ChronoGLiNER fine-tuning")
    print("=" * 80)
    print("Model:", args.model_name)
    print("Output dir:", output_dir)
    print("Label column:", args.label_column)
    print("Eval labels:", eval_labels)

    train_raw, test_raw, train_paths, test_path = load_dataset(
        data_dir=args.data_dir,
        train_files=args.train_files,
        test_file=args.test_file,
        label_column=args.label_column,
        keep_scope=args.keep_scope,
    )

    if args.dev_files:
        dev_paths = parse_file_list(args.dev_files)

        dev_raw = []
        for p in dev_paths:
            dev_raw.extend(
                read_hipe_tsv(
                    p,
                    label_column=args.label_column,
                    keep_scope=args.keep_scope,
                )
            )

        train_split_raw = train_raw
        dev_split_raw = dev_raw
    else:
        train_split_raw, dev_split_raw = train_test_split(
            train_raw,
            test_size=args.dev_size,
            random_state=args.seed,
            shuffle=True,
        )

    train_gliner = convert_examples_to_gliner(train_split_raw)
    dev_gliner = convert_examples_to_gliner(dev_split_raw)
    test_gliner = convert_examples_to_gliner(test_raw)

    print_dataset_stats(train_gliner, "TRAIN GLINER DATA")
    print_dataset_stats(dev_gliner, "DEV GLINER DATA")
    print_dataset_stats(test_gliner, "TEST GLINER DATA")

    with (output_dir / "train_gliner.json").open("w", encoding="utf-8") as f:
        json.dump(train_gliner, f, indent=2, ensure_ascii=False)

    with (output_dir / "dev_gliner.json").open("w", encoding="utf-8") as f:
        json.dump(dev_gliner, f, indent=2, ensure_ascii=False)

    with (output_dir / "test_gliner.json").open("w", encoding="utf-8") as f:
        json.dump(test_gliner, f, indent=2, ensure_ascii=False)

    metadata = {
        "model_name": args.model_name,
        "label_column": args.label_column,
        "train_files": [str(p) for p in train_paths],
        "test_file": str(test_path),
        "dev_files": args.dev_files,
        "labels": eval_labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Loading GLiNER model:", args.model_name)
    model = GLiNER.from_pretrained(args.model_name)
    model.train()

    print("Starting fine-tuning...")
    model_dir = train_with_gliner_api(
        model=model,
        train_data=train_gliner,
        dev_data=dev_gliner,
        output_dir=output_dir,
        args=args,
    )

    print("Reloading fine-tuned model from:", model_dir)
    ft_model = GLiNER.from_pretrained(str(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model.to(device)
    ft_model.eval()

    metrics = evaluate_gliner(
        model=ft_model,
        examples=test_raw,
        threshold=args.threshold,
        labels=eval_labels,
        output_predictions_path=output_dir / "test_predictions.jsonl",
    )

    print("=" * 80)
    print("ChronoGLiNER-FT TEST RESULTS")
    print("=" * 80)
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print()
    print(metrics["classification_report"])

    with (output_dir / "test_results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **metadata,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "classification_report": metrics["classification_report"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved results to {output_dir / 'test_results.json'}")

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--push_to_hub requires --hub_model_id")

        from huggingface_hub import HfApi

        print("Uploading model to Hub:", args.hub_model_id)

        api = HfApi()
        api.upload_folder(
            repo_id=args.hub_model_id,
            folder_path=str(model_dir),
            repo_type="model",
        )

        print("Upload complete.")


if __name__ == "__main__":
    main()
