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

DEFAULT_EVAL_LABELS = [
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
# Dataset discovery
# -------------------------


def resolve_dataset_dir(
    data_root: str,
    dataset: str,
    language: Optional[str] = None,
) -> Path:
    data_root = Path(data_root)

    if language:
        dataset_dir = data_root / dataset / language
    else:
        dataset_dir = data_root / dataset

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return dataset_dir


def is_readable_data_file(path: Path) -> bool:
    if not path.is_file():
        return False

    if path.name.startswith("."):
        return False

    ignored_suffixes = {
        ".json",
        ".jsonl",
        ".md",
        ".txt",
        ".py",
        ".DS_Store",
    }

    if path.suffix in ignored_suffixes:
        return False

    return True


def is_test_file(path: Path) -> bool:
    name = path.name.lower()

    return "test" in name or name.startswith("test") or name.endswith(".test")


def is_train_file(path: Path) -> bool:
    name = path.name.lower()

    return (
        "train" in name
        or name.startswith("train")
        or name.endswith(".train")
        or name.endswith("train.fr")
        or "dev" in name
        or "valid" in name
    )


def resolve_train_test_files_from_dataset(
    data_root: str,
    dataset: str,
    language: Optional[str] = None,
    test_file: Optional[str] = None,
) -> Tuple[List[Path], Path]:
    dataset_dir = resolve_dataset_dir(
        data_root=data_root,
        dataset=dataset,
        language=language,
    )

    files = sorted([p for p in dataset_dir.rglob("*") if is_readable_data_file(p)])

    if not files:
        raise FileNotFoundError(f"No data files found in {dataset_dir}")

    print("=" * 80)
    print("DATASET DISCOVERY")
    print("=" * 80)
    print("Dataset directory:", dataset_dir)
    print("Found files:")
    for p in files:
        print(" -", p)

    if test_file:
        requested = Path(test_file)

        matches = [
            p
            for p in files
            if p.name == requested.name or str(p).endswith(str(test_file))
        ]

        if not matches:
            raise FileNotFoundError(
                f"Could not find requested test file {test_file} in {dataset_dir}"
            )

        test_path = matches[0]
    else:
        test_candidates = [p for p in files if is_test_file(p)]

        if not test_candidates:
            raise FileNotFoundError(
                f"No test file found in {dataset_dir}. "
                f"Pass it manually with --test_file."
            )

        if len(test_candidates) > 1:
            print("Warning: multiple test candidates found. Using first:")
            for p in test_candidates:
                print(" -", p)

        test_path = test_candidates[0]

    train_paths = [p for p in files if p != test_path and is_train_file(p)]

    if not train_paths:
        print("Warning: no explicit train/dev file found.")
        print("Using all non-test files as training files.")
        train_paths = [p for p in files if p != test_path]

    if not train_paths:
        raise FileNotFoundError(f"No training files found in {dataset_dir}")

    return train_paths, test_path


# -------------------------
# HIPE TSV reading
# -------------------------


def read_hipe_tsv(
    path: Path,
    label_column: str = DEFAULT_LABEL_COLUMN,
    keep_scope: bool = False,
) -> List[Dict]:
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


# -------------------------
# BIO to GLiNER spans
# -------------------------


def bio_to_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
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
# Evaluation utilities
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
# GLiNER training
# -------------------------


def train_with_gliner_api(
    model,
    train_data: List[Dict],
    dev_data: List[Dict],
    output_dir: Path,
    args,
):
    try:
        from gliner.data_processing.collator import DataCollator
        from gliner.training import Trainer, TrainingArguments
    except Exception as e:
        raise ImportError(
            "Could not import GLiNER training classes. "
            "The converted train/dev/test JSON files were saved, so you can use them "
            "with the official GLiNER fine-tuning script if needed."
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
        description="Fine-tune ChronoGLiNER on HIPE-style historical NER data."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset folder under data_root, e.g. hipe2020, hipe2022.",
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language subfolder, e.g. fr, de, en.",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Optional test file name/path. If omitted, the script searches for a file containing 'test'.",
    )

    parser.add_argument(
        "--label_column",
        type=str,
        default=DEFAULT_LABEL_COLUMN,
        help="TSV column to use as gold NER labels.",
    )

    parser.add_argument(
        "--keep_scope",
        action="store_true",
        help="Keep scope labels instead of mapping them to O.",
    )

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
        help="Fraction of training examples used as dev.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Evaluation threshold after fine-tuning.",
    )

    parser.add_argument("--fp16", action="store_true")

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
    print("Dataset:", args.dataset)
    print("Language:", args.language)
    print("Data root:", args.data_root)
    print("Output dir:", output_dir)
    print("Label column:", args.label_column)
    print("Eval labels:", eval_labels)

    train_paths, test_path = resolve_train_test_files_from_dataset(
        data_root=args.data_root,
        dataset=args.dataset,
        language=args.language,
        test_file=args.test_file,
    )

    train_raw = []

    for p in train_paths:
        train_raw.extend(
            read_hipe_tsv(
                p,
                label_column=args.label_column,
                keep_scope=args.keep_scope,
            )
        )

    test_raw = read_hipe_tsv(
        test_path,
        label_column=args.label_column,
        keep_scope=args.keep_scope,
    )

    print("=" * 80)
    print("DATASET FILES")
    print("=" * 80)
    print("Test file:", test_path)
    print("Train files:")
    for p in train_paths:
        print(" -", p)

    print("=" * 80)
    print("DATASET SIZE")
    print("=" * 80)
    print("Train examples:", len(train_raw))
    print("Test examples:", len(test_raw))

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
        "dataset": args.dataset,
        "language": args.language,
        "data_root": args.data_root,
        "label_column": args.label_column,
        "train_files": [str(p) for p in train_paths],
        "test_file": str(test_path),
        "labels": eval_labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "threshold": args.threshold,
        "dev_size": args.dev_size,
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
