import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import yaml
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

# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------


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


def gliner_to_hipe_label(label: str) -> Optional[str]:
    mapping = {
        "person": "pers",
        "location": "loc",
        "organization": "org",
        "product": "prod",
        "time": "time",
    }
    return mapping.get(label)


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------


def resolve_dataset_dir(data_root: str, dataset: str, language: Optional[str]) -> Path:
    if language:
        dataset_dir = Path(data_root) / dataset / language
    else:
        dataset_dir = Path(data_root) / dataset

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
        ".py",
        ".yaml",
        ".yml",
        ".txt",
    }

    if path.suffix.lower() in ignored_suffixes:
        return False

    return True


def is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return "test" in name or name.startswith("test") or name.endswith(".test")


def is_train_or_dev_file(path: Path) -> bool:
    name = path.name.lower()

    return (
        "train" in name
        or "dev" in name
        or "valid" in name
        or name.endswith("train.fr")
        or name.endswith("dev.fr")
    )


def resolve_train_test_files(
    data_root: str,
    dataset: str,
    language: Optional[str],
    test_file: Optional[str],
) -> Tuple[List[Path], Path]:
    dataset_dir = resolve_dataset_dir(data_root, dataset, language)

    files = sorted([p for p in dataset_dir.rglob("*") if is_readable_data_file(p)])

    if not files:
        raise FileNotFoundError(f"No data files found in {dataset_dir}")

    print("=" * 80)
    print("DATASET DISCOVERY")
    print("=" * 80)
    print("Dataset directory:", dataset_dir)
    print("Files:")
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
        candidates = [p for p in files if is_test_file(p)]

        if not candidates:
            raise FileNotFoundError(
                f"No test file found in {dataset_dir}. Pass it with config data.test_file."
            )

        if len(candidates) > 1:
            print("Warning: multiple test candidates found. Using first:")
            for p in candidates:
                print(" -", p)

        test_path = candidates[0]

    train_paths = [p for p in files if p != test_path and is_train_or_dev_file(p)]

    if not train_paths:
        print("Warning: no explicit train/dev files found.")
        print("Using all non-test files as training files.")
        train_paths = [p for p in files if p != test_path]

    if not train_paths:
        raise FileNotFoundError("No training files found.")

    return train_paths, test_path


# ---------------------------------------------------------------------
# HIPE reader
# ---------------------------------------------------------------------


def read_hipe_tsv(
    path: Path,
    label_column: str,
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
                        f"Column {label_column} not found in {path}. "
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


# ---------------------------------------------------------------------
# BIO to GLiNER
# ---------------------------------------------------------------------


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


def convert_to_gliner(examples: List[Dict]) -> List[Dict]:
    converted = []

    for ex in examples:
        item = hipe_example_to_gliner(ex)
        if item is not None:
            converted.append(item)

    return converted


def print_stats(examples: List[Dict], title: str):
    counter = Counter()

    for ex in examples:
        for _, _, label in ex["ner"]:
            counter[label] += 1

    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Examples:", len(examples))
    for label, count in counter.most_common():
        print(f"{label}\t{count}")


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


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
    offsets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    overlapping = []

    for i, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            continue
        overlapping.append(i)

    if not overlapping:
        return None

    return overlapping[0], overlapping[-1]


def predictions_to_bio(
    entities: List[Dict],
    tokens: List[str],
    offsets: List[Tuple[int, int]],
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

        span = char_span_to_token_span(int(start_char), int(end_char), offsets)

        if span is None:
            continue

        start_tok, end_tok = span
        span_indices = set(range(start_tok, end_tok + 1))

        if occupied.intersection(span_indices):
            continue

        labels[start_tok] = f"B-{hipe_type}"

        for i in range(start_tok + 1, end_tok + 1):
            labels[i] = f"I-{hipe_type}"

        occupied.update(span_indices)

    return labels


@torch.no_grad()
def evaluate(
    model,
    examples: List[Dict],
    labels: List[str],
    threshold: float,
    output_path: Path,
):
    gold_sequences = []
    pred_sequences = []
    rows = []

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

        pred = predictions_to_bio(entities, tokens, offsets)

        gold_sequences.append(gold)
        pred_sequences.append(pred)

        rows.append(
            {
                "text": text,
                "tokens": tokens,
                "gold": gold,
                "pred": pred,
                "entities": entities,
                "document_id": ex.get("document_id"),
                "date": ex.get("date"),
                "source_file": ex.get("source_file"),
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "precision": precision_score(gold_sequences, pred_sequences),
        "recall": recall_score(gold_sequences, pred_sequences),
        "f1": f1_score(gold_sequences, pred_sequences),
        "classification_report": classification_report(gold_sequences, pred_sequences),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    hub_cfg = config.get("hub", {})

    seed = int(train_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = model_cfg.get(
        "labels", ["person", "location", "organization", "product", "time"]
    )

    print("=" * 80)
    print("ChronoGLiNER fine-tuning")
    print("=" * 80)
    print("Config:", args.config)
    print("Model:", model_cfg["model_name"])
    print("Output:", output_dir)
    print("Labels:", labels)

    train_paths, test_path = resolve_train_test_files(
        data_root=data_cfg.get("data_root", "data"),
        dataset=data_cfg["dataset"],
        language=data_cfg.get("language"),
        test_file=data_cfg.get("test_file"),
    )

    train_raw = []
    for path in train_paths:
        train_raw.extend(
            read_hipe_tsv(
                path,
                label_column=data_cfg.get("label_column", "NE-COARSE-LIT"),
                keep_scope=bool(data_cfg.get("keep_scope", False)),
            )
        )

    test_raw = read_hipe_tsv(
        test_path,
        label_column=data_cfg.get("label_column", "NE-COARSE-LIT"),
        keep_scope=bool(data_cfg.get("keep_scope", False)),
    )

    print("=" * 80)
    print("DATASET SIZE")
    print("=" * 80)
    print("Train examples:", len(train_raw))
    print("Test examples:", len(test_raw))

    train_split_raw, dev_split_raw = train_test_split(
        train_raw,
        test_size=float(data_cfg.get("dev_size", 0.1)),
        random_state=seed,
        shuffle=True,
    )

    train_data = convert_to_gliner(train_split_raw)
    eval_data = convert_to_gliner(dev_split_raw)
    test_gliner = convert_to_gliner(test_raw)

    print_stats(train_data, "TRAIN GLiNER DATA")
    print_stats(eval_data, "DEV GLiNER DATA")
    print_stats(test_gliner, "TEST GLiNER DATA")

    train_json = output_dir / "train_gliner.json"
    dev_json = output_dir / "dev_gliner.json"
    test_json = output_dir / "test_gliner.json"

    with train_json.open("w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with dev_json.open("w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    with test_json.open("w", encoding="utf-8") as f:
        json.dump(test_gliner, f, indent=2, ensure_ascii=False)

    metadata = {
        "config": args.config,
        "model_name": model_cfg["model_name"],
        "data": data_cfg,
        "training": train_cfg,
        "labels": labels,
        "train_files": [str(p) for p in train_paths],
        "test_file": str(test_path),
    }

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print("Loading model")
    print("=" * 80)

    model = GLiNER.from_pretrained(model_cfg["model_name"])

    print("=" * 80)
    print("Fine-tuning")
    print("=" * 80)

    train_kwargs = {
        "train_dataset": train_data,
        "eval_dataset": eval_data,
        "output_dir": str(output_dir / "model"),
        "max_steps": int(train_cfg.get("max_steps", 3000)),
        "per_device_train_batch_size": int(
            train_cfg.get("per_device_train_batch_size", 4)
        ),
        "learning_rate": float(train_cfg.get("learning_rate", 5e-6)),
    }

    if "eval_every" in train_cfg:
        train_kwargs["eval_every"] = int(train_cfg["eval_every"])

    if "save_steps" in train_cfg:
        train_kwargs["save_steps"] = int(train_cfg["save_steps"])

    if "warmup_ratio" in train_cfg:
        train_kwargs["warmup_ratio"] = float(train_cfg["warmup_ratio"])

    if "weight_decay" in train_cfg:
        train_kwargs["weight_decay"] = float(train_cfg["weight_decay"])

    if bool(train_cfg.get("fp16", False)):
        train_kwargs["fp16"] = True

    if bool(train_cfg.get("bf16", False)):
        train_kwargs["bf16"] = True

    model.train_model(**train_kwargs)

    model_dir = output_dir / "model"

    print("=" * 80)
    print("Reloading fine-tuned model")
    print("=" * 80)

    ft_model = GLiNER.from_pretrained(str(model_dir))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model.to(device)
    ft_model.eval()

    threshold = float(train_cfg.get("threshold", 0.5))

    metrics = evaluate(
        model=ft_model,
        examples=test_raw,
        labels=labels,
        threshold=threshold,
        output_path=output_dir / "test_predictions.jsonl",
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
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "classification_report": metrics["classification_report"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    if bool(hub_cfg.get("push_to_hub", False)):
        hub_model_id = hub_cfg.get("hub_model_id")

        if not hub_model_id:
            raise ValueError("hub.push_to_hub is true but hub.hub_model_id is missing.")

        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            repo_id=hub_model_id,
            folder_path=str(model_dir),
            repo_type="model",
        )

        print("Uploaded to:", hub_model_id)


if __name__ == "__main__":
    main()
