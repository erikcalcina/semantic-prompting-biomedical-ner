"""Cleans raw predictions before scoring (replaces the old hard-coded helper scripts).

* Drops entities with empty text: the prompts ask for "" when an entity is absent.
  If any entity is malformed, the sentence's predictions are discarded, as before.
* Renames "Biological Structure" to "Biological structure", the one casing fix used for
  the paper (the baseline few-shot prompt spells it that way). With --normalize_label_case
  every label is instead matched case-insensitively to the dataset's label names
  (e.g. "Disease" -> "DISEASE"); on the paper's MACCROBAT files that moves F1 by ~0.001.

    python scripts/postprocess.py --input raw.json --output clean.json --dataset_name maccrobat
"""
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from label_sets import DATASET_NAMES, LABEL_SETS

PAPER_CASING_FIX = {"Biological Structure": "Biological structure"}


def drop_empty(labels):
    try:
        return [entity for entity in labels if entity["text"] != '']
    except Exception:
        return []


def fix_casing(labels, mapping, case_insensitive):
    for entity in labels:
        if isinstance(entity, dict) and isinstance(entity.get("label"), str):
            key = entity["label"].lower() if case_insensitive else entity["label"]
            entity["label"] = mapping.get(key, entity["label"])
    return labels


def main(args):
    with open(args.input, "r", encoding="utf-8") as file:
        predictions = json.load(file)

    if args.normalize_label_case:
        mapping = {label.lower(): label for label in LABEL_SETS[args.dataset_name]["all"]}
    else:
        mapping = PAPER_CASING_FIX
    dropped = 0
    for pred in predictions:
        labels = pred.get("labels")
        before = len(labels) if isinstance(labels, list) else 0
        if not args.keep_empty:
            labels = drop_empty(labels)
        if not args.no_casing_fix and isinstance(labels, list):
            labels = fix_casing(labels, mapping, args.normalize_label_case)
        pred["labels"] = labels
        dropped += before - (len(labels) if isinstance(labels, list) else 0)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=4)
    print(f"Removed {dropped} entities; wrote {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Raw predictions from predict.py.")
    parser.add_argument("--output", required=True, help="Where to write the cleaned predictions.")
    parser.add_argument("--dataset_name", default="maccrobat", choices=DATASET_NAMES, help="Selects the canonical label names.")
    parser.add_argument("--keep_empty", action="store_true", help="Keep entities with empty text.")
    parser.add_argument("--no_casing_fix", action="store_true", help="Leave label casing as generated.")
    parser.add_argument("--normalize_label_case", action="store_true",
                        help="Match every label case-insensitively to the dataset's label names.")
    main(parser.parse_args())
