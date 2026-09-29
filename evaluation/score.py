"""Scores predictions against the gold annotations (exact / relaxed / overlap P, R, F1).

    python evaluation/score.py --dataset_true data/test.llm.json \
        --dataset_pred predictions.json --dataset_name maccrobat --per_label --output scores.json
"""
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

from ner_metrics import evaluate_ner_performance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from label_sets import DATASET_NAMES, LABEL_SETS


def filter_entities(docs, labels):
    """Keeps the well-formed entities whose label is in `labels`, per document."""
    filtered = []
    for doc in docs:
        entities = doc.get("labels") if isinstance(doc.get("labels"), list) else []
        filtered.append([
            e for e in entities
            if isinstance(e, dict) and e.get("label") in labels and isinstance(e.get("text"), str)
        ])
    return filtered


def score(data_true, data_pred, labels, match_types):
    true_ents, pred_ents = filter_entities(data_true, labels), filter_entities(data_pred, labels)
    result = {}
    for match_type in match_types:
        p, r, f1 = evaluate_ner_performance(true_ents, pred_ents, match_type, default_when_empty=0.0)
        result[match_type] = {"p": p, "r": r, "f1": f1}
    return result


def main(args):
    with open(args.dataset_true, "r", encoding="utf8") as f:
        data_true = json.load(f)
    with open(args.dataset_pred, "r", encoding="utf8") as f:
        data_pred = json.load(f)

    labels = LABEL_SETS[args.dataset_name][args.label_set]
    results = {
        "dataset_true": args.dataset_true,
        "dataset_pred": args.dataset_pred,
        "label_set": args.label_set,
        "labels": labels,
        "overall": score(data_true, data_pred, labels, args.match_types),
    }
    if args.per_label:
        results["per_label"] = {label: score(data_true, data_pred, [label], args.match_types) for label in labels}

    rows = [("OVERALL", results["overall"])] + list(results.get("per_label", {}).items())
    print(f"{'':25s}" + "".join(f"{m + ' F1':>14s}" for m in args.match_types))
    for name, res in rows:
        print(f"{name:25s}" + "".join(f"{res[m]['f1']:14.4f}" for m in args.match_types))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset_true", type=str, required=True, help="Gold test dataset.")
    parser.add_argument("--dataset_pred", type=str, required=True, help="Predicted labels dataset.")
    parser.add_argument("--dataset_name", type=str, default="maccrobat", choices=DATASET_NAMES, help="Selects the label set.")
    parser.add_argument("--label_set", type=str, default="selected", choices=["selected", "all"],
                        help="selected = the labels reported in the paper; all = the full schema.")
    parser.add_argument("--match_types", nargs="+", default=["exact", "relaxed"], choices=["exact", "relaxed", "overlap"])
    parser.add_argument("--per_label", action="store_true", help="Also score each label separately.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file for the scores.")
    main(parser.parse_args())
