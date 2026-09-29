"""McNemar test between two prediction files (e.g. baseline vs semantic prompts).

The paper's significance marks use --over-union (Yates-corrected chi-square, p < 0.01).

    python evaluation/mcnemar.py --gold data/test.llm.json --model1 baseline.json \
        --model2 semantic.json --dataset_name maccrobat --over-union
"""
import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Dict, Any, Set, Tuple, Optional

from statsmodels.stats.contingency_tables import mcnemar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompts"))
from label_sets import DATASET_NAMES, LABEL_SETS

Entity = Tuple[str, str]  # (label, normalized_text)


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize(s: str) -> str:
    # lowercase + trim + collapse internal whitespace
    return " ".join((s or "").lower().strip().split())


def extract_entity_set(doc: Dict[str, Any], allowed: Optional[Set[str]]) -> Set[Entity]:
    ents: Set[Entity] = set()
    for lab in doc.get("labels", []):
        label = lab.get("label")
        if allowed is not None and label not in allowed:
            continue
        txt = normalize(lab.get("text", ""))
        if txt:
            ents.add((label, txt))
    return ents


def build_table(
    gold_docs: List[Dict[str, Any]],
    m1_docs: List[Dict[str, Any]],
    m2_docs: List[Dict[str, Any]],
    labels_subset: Optional[List[str]] = None,
    include_spurious: bool = False,  # True: evaluate over the union G∪A∪D
):
    assert len(gold_docs) == len(m1_docs) == len(m2_docs), "Input lengths differ"
    allowed = set(labels_subset) if labels_subset is not None else None

    m11 = m00 = b = c = 0
    per_label: Dict[str, Dict[str, int]] = {}

    for g, a, d in zip(gold_docs, m1_docs, m2_docs):
        gset = extract_entity_set(g, allowed)
        aset = extract_entity_set(a, allowed)
        dset = extract_entity_set(d, allowed)

        cand = gset if not include_spurious else (gset | aset | dset)

        for (lab, txt) in cand:
            if lab not in per_label:
                per_label[lab] = {"m11": 0, "m00": 0, "b": 0, "c": 0, "n": 0}

            y = ((lab, txt) in gset)       # gold membership
            a_pred = ((lab, txt) in aset)  # model1 prediction
            d_pred = ((lab, txt) in dset)  # model2 prediction

            a_hit = (a_pred == y)          # correctness on this item
            d_hit = (d_pred == y)

            per_label[lab]["n"] += 1
            if a_hit and d_hit:
                m11 += 1; per_label[lab]["m11"] += 1
            elif a_hit and not d_hit:
                b += 1; per_label[lab]["b"] += 1
            elif (not a_hit) and d_hit:
                c += 1; per_label[lab]["c"] += 1
            else:
                m00 += 1; per_label[lab]["m00"] += 1

    table = [[m11, b],
             [c,   m00]]
    totals = {"m11": m11, "m00": m00, "b": b, "c": c, "n": m11 + m00 + b + c}
    return table, totals, per_label


def run_test(table, exact: bool, correction: bool) -> Tuple[float, float]:
    """Returns (statistic, p-value). Without discordant pairs (b + c = 0) there is no evidence
    of a difference; statsmodels would divide by zero there and report p = 0."""
    if table[0][1] + table[1][0] == 0:
        return 0.0, 1.0
    res = mcnemar(table, exact=exact, correction=correction)
    return float(res.statistic), float(res.pvalue)


def main(args):
    gold, m1, m2 = load_json(args.gold), load_json(args.model1), load_json(args.model2)
    labels = LABEL_SETS[args.dataset_name][args.label_set]
    correction = not args.no_correction

    table, totals, per_label = build_table(gold, m1, m2, labels, include_spurious=args.over_union)
    method = 'exact' if args.exact else ('chi-square w/ Yates' if correction else 'chi-square')

    statistic, p_value = run_test(table, args.exact, correction)
    print(f"=== Overall (micro over {'gold ∪ predicted' if args.over_union else 'gold'} entities) ===")
    print(f"n={totals['n']}  m11={totals['m11']}  m00={totals['m00']}  b(m10)={totals['b']}  c(m01)={totals['c']}")
    print(f"statistic={statistic}  p-value={p_value}  method={method}  "
          f"significant (p < {args.alpha}): {p_value < args.alpha}")

    print("\n=== By label ===")
    print(f"{'Label':30s}  {'n':>5s} {'m11':>5s} {'m00':>5s} {'b':>5s} {'c':>5s}  {'p-value':>10s}")
    by_label = {}
    for lab, cnts in sorted(per_label.items()):
        tbl = [[cnts['m11'], cnts['b']],
               [cnts['c'],   cnts['m00']]]
        _, p = run_test(tbl, args.exact, correction)
        by_label[lab] = {**cnts, "p_value": p}
        print(f"{lab:30s}  {cnts['n']:5d} {cnts['m11']:5d} {cnts['m00']:5d} {cnts['b']:5d} {cnts['c']:5d}  {p:10.6f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"model1": args.model1, "model2": args.model2, "method": method, "over_union": args.over_union,
                       **totals, "statistic": statistic, "p_value": p_value,
                       "significant": p_value < args.alpha, "per_label": by_label}, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", required=True)
    parser.add_argument("--model1", required=True, help="Predictions of the first system (e.g. baseline).")
    parser.add_argument("--model2", required=True, help="Predictions of the second system (e.g. semantic).")
    parser.add_argument("--dataset_name", default="maccrobat", choices=DATASET_NAMES, help="Selects the label set.")
    parser.add_argument("--label_set", default="selected", choices=["selected", "all"])
    parser.add_argument("--exact", action="store_true",
                        help="use exact binomial test (recommended when b+c is small)")
    parser.add_argument("--no-correction", action="store_true",
                        help="disable Yates correction (only relevant if --exact is not set)")
    parser.add_argument("--over-union", action="store_true",
                        help="Evaluate over the union of gold and predictions (counts spurious entities)")
    parser.add_argument("--alpha", type=float, default=0.01, help="Significance level.")
    parser.add_argument("--output", default=None, help="Optional JSON file for the result.")
    main(parser.parse_args())
