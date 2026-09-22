"""The reply-intent evaluation harness.

Runs a set of predictors over the frozen eval split and prints one comparable report
for each. The point is the *ladder*, not any single number:

    majority class      the floor. Beat it or nothing else matters.
    rules only          subject/header rules for ooo + bounce. Often ~50% alone,
                        because most real replies are automated. Free, so anything
                        learned must beat rules-plus-majority, not just majority.
    laya base           EXPECTED TO LOOK BAD -- the model card puts the base
                        checkpoints below the majority-class baseline on typed
                        decisions (0.362 vs 0.461). This run proves the harness works,
                        it does not evaluate Laya.
    laya typed-decisions the real fine-tuning starting point.
    claude              the teacher. Its accuracy is both your current production
                        quality and the ceiling fine-tuning aims at.
    laya finetuned      the deliverable.

Usage:
    python3 src/evaluate.py --corpus data/interim/corpus.jsonl --predictors majority,rules
    python3 src/evaluate.py --corpus ... --predictors laya:typed-decisions,claude
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))
sys.path.insert(0, os.path.dirname(__file__))
import taxonomy  # noqa: E402
import metrics as M  # noqa: E402
from clean import detect_automated  # noqa: E402

LABELS = taxonomy.INTENTS


# --------------------------------------------------------------------------- predictors
# Each returns (predicted_label, probability_vector_over_LABELS) per record.

def predict_majority(records, train):
    top = Counter(r["labels"]["intent"] for r in train if r["labels"].get("intent")).most_common(1)
    top = top[0][0] if top else LABELS[0]
    one = [1.0 if l == top else 0.0 for l in LABELS]
    return [(top, one) for _ in records]


def predict_rules(records, train):
    """Header/subject rules for ooo + bounce, majority class otherwise."""
    fallback = predict_majority(records, train)[0][0]
    out = []
    for r, fb in zip(records, [fallback] * len(records)):
        lab = detect_automated(subject=r.get("subject", ""), sender=r.get("sender", ""),
                               headers=r.get("headers", "")) or fb
        conf = 0.97 if lab in ("ooo", "bounce") else 0.35
        rest = (1 - conf) / (len(LABELS) - 1)
        out.append((lab, [conf if l == lab else rest for l in LABELS]))
    return out


def predict_laya(records, train, checkpoint="typed-decisions", device=None, batch=16):
    """Laya, one forward pass per record answering every question at once."""
    import laya
    kwargs = {} if checkpoint in ("root", "english") else {"subfolder": checkpoint}
    agent = laya.load("convaiinnovations/laya", **kwargs)
    questions = {"intent": taxonomy.QUESTIONS["intent"]}
    out = []
    for r in records:
        res = agent.predict(taxonomy.state_for(r), questions)
        ans = res["answers"]["intent"]
        probs = ans.get("probabilities") or ans.get("probs")
        if isinstance(probs, dict):
            probs = [float(probs.get(l, 0.0)) for l in LABELS]
        elif probs is None:
            c = float(ans.get("confidence", 1.0))
            probs = [c if l == ans["choice"] else (1 - c) / (len(LABELS) - 1) for l in LABELS]
        out.append((ans["choice"], list(probs)))
    return out


def predict_claude(records, train, model="claude-sonnet-5", concurrency=8):
    """Teacher labels. Imported lazily so the harness runs without an API key."""
    from teacher import label_batch
    out = []
    for rec, dist in zip(records, label_batch(records, model=model, concurrency=concurrency)):
        probs = [float(dist.get(l, 0.0)) for l in LABELS]
        s = sum(probs) or 1.0
        probs = [p / s for p in probs]
        out.append((LABELS[probs.index(max(probs))], probs))
    return out


PREDICTORS = {"majority": predict_majority, "rules": predict_rules,
              "laya": predict_laya, "claude": predict_claude}


# --------------------------------------------------------------------------- reporting
def evaluate(name, records, preds, prior=None):
    y_true = [r["labels"]["intent"] for r in records]
    y_pred = [p[0] for p in preds]
    probs = [p[1] for p in preds]
    conf = [max(p) for p in probs]
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    idx = {l: i for i, l in enumerate(LABELS)}

    rep = {
        "name": name,
        "n": len(records),
        "accuracy": M.accuracy(y_true, y_pred),
        "macro_f1": M.macro_f1(y_true, y_pred, LABELS),
        "ece": M.ece(conf, correct),
        "brier": M.brier(probs, [idx[t] for t in y_true]),
        "per_class": M.per_class(y_true, y_pred, LABELS),
        "confusion": M.confusion(y_true, y_pred, LABELS),
        "risk_coverage": M.risk_coverage(conf, correct),
        "reliability": M.reliability_table(conf, correct),
    }
    hard = [i for i, r in enumerate(records) if r.get("hard")]
    if hard:
        rep["accuracy_hard"] = M.accuracy([y_true[i] for i in hard], [y_pred[i] for i in hard])
        rep["n_hard"] = len(hard)
    if prior:
        rep["accuracy_reweighted"] = M.reweighted_accuracy(y_true, y_pred, prior)
    return rep


def render(rep):
    L = ["", "=" * 78, "  %s   (n=%d)" % (rep["name"], rep["n"]), "=" * 78,
         "  accuracy        %.3f" % rep["accuracy"],
         "  macro F1        %.3f" % rep["macro_f1"],
         "  ECE             %.3f   (lower is better; refit temperature before trusting)" % rep["ece"],
         "  Brier           %.3f" % rep["brier"]]
    if "accuracy_hard" in rep:
        L.append("  accuracy (hard) %.3f   on the %d annotator-disagreement cases"
                 % (rep["accuracy_hard"], rep["n_hard"]))
    if rep.get("accuracy_reweighted") is not None:
        L.append("  accuracy (prod mix) %.3f" % rep["accuracy_reweighted"])
    L += ["", "  per class:", "  %-16s %6s %6s %6s %6s" % ("label", "prec", "rec", "f1", "n")]
    for l in LABELS:
        c = rep["per_class"][l]
        L.append("  %-16s %6.3f %6.3f %6.3f %6d" % (l, c["precision"], c["recall"], c["f1"], c["support"]))
    L += ["", "  confusion:", M.format_confusion(rep["confusion"], LABELS),
          "", "  risk-coverage (the cascade table):",
          "  %-10s %8s %10s %10s" % ("coverage", "n", "accuracy", "threshold")]
    for row in rep["risk_coverage"]:
        L.append("  %-10s %8d %10.3f %10.3f" % ("%.0f%%" % (row["coverage"] * 100), row["n"], row["accuracy"], row["threshold"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/interim/corpus.jsonl")
    ap.add_argument("--predictors", default="majority,rules")
    ap.add_argument("--split", default="eval", choices=["eval", "train", "all"])
    ap.add_argument("--prior", help="JSON file of true production class frequencies")
    ap.add_argument("--out", default="out/report.json")
    args = ap.parse_args()

    corpus = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    corpus = [r for r in corpus if r["labels"].get("intent")]
    # Fit-on-train is computed from the FULL corpus before the eval filter. Deriving it
    # from `recs` after filtering silently fits the majority baseline on the eval set,
    # which makes the floor look higher than it is.
    train = [r for r in corpus if r.get("split") == "train"]
    recs = corpus
    if args.split != "all":
        sel = [r for r in corpus if r.get("split") == args.split]
        if not sel:
            print("! no records in split=%s -- run src/split.py first; evaluating on ALL %d"
                  % (args.split, len(corpus)))
        else:
            recs = sel
    if not train:
        print("! no train split; fitting baselines on the evaluated records (optimistic)")
        train = recs
    prior = json.load(open(args.prior)) if args.prior else None

    reports = []
    for spec in args.predictors.split(","):
        spec = spec.strip()
        if not spec:
            continue
        name, _, arg = spec.partition(":")
        fn = PREDICTORS.get(name)
        if not fn:
            sys.exit("unknown predictor %r (have: %s)" % (name, ", ".join(PREDICTORS)))
        try:
            preds = fn(recs, train, arg) if arg else fn(recs, train)
        except ImportError as e:
            print("\n! skipping %-22s %s" % (spec, e))
            continue
        rep = evaluate(spec, recs, preds, prior)
        reports.append(rep)
        print(render(rep))

    if reports:
        print("\n" + "=" * 78 + "\n  SUMMARY\n" + "=" * 78)
        print("  %-26s %9s %9s %7s %9s" % ("predictor", "accuracy", "macro F1", "ECE", "acc@80%cov"))
        for r in reports:
            cov80 = next((x["accuracy"] for x in r["risk_coverage"] if abs(x["coverage"] - .8) < 1e-6), float("nan"))
            print("  %-26s %9.3f %9.3f %7.3f %9.3f" % (r["name"], r["accuracy"], r["macro_f1"], r["ece"], cov80))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(reports, open(args.out, "w"), indent=2)
        print("\n  full report -> %s" % args.out)


if __name__ == "__main__":
    main()
