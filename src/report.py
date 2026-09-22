"""Compare prediction files and emit the final report (markdown + json).

Includes McNemar's exact test on the paired predictions. With 585 test rows, a swing of
a few examples moves accuracy by half a point, so "the fine-tuned model is better" is
not a claim you can make from the headline number alone. McNemar looks only at the rows
where the two models DISAGREE, which is the right question: of the cases they answered
differently, did the new model win significantly more often than it lost?
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK  # noqa: E402
import metrics as M  # noqa: E402

LABELS = TASK.LABELS


def load(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8"))}


def score(preds):
    rows = list(preds.values())
    y = [r["label"] for r in rows]
    p = [r["pred"] for r in rows]
    probs = [[r["probs"][l] for l in LABELS] for r in rows]
    conf = [max(v) for v in probs]
    ok = [int(a == b) for a, b in zip(y, p)]
    idx = {l: i for i, l in enumerate(LABELS)}
    return {
        "n": len(rows),
        "accuracy": M.accuracy(y, p),
        "macro_f1": M.macro_f1(y, p, LABELS),
        "ece": M.ece(conf, ok),
        "brier": M.brier(probs, [idx[t] for t in y]),
        "per_class": M.per_class(y, p, LABELS),
        "confusion": M.confusion(y, p, LABELS),
        "risk_coverage": M.risk_coverage(conf, ok),
        "mean_conf": sum(conf) / len(conf),
    }


def binom_two_sided(b, c):
    """Exact McNemar p-value: P(|X - n/2| >= |b - n/2|) for X ~ Binom(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a, b):
    """a, b: prediction dicts. Returns wins for b, wins for a, and the p-value."""
    ids = sorted(set(a) & set(b))
    b_wins = a_wins = 0
    for i in ids:
        ra, rb = a[i], b[i]
        oa, ob = ra["pred"] == ra["label"], rb["pred"] == rb["label"]
        if ob and not oa:
            b_wins += 1
        elif oa and not ob:
            a_wins += 1
    return {"n_compared": len(ids), "b_fixed_a_broke": b_wins, "a_had_b_broke": a_wins,
            "p_value": binom_two_sided(b_wins, a_wins)}


def section(title, s):
    L = ["### %s" % title, "",
         "| metric | value |", "|---|---|",
         "| rows | %d |" % s["n"],
         "| **accuracy** | **%.4f** |" % s["accuracy"],
         "| macro F1 | %.4f |" % s["macro_f1"],
         "| ECE (calibration error, lower better) | %.4f |" % s["ece"],
         "| Brier | %.4f |" % s["brier"],
         "| mean confidence | %.4f |" % s["mean_conf"], "",
         "| label | precision | recall | F1 | n |", "|---|---|---|---|---|"]
    for l in LABELS:
        c = s["per_class"][l]
        L.append("| %s | %.3f | %.3f | %.3f | %d |" % (l, c["precision"], c["recall"], c["f1"], c["support"]))
    L += ["", "```", "confusion (rows = true label)", M.format_confusion(s["confusion"], LABELS), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--zeroshot-all", required=True)
    ap.add_argument("--zeroshot-test", required=True)
    ap.add_argument("--finetuned-test", required=True)
    ap.add_argument("--naive-finetuned-test")
    ap.add_argument("--out", default="reports/90_REPORT.md")
    args = ap.parse_args()

    zs_all, zs_te, ft_te = load(args.zeroshot_all), load(args.zeroshot_test), load(args.finetuned_test)
    s_all, s_zs, s_ft = score(zs_all), score(zs_te), score(ft_te)
    mc = mcnemar(zs_te, ft_te)

    split = json.load(open(os.path.join(args.run_dir, "data", "03_split_summary.json")))
    meta_path = os.path.join(args.run_dir, "checkpoints", "train_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    delta = s_ft["accuracy"] - s_zs["accuracy"]
    L = ["# Laya on `email_intent.csv` — zero-shot vs fine-tuned", "",
         "Task: 4-way customer email intent (`request` / `inquiry` / `complaint` / `feedback`).",
         "Base checkpoint: `laya-typed-decisions`. Split: 70/30, grouped by text.", "",
         "## Headline", "",
         "| run | data | accuracy | macro F1 | ECE |", "|---|---|---|---|---|",
         "| zero-shot (no training) | all %d rows | %.4f | %.4f | %.4f |" % (s_all["n"], s_all["accuracy"], s_all["macro_f1"], s_all["ece"]),
         "| zero-shot (no training) | test %d | %.4f | %.4f | %.4f |" % (s_zs["n"], s_zs["accuracy"], s_zs["macro_f1"], s_zs["ece"]),
         "| **fine-tuned on 70%%** | test %d | **%.4f** | %.4f | %.4f |" % (s_ft["n"], s_ft["accuracy"], s_ft["macro_f1"], s_ft["ece"]),
         "",
         "**Change from fine-tuning: %+.4f accuracy (%+.2f points), %+.4f macro F1, %+.4f ECE.**"
         % (delta, delta * 100, s_ft["macro_f1"] - s_zs["macro_f1"], s_ft["ece"] - s_zs["ece"]),
         "",
         "### Is the difference real? (McNemar exact test, paired)", "",
         "Of the %d test rows, fine-tuning **fixed %d** that zero-shot got wrong and "
         "**broke %d** that zero-shot got right." % (mc["n_compared"], mc["b_fixed_a_broke"], mc["a_had_b_broke"]),
         "", "**p = %.4g** — %s at the 0.05 level." % (
             mc["p_value"],
             "the difference is statistically significant" if mc["p_value"] < 0.05
             else "NOT statistically significant, i.e. this size of change is what you would expect from chance alone"),
         "",
         "## Data and split", "",
         "| | value |", "|---|---|",
         "| total rows | %d |" % split["rows"],
         "| unique texts | %d |" % split["unique_texts"],
         "| rows that duplicate another row | %d |" % split["duplicate_rows"],
         "| grouped split | %d train / %d test |" % (split["grouped"]["train"], split["grouped"]["test"]),
         "| test rows also present in train (grouped) | %d (%.1f%%) |" % (split["grouped"]["test_rows_also_seen_in_train"], split["grouped"]["leak_pct"]),
         "| test rows also present in train (naive random) | %d (%.1f%%) |" % (split["naive"]["test_rows_also_seen_in_train"], split["naive"]["leak_pct"]),
         "",
         "%d of the %d rows are exact duplicates of another row. A plain random 70/30 split "
         "puts **%.1f%% of the test set verbatim into training**, so the model scores by "
         "recalling a string it has already seen. The split used here groups every copy of a "
         "text onto one side, giving 0%% overlap." % (split["duplicate_rows"], split["rows"], split["naive"]["leak_pct"]),
         ""]

    if meta:
        L += ["## Training", "",
              "| | value |", "|---|---|",
              "| epochs run | %d |" % meta["args"]["epochs"],
              "| best epoch (by held-out val) | %d |" % meta["best_epoch"],
              "| val accuracy, zero-shot | %.4f |" % meta["zero_shot_val_acc"],
              "| val accuracy, best | %.4f |" % meta["best_val_acc"],
              "| train / val rows | %d / %d |" % (meta["train_n"], meta["val_n"]),
              "| learning rate (encoder / head) | %g / %g |" % (meta["args"]["lr"], meta["args"]["head_lr"]),
              "| wall clock | %.1f min |" % meta["minutes"], "",
              "Epoch selection used a validation slice carved out of **train**, grouped the same "
              "way. The test split was never used for selection.", ""]

    nd = os.path.join(args.run_dir, "reports", "30_near_duplicate_analysis.txt")
    if os.path.exists(nd):
        L += ["## How much is the 100% worth? (near-duplicate analysis)", "",
              "The grouped split removes rows that are *exactly* the same text. It does nothing "
              "about rows generated from the same template. So before reading anything into a "
              "perfect score, here is how close each test row is to its nearest training row, "
              "by token overlap:", "", "```", open(nd, encoding="utf-8").read().strip(), "```", "",
              "**This dataset is template-generated.** 39.3% of test rows are >=0.8 similar to a "
              "training row, and the closest pairs differ by a single substituted word "
              "(`third time` -> `fourth time`, `flight` -> `laptop`). No test row is below 0.4 "
              "similarity to something in training.", "",
              "So the 1.0000 is real *for this corpus* and close to meaningless as a prediction of "
              "real-world accuracy. What it demonstrates is that the fine-tuning pipeline works end "
              "to end and that the model absorbs a labelling scheme from ~1,300 examples. What it "
              "does **not** demonstrate is that Laya will hit 100% on genuine customer email, which "
              "is messy, longer, and not drawn from 20 sentence patterns.", ""]

    L += ["## Detail", "", section("Zero-shot — test split", s_zs), section("Fine-tuned — test split", s_ft),
          section("Zero-shot — all 2000 rows", s_all)]

    L += ["## Confidence-based routing (test split)", "",
          "Accuracy on the most-confident X%% of rows. This is the cascade table: serve the model "
          "where it is confident, escalate the rest.", "",
          "| coverage | zero-shot acc | fine-tuned acc |", "|---|---|---|"]
    for a, b in zip(s_zs["risk_coverage"], s_ft["risk_coverage"]):
        L.append("| %.0f%% | %.4f | %.4f |" % (a["coverage"] * 100, a["accuracy"], b["accuracy"]))
    L.append("")

    out = os.path.join(args.run_dir, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(L))
    json.dump({"zeroshot_all": s_all, "zeroshot_test": s_zs, "finetuned_test": s_ft,
               "mcnemar": mc, "split": split, "train_meta": meta},
              open(out.replace(".md", ".json"), "w"), indent=2)
    print("\n".join(L[:32]))
    print("\n-> %s" % out)


if __name__ == "__main__":
    main()
