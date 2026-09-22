"""Three-way comparison on the hand-written holdout: Jev vs Laya zero-shot vs Laya fine-tuned.

Fairness note that belongs next to every number here: the labels follow the convention
of the supplied dataset. Laya fine-tuned was TRAINED on that convention; Jev and Laya
zero-shot have only the four criteria strings to go on. So the honest reading is
Jev vs Laya-zero-shot as a like-for-like comparison, and Jev vs Laya-fine-tuned as a
"bought vs built" comparison where the built one had the advantage of seeing your data.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK, metrics as M
from report import mcnemar, score

run = sys.argv[1]
data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(run, "data/04_holdout100.jsonl"))}
def load(f): return {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(run, "reports", f)))}
runs = [("Laya zero-shot", load("41_zeroshot_holdout.jsonl")),
        ("Laya fine-tuned", load("42_finetuned_holdout.jsonl")),
        ("Jev 1.13.0", load("50_jev_holdout.jsonl"))]

print("=" * 78); print("  HAND-WRITTEN HOLDOUT (n=100) -- three-way"); print("=" * 78)
print("  %-18s %8s %9s %7s %8s %11s" % ("model", "accuracy", "macro F1", "ECE", "Brier", "mean conf"))
scores = {}
for name, p in runs:
    s = score(p); scores[name] = s
    print("  %-18s %8.3f %9.3f %7.3f %8.3f %11.3f"
          % (name, s["accuracy"], s["macro_f1"], s["ece"], s["brier"], s["mean_conf"]))

print("\n  per-class F1:")
print("  %-18s %10s %10s %11s %10s" % ("model", "request", "inquiry", "complaint", "feedback"))
for name, _ in runs:
    s = scores[name]
    print("  %-18s %10.3f %10.3f %11.3f %10.3f" % (name, *[s["per_class"][l]["f1"] for l in TASK.LABELS]))

print("\n  hard vs easy (13 hard / 87 easy):")
hard = [i for i in data if data[i]["hard"]]; easy = [i for i in data if not data[i]["hard"]]
for name, p in runs:
    ah = sum(p[i]["pred"] == p[i]["label"] for i in hard) / len(hard)
    ae = sum(p[i]["pred"] == p[i]["label"] for i in easy) / len(easy)
    print("  %-18s hard %.3f   easy %.3f" % (name, ah, ae))

print("\n  CAN YOU ROUTE ON ITS CONFIDENCE? (the cascade question)")
print("  %-18s %-22s %-18s %s" % ("model", "conf when WRONG", "conf when right", "rows >=0.99"))
for name, p in runs:
    rows = list(p.values())
    w = sorted(max(r["probs"].values()) for r in rows if r["pred"] != r["label"])
    c = sorted(max(r["probs"].values()) for r in rows if r["pred"] == r["label"])
    over = sum(1 for r in rows if max(r["probs"].values()) >= 0.99)
    wr = "%.2f-%.2f" % (w[0], w[-1]) if w else "(none wrong)"
    print("  %-18s %-22s %.2f-%.2f (med %.2f) %8d/100" % (name, wr, c[0], c[-1], c[len(c)//2], over))

print("\n  PAIRED TESTS (McNemar exact):")
for a, b in [("Laya zero-shot", "Jev 1.13.0"), ("Laya fine-tuned", "Jev 1.13.0"),
             ("Laya zero-shot", "Laya fine-tuned")]:
    pa = dict(runs)[a]; pb = dict(runs)[b]
    mc = mcnemar(pa, pb)
    verdict = "significant" if mc["p_value"] < 0.05 else "NOT significant"
    print("  %-17s vs %-17s  %s won %2d, %s won %2d  -> p=%.4g (%s)"
          % (a, b, b.split()[0], mc["b_fixed_a_broke"], a.split()[0], mc["a_had_b_broke"], mc["p_value"], verdict))

print("\n  WHERE THEY DISAGREE WITH THE LABEL:")
allp = dict(runs)
for i in sorted(data):
    wrong = [n for n, p in runs if p[i]["pred"] != p[i]["label"]]
    if not wrong: continue
    d = data[i]
    print("\n  true=%-10s %s" % (d["label"], "[HARD]" if d["hard"] else ""))
    print("    %r" % d["body"].replace("\n", " ")[:104])
    for n, p in runs:
        mark = "X" if p[i]["pred"] != p[i]["label"] else "."
        print("      %s %-17s -> %-10s (%.2f)" % (mark, n, p[i]["pred"], max(p[i]["probs"].values())))
