"""Detailed comparison on the hand-written holdout, including per-error listing."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK, metrics as M
from report import mcnemar, score

run = sys.argv[1]
data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(run, "data/04_holdout100.jsonl"))}
zs = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(run, "reports/41_zeroshot_holdout.jsonl")))}
ft = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(run, "reports/42_finetuned_holdout.jsonl")))}
L = TASK.LABELS

for name, preds in (("ZERO-SHOT", zs), ("FINE-TUNED", ft)):
    s = score(preds)
    print("\n" + "=" * 72); print("  %s  -- hand-written holdout (n=100)" % name); print("=" * 72)
    print("  accuracy %.3f | macro F1 %.3f | ECE %.3f | mean conf %.3f"
          % (s["accuracy"], s["macro_f1"], s["ece"], s["mean_conf"]))
    print("\n  %-12s %6s %6s %6s %4s" % ("label", "prec", "rec", "f1", "n"))
    for l in L:
        c = s["per_class"][l]; print("  %-12s %6.3f %6.3f %6.3f %4d" % (l, c["precision"], c["recall"], c["f1"], c["support"]))
    print("\n" + M.format_confusion(s["confusion"], L))
    hard = [i for i in data if data[i]["hard"]]; easy = [i for i in data if not data[i]["hard"]]
    ah = sum(preds[i]["pred"] == preds[i]["label"] for i in hard) / len(hard)
    ae = sum(preds[i]["pred"] == preds[i]["label"] for i in easy) / len(easy)
    print("\n  hard cases (%d): %.3f     easy cases (%d): %.3f" % (len(hard), ah, len(easy), ae))

mc = mcnemar(zs, ft)
print("\n" + "=" * 72)
print("  McNemar: fine-tuning fixed %d, broke %d  ->  p = %.4g"
      % (mc["b_fixed_a_broke"], mc["a_had_b_broke"], mc["p_value"]))
print("=" * 72)

print("\n  ERRORS THE FINE-TUNED MODEL STILL MAKES (%d):\n" % sum(1 for i in ft if ft[i]["pred"] != ft[i]["label"]))
for i in sorted(ft):
    if ft[i]["pred"] != ft[i]["label"]:
        d = data[i]
        print("  true=%-10s pred=%-10s conf=%.2f %s" % (d["label"], ft[i]["pred"], max(ft[i]["probs"].values()), "[HARD]" if d["hard"] else ""))
        print("      %r\n" % d["body"].replace("\n", " ")[:110])
