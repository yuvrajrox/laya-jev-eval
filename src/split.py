"""Freeze the eval split. Run this ONCE, then never again on the same corpus.

The eval set is the only asset in this project whose value depends on nobody having
looked at it. Two rules it enforces:

  Stratify by intent. A random split leaves ~4 unsubscribes in eval, so unsubscribe
  recall becomes unmeasurable -- and unsubscribe recall is the one number with legal
  consequences attached.

  Split by thread, not by message. Two replies from the same thread share the sender,
  the pitch and often the phrasing. Letting one land in train and one in eval leaks,
  and inflates the eval score by several points.

Splits are assigned by hashing the record id, so re-running after adding new data
keeps every existing assignment stable.
"""
import argparse
import hashlib
import json
from collections import defaultdict


def bucket(key, salt):
    h = hashlib.sha1(("%s|%s" % (salt, key)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/interim/corpus.jsonl")
    ap.add_argument("--out", default="data/interim/corpus.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.30)
    ap.add_argument("--salt", default="rox-reply-intent-v1",
                    help="changing this reshuffles everything; do not change it casually")
    ap.add_argument("--group-key", default="thread_id",
                    help="field holding the thread/conversation id, if your export has one")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    by_intent = defaultdict(list)
    for r in recs:
        by_intent[r["labels"].get("intent") or "_unlabelled"].append(r)

    counts = defaultdict(lambda: defaultdict(int))
    for intent, group in by_intent.items():
        for r in group:
            key = r.get(args.group_key) or r["id"]
            r["split"] = "eval" if bucket(key, args.salt + intent) < args.eval_frac else "train"
            counts[intent][r["split"]] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("%-18s %6s %6s" % ("intent", "eval", "train"))
    for intent in sorted(counts):
        print("%-18s %6d %6d" % (intent, counts[intent]["eval"], counts[intent]["train"]))
    tot_e = sum(c["eval"] for c in counts.values())
    print("%-18s %6d %6d" % ("TOTAL", tot_e, len(recs) - tot_e))
    thin = [i for i, c in counts.items() if c["eval"] < 30 and i != "_unlabelled"]
    if thin:
        print("\n! thin eval classes (<30): %s" % ", ".join(thin))
        print("  per-class recall on these is noise. Source more before trusting them.")


if __name__ == "__main__":
    main()
