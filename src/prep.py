"""CSV -> canonical JSONL + a frozen 70/30 split, with every artefact written to disk.

The split is GROUPED BY NORMALISED TEXT. The supplied file has 790 rows that are exact
duplicates of another row, so a plain random split puts the same sentence in train and
test. The model then scores well by memorising a string it has literally already seen,
and the fine-tuned number is inflated for a reason that has nothing to do with learning.

A naive random split is also written, so the size of that inflation can be measured
rather than argued about.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK  # noqa: E402


def norm(t):
    return " ".join((t or "").lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    data = os.path.join(args.run_dir, "data")
    os.makedirs(data, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8-sig")))
    recs = []
    for i, r in enumerate(rows):
        body = (r.get("email") or "").strip()
        label = (r.get("intent") or "").strip().lower()
        if not body or label not in TASK.LABELS:
            continue
        recs.append({
            "id": "ei-%05d" % i,
            "body": body,
            "norm": norm(body),
            "label": label,
            "group": hashlib.sha1(norm(body).encode("utf-8")).hexdigest()[:16],
        })

    with open(os.path.join(data, "01_corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- grouped split: every copy of a text lands on the same side ----------
    rng = random.Random(args.seed)
    by_group = defaultdict(list)
    for r in recs:
        by_group[r["group"]].append(r)
    # Stratify by the group's label so class balance survives the grouping.
    groups_by_label = defaultdict(list)
    for g, members in by_group.items():
        groups_by_label[members[0]["label"]].append(g)
    train_groups = set()
    for label, gs in groups_by_label.items():
        gs = sorted(gs)
        rng.shuffle(gs)
        train_groups.update(gs[:int(round(len(gs) * args.train_frac))])
    for r in recs:
        r["split"] = "train" if r["group"] in train_groups else "test"

    # ---- naive split, for measuring the leak, not for training --------------
    naive = list(recs)
    rng2 = random.Random(args.seed)
    rng2.shuffle(naive)
    cut = int(round(len(naive) * args.train_frac))
    naive_train = {r["id"] for r in naive[:cut]}
    for r in recs:
        r["split_naive"] = "train" if r["id"] in naive_train else "test"

    for split in ("train", "test"):
        for key, tag in (("split", "grouped"), ("split_naive", "naive")):
            sel = [r for r in recs if r[key] == split]
            path = os.path.join(data, "02_%s_%s.jsonl" % (tag, split))
            with open(path, "w", encoding="utf-8") as f:
                for r in sel:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(data, "01_corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- report -------------------------------------------------------------
    tr = [r for r in recs if r["split"] == "train"]
    te = [r for r in recs if r["split"] == "test"]
    tr_norm = {r["norm"] for r in tr}
    leak_grouped = sum(1 for r in te if r["norm"] in tr_norm)
    ntr = {r["norm"] for r in recs if r["split_naive"] == "train"}
    nte = [r for r in recs if r["split_naive"] == "test"]
    leak_naive = sum(1 for r in nte if r["norm"] in ntr)

    summary = {
        "rows": len(recs), "unique_texts": len(by_group),
        "duplicate_rows": len(recs) - len(by_group),
        "grouped": {"train": len(tr), "test": len(te),
                    "test_rows_also_seen_in_train": leak_grouped,
                    "leak_pct": round(100.0 * leak_grouped / max(1, len(te)), 1)},
        "naive": {"train": cut, "test": len(nte),
                  "test_rows_also_seen_in_train": leak_naive,
                  "leak_pct": round(100.0 * leak_naive / max(1, len(nte)), 1)},
        "train_label_counts": dict(Counter(r["label"] for r in tr)),
        "test_label_counts": dict(Counter(r["label"] for r in te)),
        "seed": args.seed,
    }
    json.dump(summary, open(os.path.join(data, "03_split_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\nartefacts -> %s" % data)


if __name__ == "__main__":
    main()
