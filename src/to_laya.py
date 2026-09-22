"""Canonical corpus -> Laya fine-tuning JSONL.

Target format, derived from `encode_record` in the model repo's rl_common.py:

    {"state": {...}, "src": "...", "qs": [
        {"t": "choice", "ins": "...", "crit": {...}, "y": 3, "soft": [...]},
        {"t": "noul",   "ins": "...", "y": 1},
    ]}

Notes that matter for training quality:

  `soft` carries the teacher's full distribution. RLCD's reward is a strictly proper
  scoring rule, so soft targets transfer calibration, not just the argmax. Always pass
  it when the teacher produced one -- it is the difference between a model that is
  right 76% of the time and one that also knows which 24% it got wrong.

  Option order is shuffled during training by the trainer itself (rl_common.py shuffles
  non-score questions per epoch), so do not pre-shuffle here.

  Human labels are repeated `--gold-weight` times. A user clicking "Mark Interested" on
  a reply the system mislabelled is a corrected error, which is worth far more than
  another easy teacher-labelled out-of-office. Oversampling is the crude way to say so,
  and it works.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))
import taxonomy  # noqa: E402

GOLD = {"human", "user_action"}


def to_question_records(rec, questions):
    qs = []
    for qid in questions:
        spec = taxonomy.QUESTIONS[qid]
        label = rec["labels"].get(qid)
        if label is None:
            continue
        q = {"t": spec["type"], "ins": spec["instructions"]}
        if spec["type"] == "choice":
            q["crit"] = spec["criteria"]
            if label not in taxonomy.INTENTS:
                continue
            q["y"] = taxonomy.INTENTS.index(label)
        elif spec["type"] == "score":
            q["crit"] = spec["criteria"]
            q["y"] = int(label)
        else:                                     # noul: options are always [false, true]
            q["y"] = 1 if label else 0
        soft = (rec.get("soft") or {}).get(qid)
        if soft and len(soft) == (len(q.get("crit") or []) or 2):
            q["soft"] = list(soft)
        qs.append(q)
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/interim/corpus.jsonl")
    ap.add_argument("--out-dir", default="data/train")
    ap.add_argument("--questions", default=",".join(taxonomy.LABELLED))
    ap.add_argument("--gold-weight", type=int, default=3,
                    help="times to repeat a human/user-action labelled record")
    ap.add_argument("--min-teacher-conf", type=float, default=0.75)
    args = ap.parse_args()

    questions = [q.strip() for q in args.questions.split(",") if q.strip()]
    unknown = [q for q in questions if q not in taxonomy.QUESTIONS]
    if unknown:
        sys.exit("unknown questions: %s" % unknown)

    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    os.makedirs(args.out_dir, exist_ok=True)
    counts, written = Counter(), Counter()

    for split in ("train", "eval"):
        path = os.path.join(args.out_dir, "%s.jsonl" % split)
        with open(path, "w", encoding="utf-8") as f:
            for rec in recs:
                if (rec.get("split") or "train") != split:
                    continue
                if rec["label_source"] == "teacher" and \
                        rec.get("teacher_conf", 1.0) < args.min_teacher_conf:
                    continue                      # low-confidence teacher output is poison
                qs = to_question_records(rec, questions)
                if not qs:
                    continue
                out = {"state": taxonomy.state_for(rec), "src": rec["source"], "qs": qs}
                # The eval split is never oversampled -- weighting it would distort metrics.
                reps = args.gold_weight if (split == "train" and rec["label_source"] in GOLD) else 1
                for _ in range(reps):
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    written[split] += 1
                counts[(split, rec["label_source"])] += 1
        print("%-6s %6d records (%d rows after gold oversampling x%d) -> %s"
              % (split, sum(v for (s, _), v in counts.items() if s == split),
                 written[split], args.gold_weight, path))

    print("\nby label source:")
    for (split, src), n in sorted(counts.items()):
        print("  %-6s %-18s %5d" % (split, src, n))
    print("\nNext: run the fine-tune notebook from the model repo against data/train/train.jsonl")
    print("      starting from `convaiinnovations/laya-typed-decisions`, NOT the base checkpoint.")


if __name__ == "__main__":
    main()
