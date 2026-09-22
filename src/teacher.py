"""Claude as the teacher: pseudo-label the unlabelled pile, with soft targets.

This is Apollo's step 2-3 (seed labels -> fine-tuned teacher -> pseudo-label -> train
the small fast model on high-confidence predictions only), and it is what makes the
project affordable: you hand-label ~300 for eval and the teacher labels the other
10,000 for the price of a batch job.

Two things this does that a naive labelling script does not:

  It asks for a DISTRIBUTION, not a label. RLCD trains against strictly proper scoring
  rules, so a soft target transfers the teacher's uncertainty as well as its answer.
  A hard label throws that away, and calibration is the whole reason to pick Laya.

  It filters by confidence before training. Teacher mistakes are not random -- they
  cluster on exactly the ambiguous cases -- so training on low-confidence teacher
  output bakes in a systematic error. Keep p_max >= --min-conf and route the rest to
  a human. Those routed cases are your next annotation batch and your highest-value
  training data.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))
import taxonomy  # noqa: E402

SYSTEM = """You label replies to cold outbound sales emails for a training corpus.

Return ONLY a JSON object mapping every label below to a probability, summing to 1.0.
No prose, no code fence. Express genuine uncertainty: if a reply is ambiguous between
two labels, split the mass. Do not round to 0 and 1 -- the distribution is the signal.

Labels:
%s

Rules that annotators get wrong and you must not:
- "Not right now, try us in Q3" is not_interested, even though it is polite and open.
- An out-of-office that also says "yes, let's talk when I'm back" is interested; the
  intent of the human author wins over the automated wrapper.
- An autoresponder saying the person has left the company is referral if it names a
  successor or a team, and bounce if the mailbox itself is dead.
- "Thanks" alone is other, not interested.
- A reply that both declines and asks for removal is unsubscribe -- the removal request
  is the one with legal consequences, so it takes precedence.
- Asking "how did you get my email?" is other unless removal is also requested.
""" % "\n".join("- %s: %s" % (k, v) for k, v in taxonomy.INTENT_CRITERIA.items())


def _prompt(rec):
    parts = ["Subject: %s" % (rec.get("subject") or "(none)")]
    if rec.get("account", {}).get("name"):
        parts.append("Prospect company: %s" % rec["account"]["name"])
    thread = rec.get("thread") or []
    if thread:
        parts.append("Earlier in the thread (oldest first):\n%s" % "\n---\n".join(thread[-2:])[:1500])
    parts.append("Reply to classify:\n%s" % rec["body"][:3000])
    return "\n\n".join(parts)


def label_batch(records, model="claude-sonnet-5", concurrency=8, max_tokens=300):
    import anthropic
    client = anthropic.Anthropic()

    def one(rec):
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model=model, max_tokens=max_tokens, system=SYSTEM,
                    messages=[{"role": "user", "content": _prompt(rec)}])
                text = msg.content[0].text.strip().strip("`")
                text = text[text.index("{"):text.rindex("}") + 1]
                dist = json.loads(text)
                dist = {k: float(v) for k, v in dist.items() if k in taxonomy.INTENTS}
                total = sum(dist.values())
                if total <= 0:
                    raise ValueError("empty distribution")
                return {k: v / total for k, v in dist.items()}
            except Exception as e:                       # noqa: BLE001
                if attempt == 2:
                    print("! teacher failed on %s: %s" % (rec["id"], e), file=sys.stderr)
                    return {l: 1.0 / len(taxonomy.INTENTS) for l in taxonomy.INTENTS}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(one, records))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/interim/corpus.jsonl")
    ap.add_argument("--out", default="data/interim/corpus_teacher.jsonl")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--min-conf", type=float, default=0.75,
                    help="records below this are left unlabelled and routed to a human")
    ap.add_argument("--only-unlabelled", action="store_true", default=True)
    ap.add_argument("--relabel-all", dest="only_unlabelled", action="store_false",
                    help="also label records that already have a label, to measure teacher-vs-human agreement")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    todo = [r for r in recs if not (args.only_unlabelled and r["labels"].get("intent"))]
    # Never let the teacher overwrite a human label on the frozen eval split.
    todo = [r for r in todo if r.get("split") != "eval" or not r["labels"].get("intent")]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        sys.exit("nothing to label (use --relabel-all to measure teacher/human agreement)")
    print("labelling %d records with %s ..." % (len(todo), args.model))

    dists = label_batch(todo, model=args.model, concurrency=args.concurrency)
    kept = 0
    for rec, dist in zip(todo, dists):
        top = max(dist, key=dist.get)
        rec["soft"] = {"intent": [dist.get(l, 0.0) for l in taxonomy.INTENTS]}
        rec["teacher_conf"] = dist[top]
        if dist[top] >= args.min_conf:
            rec["labels"]["intent"] = top
            rec["label_source"] = "teacher"
            kept += 1
        else:
            rec["label_source"] = "teacher_low_conf"

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("kept %d/%d above conf>=%.2f -> %s" % (kept, len(todo), args.min_conf, args.out))
    print("the %d rejected are your next human annotation batch -- label those first"
          % (len(todo) - kept))


if __name__ == "__main__":
    main()
