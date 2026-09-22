"""Run a checkpoint over a JSONL split and write predictions.jsonl. No tricks:
argmax of the model's own distribution, no thresholding, no post-processing."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK  # noqa: E402
from layakit import load_checkpoint, predict_probs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device")
    ap.add_argument("--fp32", action="store_true", help="disable autocast entirely")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    model, tok, cfg, dev = load_checkpoint(args.model_dir, args.device)
    print("  %s | %d rows | device=%s" % (args.tag, len(recs), dev))

    t0 = time.time()
    probs = predict_probs(model, tok, cfg, [TASK.state_for(r) for r in recs],
                          TASK.QUESTIONS["intent"], dev, args.batch_size,
                          use_amp=not args.fp32, progress=10)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, p in zip(recs, probs):
            f.write(json.dumps({
                "id": r["id"], "label": r["label"],
                "pred": TASK.LABELS[int(p.argmax())],
                "probs": {l: round(float(v), 6) for l, v in zip(TASK.LABELS, p)},
            }) + "\n")
    acc = sum(json.loads(l)["label"] == json.loads(l)["pred"] for l in open(args.out)) / len(recs)
    print("  accuracy %.4f | %.1fs total | %.1f ms/row -> %s"
          % (acc, elapsed, 1000 * elapsed / len(recs), args.out))


if __name__ == "__main__":
    main()
