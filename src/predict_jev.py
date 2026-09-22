"""Run TypeSafe Jev over a split and write predictions in the same format as predict.py.

Goes at the raw HTTP API with stdlib urllib rather than `typesafe-sdk`, for one practical
reason: the SDK needs Python 3.10+ and this machine has 3.9. The endpoint is small enough
that the SDK buys nothing here.

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    {"model": ..., "state": ..., "questions": {...}}
    -> {"answers": {qid: {"choice", "probabilities", "confidence"}}, "usage": {...}}

Questions come from the same task spec Laya was evaluated with, so the two models see
byte-identical instructions and criteria. That is the whole point -- the published
Jev-vs-Laya numbers compare different prompts on different samples, so they settle
nothing.

Responses are cached to disk by (model, state) hash. A re-run costs nothing and a crash
halfway through does not mean paying twice.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK  # noqa: E402

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def build_questions():
    """The task spec, in Jev's request shape (identical to Laya's by construction)."""
    q = TASK.QUESTIONS["intent"]
    return {"intent": {"type": "choice", "instructions": q["instructions"],
                       "criteria": dict(q["criteria"])}}


def call(state, questions, model, api_key, timeout=60, retries=5):
    body = json.dumps({"model": model, "state": state, "questions": questions}).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    })
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            last = "HTTP %s: %s" % (e.code, detail)
            if e.code in (401, 403):
                raise SystemExit("auth failed (%s). Check TYPESAFE_API_KEY." % last)
            if e.code == 400:
                raise SystemExit("bad request -- the payload shape is wrong:\n%s" % last)
            if e.code not in (408, 429, 500, 502, 503, 504):
                raise SystemExit(last)
        except Exception as e:                                   # noqa: BLE001
            last = repr(e)
        time.sleep(min(30, (2 ** attempt) + random.random()))     # backoff + jitter
    raise SystemExit("gave up after %d attempts: %s" % (retries, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="jev")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--cache", default="data/jev_cache")
    ap.add_argument("--limit", type=int, help="run only the first N rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact payload for row 1 and exit, without calling")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    if args.limit:
        recs = recs[:args.limit]
    questions = build_questions()

    if args.dry_run:
        print("POST %s\nAuthorization: Bearer $TYPESAFE_API_KEY\n" % ENDPOINT)
        print(json.dumps({"model": args.model, "state": TASK.state_for(recs[0]),
                          "questions": questions}, indent=2))
        print("\n%d rows would be sent. No request made." % len(recs))
        return

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        sys.exit("TYPESAFE_API_KEY is not set.\n"
                 "  export TYPESAFE_API_KEY=... then re-run.\n"
                 "  Use --dry-run to inspect the payload without a key.")

    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def one(rec):
        state = TASK.state_for(rec)
        key = hashlib.sha1(json.dumps([args.model, state, questions], sort_keys=True)
                           .encode("utf-8")).hexdigest()
        path = os.path.join(args.cache, key + ".json")
        if os.path.exists(path):
            return json.load(open(path)), True
        resp = call(state, questions, args.model, api_key)
        json.dump(resp, open(path, "w"))
        return resp, False

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one, recs))
    elapsed = time.time() - t0

    cached = sum(1 for _, c in results if c)
    tokens = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for rec, (resp, _) in zip(recs, results):
            ans = resp["answers"]["intent"]
            probs = ans.get("probabilities") or {}
            if isinstance(probs, list):
                probs = {l: float(v) for l, v in zip(TASK.LABELS, probs)}
            probs = {l: float(probs.get(l, 0.0)) for l in TASK.LABELS}
            total = sum(probs.values()) or 1.0
            probs = {l: v / total for l, v in probs.items()}
            tokens += (resp.get("usage") or {}).get("input_tokens", 0) or 0
            f.write(json.dumps({"id": rec["id"], "label": rec["label"],
                                "pred": ans["choice"], "probs": {l: round(v, 6) for l, v in probs.items()},
                                "jev_confidence": ans.get("confidence")}) + "\n")

    acc = sum(1 for rec, (resp, _) in zip(recs, results)
              if resp["answers"]["intent"]["choice"] == rec["label"]) / len(recs)
    print("  %s | %d rows (%d from cache)" % (args.tag, len(recs), cached))
    print("  accuracy %.4f | %.1fs wall | %.0f ms/row @ concurrency %d"
          % (acc, elapsed, 1000 * elapsed / max(1, len(recs) - cached or 1), args.concurrency))
    if tokens:
        print("  input tokens %d  ~= $%.4f at $0.042/1M" % (tokens, tokens * 0.042 / 1e6))
    print("  -> %s" % args.out)


if __name__ == "__main__":
    main()
