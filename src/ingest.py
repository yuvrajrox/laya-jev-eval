"""CSV -> canonical JSONL corpus.

Designed so you can drop in an export from anywhere -- a Rox report, a warehouse
query, a Gmail export, the hand-written seed -- without editing code. Column names are
matched case-insensitively against the aliases below; anything unmatched is ignored.

    python3 src/ingest.py data/raw/*.csv --source rox_dev --out data/interim/corpus.jsonl

Labels are optional. Rows without an intent come out with labels={} and are the pile
the teacher pseudo-labels later.
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))
sys.path.insert(0, os.path.dirname(__file__))
import taxonomy  # noqa: E402
from clean import clean, detect_automated  # noqa: E402

# Map many plausible export column names onto the canonical field.
ALIASES = {
    "body":     ["body", "body_raw", "text", "message", "reply", "reply_body", "content",
                 "email_body", "plain_text", "text_body", "snippet"],
    "subject":  ["subject", "subject_line", "title", "email_subject"],
    "sender":   ["from", "sender", "from_email", "sender_email", "reply_from", "email"],
    "headers":  ["headers", "raw_headers", "header_blob"],
    "id":       ["id", "message_id", "email_id", "reply_id", "uuid"],
    "intent":   ["intent", "label", "category", "reply_intent", "classification", "y"],
    "meeting_booked": ["meeting_booked", "meeting", "booked"],
    "is_human": ["is_human", "human"],
    "hostility": ["hostility", "tone"],
    "revisit_later": ["revisit_later", "revisit"],
    "lang":     ["lang", "language", "locale"],
    "account":  ["account", "account_name", "company", "company_name"],
    "user_action": ["user_action", "action", "batch_action"],
    "contact_state": ["contact_state", "state", "status", "lifecycle_state"],
    "hard":     ["hard"],
}
CANON = {alias: field for field, aliases in ALIASES.items() for alias in aliases}

TRUTHY = {"1", "true", "t", "yes", "y"}


def _bool(v):
    if v is None or v == "":
        return None
    return str(v).strip().lower() in TRUTHY


def _remap(row):
    """Raw CSV row -> dict keyed by canonical field names."""
    out = {}
    for k, v in row.items():
        if k is None:
            continue
        field = CANON.get(k.strip().lower())
        if field and out.get(field) in (None, ""):
            out[field] = v
    return out


def _labels_from(r, source):
    """Assemble labels and record where they came from.

    Precedence, strongest first:
      1. an explicit intent column          -> human / synthetic
      2. a Rox user action (Mark Interested) -> user_action, gold
      3. header + subject rules for ooo/bounce -> rule
    A contact-state transition alone is deliberately NOT used as an intent label; the
    product brief flags Paused as only usually meaning out-of-office, and a wrong
    label is worse than no label because it trains the error in.
    """
    labels, quality = {}, None
    intent = (r.get("intent") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if intent in taxonomy.INTENTS:
        labels["intent"] = intent
        quality = "synthetic" if source == "synthetic" else "human"
    elif intent:
        raise ValueError("unknown intent %r (expected one of %s)" % (intent, taxonomy.INTENTS))

    action = (r.get("user_action") or "").strip().lower()
    if not labels and action in taxonomy.USER_ACTION_LABELS:
        mapped, q = taxonomy.USER_ACTION_LABELS[action]
        if q == "gold" and mapped:
            labels.update(mapped)
            quality = "user_action"

    if not labels.get("intent"):
        rule = detect_automated(subject=r.get("subject", ""), sender=r.get("sender", ""),
                                headers=r.get("headers", ""))
        if rule:
            labels["intent"] = rule
            labels.setdefault("is_human", False)
            quality = quality or "rule"

    for key in ("meeting_booked", "is_human", "revisit_later"):
        b = _bool(r.get(key))
        if b is not None:
            labels[key] = b
    if r.get("hostility") not in (None, ""):
        labels["hostility"] = int(float(r["hostility"]))
    return labels, quality


def ingest_rows(rows, source, id_prefix=None):
    seen, out = set(), []
    for i, raw in enumerate(rows):
        r = _remap(raw)
        body_raw = (r.get("body") or "").strip()
        if not body_raw:
            continue
        body, thread = clean(body_raw)
        if not body:
            continue
        rid = r.get("id") or "%s-%s" % (id_prefix or source,
                                        hashlib.sha1(body_raw.encode("utf-8")).hexdigest()[:12])
        if rid in seen:          # exact-duplicate bodies are common in email exports
            continue
        seen.add(rid)
        labels, quality = _labels_from(r, source)
        out.append({
            "id": rid,
            "source": source,
            "subject": (r.get("subject") or "").strip(),
            "body": body,
            "body_raw": body_raw,
            "thread": thread,
            "account": {"name": r.get("account")} if r.get("account") else {},
            "lang": (r.get("lang") or "").strip() or None,
            "labels": labels,
            "label_source": quality,
            "hard": bool(_bool(r.get("hard"))),
            "split": None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV files or globs")
    ap.add_argument("--source", required=True, help="rox_prod | rox_dev | enron | spamassassin | synthetic")
    ap.add_argument("--out", default="data/interim/corpus.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    paths = [p for pattern in args.inputs for p in sorted(glob.glob(pattern))]
    if not paths:
        sys.exit("no input files matched")

    records = []
    for path in paths:
        with open(path, newline="", encoding="utf-8-sig") as f:
            records += ingest_rows(list(csv.DictReader(f)), args.source,
                                   id_prefix=os.path.splitext(os.path.basename(path))[0])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    labelled = sum(1 for r in records if r["labels"].get("intent"))
    print("%d records -> %s  (%d with an intent label, %d unlabelled)"
          % (len(records), args.out, labelled, len(records) - labelled))
    by_src = {}
    for r in records:
        by_src[r["label_source"]] = by_src.get(r["label_source"], 0) + 1
    for k, v in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print("  label_source=%-12s %d" % (k, v))


if __name__ == "__main__":
    main()
