#!/usr/bin/env bash
# End-to-end smoke test. Runs on hand-written + public data only -- no Rox data needed.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
VPY=${VPY:-.venv/bin/python}

echo "== 1. seed replies"
$PY data/seed/seed_replies.py

echo; echo "== 2. public autoreplies + bounces (skipped if offline)"
$PY src/public_data.py --limit 150 || echo "  skipped"

echo; echo "== 3. ingest"
$PY src/ingest.py data/seed/seed_replies.csv --source synthetic --out data/interim/corpus.jsonl
[ -f data/public/spamassassin_auto.csv ] && \
  $PY src/ingest.py data/public/spamassassin_auto.csv --source spamassassin \
      --out data/interim/corpus.jsonl --append

echo; echo "== 4. freeze the eval split"
$PY src/split.py

echo; echo "== 5. evaluate"
if [ -x "$VPY" ]; then
  USE_TF=0 $VPY src/evaluate.py --predictors majority,rules,laya:typed-decisions --split eval
else
  echo "  no venv -- baselines only. Run: python3 -m venv .venv && .venv/bin/pip install laya"
  $PY src/evaluate.py --predictors majority,rules --split eval
fi

echo; echo "== 6. training set"
$PY src/to_laya.py
