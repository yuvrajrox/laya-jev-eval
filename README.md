# laya-jev-eval

Head-to-head evaluation of two "System One" decision models on email intent
classification: [Laya](https://huggingface.co/convaiinnovations/laya) (open weights,
self-hosted, fine-tunable) and [TypeSafe Jev](https://docs.typesafe.ai/) (closed API).

Both models were given byte-identical instructions and criteria from a single task spec,
which is the point — published Jev-vs-Laya figures compare different prompts on different
samples.

## Results

Task: 4-way intent (`request` / `inquiry` / `complaint` / `feedback`).
Laya checkpoint: `laya-typed-decisions`. Jev: `jev-1.13.0`.

### Supplied dataset (2,000 rows, 1,210 unique texts) — neither model trained on it

| model | correct | accuracy |
|---|---|---|
| Laya, zero-shot | 1904 / 2000 | 95.20% |
| Jev 1.13.0 | 1920 / 2000 | 96.00% |

### Hand-written holdout (100 emails) — the only clean test

| model | correct | accuracy | macro F1 | ECE | Brier |
|---|---|---|---|---|---|
| Laya, zero-shot | 91 / 100 | 91.00% | 0.907 | 0.100 | 0.122 |
| Laya, fine-tuned | 98 / 100 | 98.00% | 0.980 | 0.021 | 0.040 |
| Jev 1.13.0 | 98 / 100 | 98.00% | 0.980 | 0.024 | 0.023 |

Paired McNemar on the holdout:

- Jev vs Laya zero-shot — Jev won 8, lost 1, **p = 0.039** (significant)
- Jev vs Laya fine-tuned — won 1, lost 1, **p = 1.0** (a tie)
- Laya fine-tuned vs zero-shot — fixed 7, broke 0, **p = 0.016** (significant)

### The finding that isn't in the accuracy column

Identical 98% accuracy, opposite behaviour under uncertainty:

| model | confidence when **wrong** | rows at ≥0.99 conf |
|---|---|---|
| Laya, zero-shot | 0.37 – 0.84 | 0 / 100 |
| Laya, fine-tuned | **1.00** | 98 / 100 |
| Jev 1.13.0 | 0.61 – 0.79 | 86 / 100 |

Fine-tuning raised accuracy but collapsed the model to certainty — it reports 1.00 on
both answers it gets wrong, so no confidence threshold separates its errors from its
successes. Jev's errors sit below its correct answers, so they can be routed to a larger
model. A cascade design ("serve the cheap model when confident, escalate otherwise")
works with Jev and does not work with the fine-tuned Laya as trained here.

Note this rests on 2 errors per model. On the 585-row split Jev's errors had a median
confidence of 0.97, so the separation may not hold at scale.

## Caveats that matter

- **The supplied dataset is template-generated.** 790 of 2,000 rows are exact duplicates
  (1,210 unique texts), and in the held-out split 39.3% of test rows are ≥0.8 token-similar
  to a training row. The closest pairs differ by one substituted word. No number measured
  on it predicts real-world accuracy.
- **A random 70/30 split leaks 49.7% of the test set into training.** The split here is
  grouped by text, giving 0% overlap. Both splits are in `runs/*/data/` for comparison.
- **Do not compare fine-tuned Laya to Jev on the template data.** Fine-tuned Laya scores
  100.00% on the 585-row split, but it trained on 70% of that data; Jev never saw any of
  it. Only the hand-written holdout is a fair comparison.
- **The labels follow the supplied dataset's convention.** Fine-tuned Laya was trained on
  that convention; Jev only ever saw four one-line criteria strings. Jev vs Laya-zero-shot
  is like-for-like; Jev vs Laya-fine-tuned is not.
- **n=100, single run, one seed.** The 95% interval on 98% at n=100 is roughly 93–99.5%.
- All three models score identically on the 13 hard cases (0.846). Neither fine-tuning nor
  Jev moves the genuinely ambiguous examples.
- All three call *"Third time asking. Please stop sending marketing texts. I have
  unsubscribed twice already."* a `request` against a `complaint` label. Three independent
  models agreeing against a label is evidence about the label.

## Cost and speed

| | Laya (local, M-series GPU) | Jev (API) |
|---|---|---|
| per email | 6–7 ms batched | ~150 ms @ concurrency 8 |
| cost | $0 | ~$0.000017 |
| fine-tuning | 25 min, 1,298 examples | not possible |

Total Jev spend for every run in this repo: **$0.045**.

## Layout

```
schema/taxonomy.py             Rox reply-intent task (7 labels) — the original target task
schema/annotation_guide.md     labelling rules and edge cases
src/tasks/email_intent.py      the 4-label task actually evaluated here
src/clean.py                   quoted-thread / signature stripping, ooo+bounce rules
src/ingest.py                  any CSV -> canonical JSONL
src/prep.py                    grouped 70/30 split (+ the naive split, for comparison)
src/layakit.py                 batched encode/predict over a Laya checkpoint
src/predict.py                 Laya inference -> predictions.jsonl
src/predict_jev.py             Jev HTTP inference -> same format (stdlib only, no SDK)
src/train.py                   fine-tuning
src/metrics.py                 accuracy, per-class, ECE, Brier, risk-coverage, temperature
src/report.py                  markdown + json report, McNemar
src/compare3.py                three-way comparison
src/nearfdup.py                train/test near-duplicate analysis
data/handwritten/              the 100 hand-written holdout emails
data/seed/                     100 hand-written Rox sales-reply examples
data/jev_cache/                1,310 cached Jev responses — reruns cost nothing
runs/<timestamp>/              data, splits, checkpoints, predictions, reports
```

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install laya
RUN=runs/$(date +%Y%m%d-%H%M%S)-email-intent
mkdir -p $RUN/{data,checkpoints,reports,logs}

python3 src/prep.py --csv email_intent.csv --run-dir $RUN
USE_TF=0 .venv/bin/python src/predict.py --model-dir <laya-typed-decisions> \
    --data $RUN/data/02_grouped_test.jsonl --out $RUN/reports/zeroshot.jsonl --tag zs --fp32
USE_TF=0 .venv/bin/python src/train.py --model-dir <laya-typed-decisions> \
    --train $RUN/data/02_grouped_train.jsonl --run-dir $RUN --epochs 3
export TYPESAFE_API_KEY=...
python3 src/predict_jev.py --data $RUN/data/04_holdout100.jsonl --out $RUN/reports/jev.jsonl
python3 src/compare3.py $RUN
```

The fine-tuned checkpoint (1.6 GB) is attached to the GitHub Release rather than tracked
in git. `src/train.py` reproduces it in ~25 minutes on a GPU.

## Notes on method

`src/train.py` optimises Laya's own `proper_reward` (imported from the library, not
reimplemented) directly by gradient descent, rather than through the REINFORCE estimator
RLCD uses. With hard single-turn labels and a differentiable output, sampling adds
variance without adding exploration. Same objective, lower-variance estimator — but it is
not literally RLCD, and it leaves the model's act/escalate head untrained, which is a
plausible contributor to the confidence collapse above.

fp16 autocast on MPS produced NaN logits for an entire batch during one holdout run, and a
NaN row silently argmaxes to label 0 — indistinguishable from a wrong prediction.
`src/layakit.py` now retries any non-finite batch in fp32 and raises if that fails.
