# laya-jev-eval

Two halves, on the same 150M encoder.

**1. A decision-model bake-off.** [Laya](https://huggingface.co/convaiinnovations/laya)
(open weights, fine-tunable) against [TypeSafe Jev](https://docs.typesafe.ai/) (closed
API) on email intent, with byte-identical instructions and criteria from one task spec —
published Jev-vs-Laya figures compare different prompts on different samples, so they
settle nothing.

**2. Making that same model generate.** A decision model picks from a list and cannot
produce text. But Laya's backbone is ModernBERT, pretrained as a masked language model;
restoring that head turns the classifier into a generator on the same weights. It writes
sentences, and it draws 16×16 pixel art — all positions decoded in parallel, never left
to right.

| | |
|---|---|
| Jev vs Laya, 100 hand-written emails | **98% vs 98%** — and only one has usable confidence |
| Explain a decision in a written sentence | **179 ms**, local, $0 |
| Draw a pet from a caption | **95.9%** cell match on unseen prompts |

Demo clips: [`demo/clips/`](demo/clips) · run it yourself: `python3 demo/server.py`

## Part 1 — Laya vs Jev

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

## Part 1 — caveats that matter

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

## Part 1 — cost and speed

| | Laya (local, M-series GPU) | Jev (API) |
|---|---|---|
| per email | 6–7 ms batched | ~150 ms @ concurrency 8 |
| cost | $0 | ~$0.000017 |
| fine-tuning | 25 min, 1,298 examples | not possible |

Total Jev spend for every run in this repo: **$0.045**.

## Part 2 — generation from the same encoder

Laya's decision head scores options at `[MASK]` tokens, which is the masked-language-model
interface. Laya's authors discarded ModernBERT's MLM head when they bolted the decision
head on; putting it back gives a generator for free.

Decoding is **mask-predict**, not autoregression: mask every position, predict them all in
one pass, keep the most confident, re-mask the rest, repeat. A sentence or an image
resolves like a crossword rather than a sequence.

### Text — explaining a decision

Two ways to attach a reason to a classification, scored on the 100-email holdout by
feeding each reason back into the classifier alone and checking it reproduces the
decision. An explanation that cannot reproduce the decision it explains is decoration.

| method | faithful | notes |
|---|---|---|
| **retrieval** — one choice question over 12 written sentences | **88%** | cannot hallucinate; ships today |
| **generation** — mask-predict on the MLM head | **78%** | novel text, untrained for summarisation |

Decision plus both reasons costs **162 ms** per email, local, $0.

Generation is grammatical but unreliable on long emails — it is a base MLM that has never
been trained to summarise. When it lands it carries specifics no fixed bank could hold:

```
"What's the notice period on the annual contract?"
   -> "the customer is asking about the annual contract renewal period"

"Lovely to deal with a company that answers the phone. That alone."
   -> "the customer is happy that the company answers the phone"
```

Two negative results are kept in the repo because they explain why this design is the one
that works:

- `src/wordgen.py` — choosing one word at a time from a 75-word vocabulary. Produces
  `email twice twice twice...`: the decision head knows which words are relevant but has
  no notion of grammar or position.
- `src/pixelgen.py` — asking Laya for each pixel's colour independently. Returns a flat
  red rectangle at 0.518 mean confidence. Each cell is answered in isolation, and an image
  is made entirely of the relationships between neighbouring cells.

### Images — pixel pets from a caption

A 16×16 grid flattens to 256 single-token cells (16 colours mapped to `A`–`P`), which fits
in ModernBERT's context. That is the whole unlock: every cell attends to every other cell,
which per-cell prediction could not do.

Training data is procedural (`src/avatars.py`) — 6,000 sprites from 8 hand-drawn templates
across 40,960 possible combinations. Procedural because the caption is then correct by
construction, with no licensing questions.

**The one fix that mattered.** Sampling the training mask ratio uniformly over 30–100%
leaves too few near-fully-masked examples, and generation always starts fully masked:

| mask schedule | cell accuracy from a blank canvas |
|---|---|
| uniform 30–100%, 1 epoch | 0.238 |
| MaskGIT cosine, 1 epoch | 0.900 |
| MaskGIT cosine, 2 epochs | 0.973 |
| MaskGIT cosine, 3 epochs | 0.974 |

On 40 captions never seen in training: **95.9% mean cell match, 5 of 40 pixel-perfect.**
Inpainting a real sprite with half its cells erased: **95%**.

Generation takes ~720 ms unbatched at 12 rounds, ~230 ms at 1 round, on an M-series GPU.

**Scope.** The model learned to recolour and place 8 hand-drawn templates, generalising to
colour combinations it never saw. It will not invent a species nobody drew. Note also that
the `render()` function which produced the training data draws these sprites perfectly and
instantly — as a product this is a slower, less accurate copy of code that already exists.
Its value is proving the pipeline for a domain where no such function exists.

## Demo

```bash
python3 demo/server.py          # http://localhost:8000, models load on first use
python3 demo/record.py          # drives it with Playwright -> demo/clips/*.mp4
```

The page runs both halves live: a caption draws a pet with the mask-predict rounds
animating, and an email returns a decision plus a written reason. `record.py` uses
Playwright's video capture and the ffmpeg bundled in `imageio-ffmpeg`, so recording needs
no system install.

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

src/decide_explain.py          decision + retrieved reason + generated reason
src/avatars.py                 procedural 16x16 pet dataset (6,000 captioned sprites)
src/train_avatar.py            mask-predict training, MaskGIT cosine schedule
src/gen_avatar.py              draw a pet from a caption
src/wordgen.py                 negative result: per-word choice decoding
src/pixelgen.py                negative result: per-pixel independent choice
demo/                          local server, page, Playwright recorder, MP4 clips
data/avatars/train.jsonl       the generated sprite dataset
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

Generation:

```bash
python3 src/avatars.py 6000                                  # build the sprite dataset
USE_TF=0 .venv/bin/python src/train_avatar.py 3 checkpoints/avatar-mlm   # ~35 min
USE_TF=0 .venv/bin/python src/gen_avatar.py                  # draw from a caption
USE_TF=0 .venv/bin/python src/decide_explain.py runs/<timestamp>  # decision + reasons
```

Avatar checkpoints are gitignored at ~585 MB each and rebuild in ~35 minutes.

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
