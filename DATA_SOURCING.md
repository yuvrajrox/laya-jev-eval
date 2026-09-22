# Where to source reply data

Read this first: **there is no public cold-outbound reply corpus.** I looked. The
closest public things are inbox-triage datasets with the wrong taxonomy, or aggregate
reply-*rate* benchmarks with no individual emails. Apollo.io published their
architecture for this exact problem and confirmed the same gap — they hand-labelled a
seed, used an LLM as a teacher to pseudo-label the rest, and trained a small fast model
on high-confidence output only. That is the plan below.

Sources are listed in priority order. Do them in this order; each one is cheaper and
lower-risk than a naive "export production" approach, and the first two need no
customer data at all.

---

## The CSV schema

Every source lands as a CSV with these columns. Only `body` is required. Column names
are matched case-insensitively against a list of aliases in `src/ingest.py`, so an
export that calls it `message` or `reply_body` or `text` works without editing anything.

| column | required | notes |
|---|---|---|
| `body` | **yes** | raw reply body. Do **not** pre-clean it — `src/clean.py` strips quoted thread, signature and disclaimer, and it needs the raw text to find them. |
| `subject` | strongly | carries most of the `ooo` / `bounce` signal. `Automatic reply:` in the subject is near-deterministic. |
| `from` | strongly | `mailer-daemon@` / `postmaster@` is how bounces are detected for free. |
| `headers` | if cheap | raw header blob. `Auto-Submitted`, `X-Autoreply`, `X-Failed-Recipients` are RFC-standard and give >99% precision on the two automated classes with zero model involvement. |
| `id` | no | stable id. Generated from a body hash if absent, which also dedupes. |
| `thread_id` | no | **get this if you can.** The split is grouped by thread; without it, two replies from the same conversation can straddle train and eval and inflate your score. |
| `intent` | no | one of the 7 labels, if already known |
| `user_action` | no | `mark_interested`, `mark_not_interested`, `mark_unsubscribed`, `mark_meeting_booked` — these are gold labels |
| `account`, `lang` | no | context; `account` is passed to the model, `lang` is for slicing results |

```bash
python3 src/ingest.py 'data/raw/*.csv' --source rox_dev --out data/interim/corpus.jsonl
```

---

## 1. Rox's own outbound mailboxes — start here

**Rox sells to revenue teams by running outbound. Your own SDRs' mailboxes are full of
real cold-outbound replies, in exactly the target distribution, and they are Rox's own
first-party data.** No customer correspondence, no tenant data, no "is this allowed"
question that needs a governance decision before you can start.

This is the single best answer to your compliance being unresolved: it sidesteps it
entirely. A team running sequences for a few months has thousands of replies.

- **How:** export from the connected Gmail mailboxes, or from your own Rox tenant's
  Sequences → Replies tab, which is the same data with the current intent label already
  attached.
- **Yield:** realistically 1,000–5,000 replies per active SDR-year.
- **Labels:** free. Whatever your existing classifier assigned, plus every time someone
  manually corrected a contact's state.
- **Get `thread_id` and the raw headers here** — you control this mailbox, so you can
  export the full RFC822 rather than a rendered body.

**Do this one first even if the others get approved.** It is the only source that is
in-distribution, labelled, and free of governance questions simultaneously.

## 2. Rox dev / devtwo tenants

Test sequences that actually sent and received. Lower volume and the replies are often
synthetic-ish (QA people replying to themselves), so treat this as a *plumbing* source:
it proves the export path and the column mapping work end to end, before you ask anyone
for permission to touch prod.

## 3. Rox production — the real training set, once cleared

This is where the 10k+ comes from, and the only place Tier-1 human labels exist at scale.

**The gold signals** (from the product brief — these are user actions, not model output):

| Rox signal | Label | Quality |
|---|---|---|
| Batch action "Mark Interested" | `interested` | gold |
| Batch action "Mark Not Interested" | `not_interested` | gold |
| "Mark Unsubscribed" | `unsubscribe` | gold |
| "Mark Meeting Booked" | `meeting_booked=true` | gold |
| Bounce detection | `bounce` | gold (system) |
| Contact state → `Paused` | `ooo` | **weak — do not train on directly** |
| Contact state → `Needs Attention` | ambiguous | useful as an escalation example |

A user clicking "Mark Interested" on a reply the system called `other` is a **corrected
error**. Those are the most valuable rows in the entire corpus — `src/to_laya.py`
oversamples them 3× by default for exactly this reason.

**Where it physically lives:** Sequences → Replies and Outbox per tenant. If reply and
engagement events land in the warehouse, one query beats any UI export. You have
Snowflake connectors available in this session; I can explore the schema whenever you
want to authenticate.

**Compliance shape to propose**, since this is the blocker: pull only `body`, `subject`,
`from`-domain (not local part), `thread_id` and the label; redact person names, emails,
phone numbers and URLs at export time with a regex pass before anything is written to
disk. The classifier does not need to know *who* replied — it needs to know what they
said. Redaction costs you almost nothing in accuracy here and turns the request from
"export customer correspondence" into "export redacted text spans", which is a far
easier conversation.

## 4. Public corpora — for `ooo` and `bounce` only

These two classes are ~70% of real reply volume and have not changed since 2001, so
public data transfers perfectly. Nothing else in these corpora is useful — they are
internal corporate email, not replies to cold outbound.

- **SpamAssassin public corpus** — <https://spamassassin.apache.org/old/publiccorpus/>
  Full RFC822 with headers intact, which is what you want for the header rules.
  **Measured yield: 4 bounces and zero out-of-office** from the two ham archives. The
  ham corpora are legitimate human mail, so they barely contain autoreplies. `src/public_data.py`
  works, but this source is close to worthless on its own — do not budget for it.
- **Enron — the better public source for `ooo`.** 500k messages of real corporate mail
  across several years, so it contains a genuine population of vacation autoresponders.
  [CMU original](https://www.cs.cmu.edu/~enron/), or on Hugging Face as
  [`corbt/enron-emails`](https://huggingface.co/datasets/corbt/enron-emails) (cleaned and
  structured) and [`LLM-PBE/enron-email`](https://huggingface.co/datasets/LLM-PBE/enron-email).
  ~500k messages; mine it for out-of-office autoreplies.
  [`Yale-LILY/aeslc`](https://huggingface.co/datasets/Yale-LILY/aeslc) is body+subject
  only with no headers or threads — less useful here.

`src/public_data.py` downloads SpamAssassin and mines both classes automatically.

**Target: ~150 real automated messages, mostly from Enron.** Do not take more. If you flood the corpus with
Enron autoreplies the model gets very good at the easy classes and you learn nothing
about the hard ones.

## 5. Synthetic — rare classes only, train split only

`data/seed/seed_replies.py` holds 100 hand-written replies, deliberately balanced, with
33 marked `hard=True` (the cases where reasonable annotators disagree) and 4 non-English.
Expand it with Claude for `unsubscribe`, `referral` and hostile replies, which real
stratified sampling will not give you enough of.

**Synthetic never goes in the eval split.** If it does, your numbers measure how well
Laya learned Claude's writing style, not how well it reads real prospects. The splitter
does not enforce this for you — it is a discipline, so check it.

---

## Volume targets

| Stage | Size | Composition | Labelling |
|---|---|---|---|
| **Smoke test** | 1,000 | 600 real Rox + 150 public auto + 250 synthetic | 300 hand-labelled eval, frozen |
| **First fine-tune** | 8–15k | mostly real, unstratified | teacher pseudo-labels at conf ≥ 0.75, gold oversampled 3× |
| **Steady state** | +2k/month | the escalation log | the cases Laya punted to Claude *are* next month's training set |

Hand-labelling the 300-row eval set takes 2–3 hours and is the highest-value work in the
whole project. Do it yourself, use `schema/annotation_guide.md` for the edge cases, and
freeze it before anyone looks at a model output.

---

## Stratify the eval set, then report both numbers

Real reply distribution is roughly 70% `ooo`/`bounce`. A random 1,000 gives you ~15
unsubscribes, and unsubscribe recall is the one metric with legal consequences attached
— you cannot measure it from 15 examples.

So: sample ~150 per class for eval, and separately estimate the true class frequencies
from an *unstratified* sample of a few hundred. Save those frequencies as JSON and pass
`--prior` to the harness. It will report stratified accuracy (which tells you whether
each class works) and production-mix accuracy (which predicts what users will see).
Quoting only one of them is how these projects mislead their own teams.
