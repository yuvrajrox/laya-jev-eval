# Laya on `email_intent.csv` — zero-shot vs fine-tuned

Task: 4-way customer email intent (`request` / `inquiry` / `complaint` / `feedback`).
Base checkpoint: `laya-typed-decisions`. Split: 70/30, grouped by text.

## Headline

| run | data | accuracy | macro F1 | ECE |
|---|---|---|---|---|
| zero-shot (no training) | all 2000 rows | 0.9520 | 0.9510 | 0.0762 |
| zero-shot (no training) | test 585 | 0.9504 | 0.9522 | 0.0782 |
| **fine-tuned on 70%** | test 585 | **1.0000** | 1.0000 | 0.0000 |

**Change from fine-tuning: +0.0496 accuracy (+4.96 points), +0.0478 macro F1, -0.0782 ECE.**

### Is the difference real? (McNemar exact test, paired)

Of the 585 test rows, fine-tuning **fixed 29** that zero-shot got wrong and **broke 0** that zero-shot got right.

**p = 3.725e-09** — the difference is statistically significant at the 0.05 level.

## Data and split

| | value |
|---|---|
| total rows | 2000 |
| unique texts | 1210 |
| rows that duplicate another row | 790 |
| grouped split | 1415 train / 585 test |
| test rows also present in train (grouped) | 0 (0.0%) |
| test rows also present in train (naive random) | 298 (49.7%) |

790 of the 2000 rows are exact duplicates of another row. A plain random 70/30 split puts **49.7% of the test set verbatim into training**, so the model scores by recalling a string it has already seen. The split used here groups every copy of a text onto one side, giving 0% overlap.

## Training

| | value |
|---|---|
| epochs run | 3 |
| best epoch (by held-out val) | 1 |
| val accuracy, zero-shot | 0.9316 |
| val accuracy, best | 1.0000 |
| train / val rows | 1298 / 117 |
| learning rate (encoder / head) | 1e-05 / 0.0001 |
| wall clock | 3.8 min |

Epoch selection used a validation slice carved out of **train**, grouped the same way. The test split was never used for selection.

## How much is the 100% worth? (near-duplicate analysis)

The grouped split removes rows that are *exactly* the same text. It does nothing about rows generated from the same template. So before reading anything into a perfect score, here is how close each test row is to its nearest training row, by token overlap:

```
Jaccard token similarity of each TEST row to its nearest TRAIN row

  similarity       rows   share
  0.9 - 1.0          11    1.9%
  0.8 - 0.9         219   37.4%
  0.7 - 0.8         191   32.6%
  0.6 - 0.7          86   14.7%
  0.5 - 0.6          68   11.6%
  0.4 - 0.5          10    1.7%

  test rows >= 0.8 similar to a training row: 230 / 585 (39.3%)

near-duplicate examples (test vs its closest train row):
  sim 0.80  [complaint] 'My delivery is broken and I need a fix.'
            [complaint] 'My delivery is broken and I need a upgrade.'
  sim 0.80  [complaint] 'Still no response regarding my delay, this is disappointed.'
            [complaint] 'Still no response regarding my overcharge, this is disappointed.'
  sim 0.82  [complaint] "This is the fourth time I'm contacting you about the overcharge."
            [complaint] "This is the third time I'm contacting you about the overcharge."
  sim 0.82  [request] 'I would like to request a refund for my delivery.'
            [request] 'I would like to request a cancellation for my delivery.'
  sim 0.82  [complaint] 'I am writing to complain about the dissatisfied medical staff.'
            [complaint] 'I am writing to complain about the disappointed medical staff.'
  sim 0.86  [inquiry] 'Dear Team, can you please update me on the status of my flight?'
            [inquiry] 'Dear Team, can you please update me on the status of my laptop?'
```

**This dataset is template-generated.** 39.3% of test rows are >=0.8 similar to a training row, and the closest pairs differ by a single substituted word (`third time` -> `fourth time`, `flight` -> `laptop`). No test row is below 0.4 similarity to something in training.

So the 1.0000 is real *for this corpus* and close to meaningless as a prediction of real-world accuracy. What it demonstrates is that the fine-tuning pipeline works end to end and that the model absorbs a labelling scheme from ~1,300 examples. What it does **not** demonstrate is that Laya will hit 100% on genuine customer email, which is messy, longer, and not drawn from 20 sentence patterns.

## Detail

### Zero-shot — test split

| metric | value |
|---|---|
| rows | 585 |
| **accuracy** | **0.9504** |
| macro F1 | 0.9522 |
| ECE (calibration error, lower better) | 0.0782 |
| Brier | 0.1025 |
| mean confidence | 0.8747 |

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| request | 0.986 | 0.844 | 0.910 | 167 |
| inquiry | 0.857 | 1.000 | 0.923 | 138 |
| complaint | 0.973 | 0.986 | 0.980 | 147 |
| feedback | 1.000 | 0.992 | 0.996 | 133 |

```
confusion (rows = true label)
               request inquirycomplainfeedback     (rows = true)
request            141      23       3       0
inquiry              0     138       0       0
complaint            2       0     145       0
feedback             0       0       1     132
```

### Fine-tuned — test split

| metric | value |
|---|---|
| rows | 585 |
| **accuracy** | **1.0000** |
| macro F1 | 1.0000 |
| ECE (calibration error, lower better) | 0.0000 |
| Brier | 0.0000 |
| mean confidence | 1.0000 |

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| request | 1.000 | 1.000 | 1.000 | 167 |
| inquiry | 1.000 | 1.000 | 1.000 | 138 |
| complaint | 1.000 | 1.000 | 1.000 | 147 |
| feedback | 1.000 | 1.000 | 1.000 | 133 |

```
confusion (rows = true label)
               request inquirycomplainfeedback     (rows = true)
request            167       0       0       0
inquiry              0     138       0       0
complaint            0       0     147       0
feedback             0       0       0     133
```

### Zero-shot — all 2000 rows

| metric | value |
|---|---|
| rows | 2000 |
| **accuracy** | **0.9520** |
| macro F1 | 0.9510 |
| ECE (calibration error, lower better) | 0.0762 |
| Brier | 0.0948 |
| mean confidence | 0.8769 |

| label | precision | recall | F1 | n |
|---|---|---|---|---|
| request | 0.984 | 0.827 | 0.899 | 509 |
| inquiry | 0.863 | 1.000 | 0.926 | 477 |
| complaint | 0.980 | 0.986 | 0.983 | 508 |
| feedback | 0.994 | 0.998 | 0.996 | 506 |

```
confusion (rows = true label)
               request inquirycomplainfeedback     (rows = true)
request            421      76       9       3
inquiry              0     477       0       0
complaint            7       0     501       0
feedback             0       0       1     505
```

## Confidence-based routing (test split)

Accuracy on the most-confident X%% of rows. This is the cascade table: serve the model where it is confident, escalate the rest.

| coverage | zero-shot acc | fine-tuned acc |
|---|---|---|
| 100% | 0.9504 | 1.0000 |
| 95% | 0.9640 | 1.0000 |
| 90% | 0.9677 | 1.0000 |
| 85% | 0.9678 | 1.0000 |
| 80% | 0.9679 | 1.0000 |
| 70% | 0.9707 | 1.0000 |
| 60% | 0.9715 | 1.0000 |
| 50% | 0.9692 | 1.0000 |
