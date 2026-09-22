"""How similar is each test row to its closest training row?

A grouped split removes *exact* duplicates. It does nothing about *template* duplicates:
"I want to change my payment method" and "I want to change my phone number" are distinct
strings drawn from one pattern. If the test set is full of those, a held-out score is
measuring pattern recall, not generalisation -- and the number will not survive contact
with real email.
"""
import json
import sys
from collections import Counter, defaultdict

train = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
test = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8")]


def toks(t):
    return set(t.lower().replace(".", " ").replace(",", " ").replace("?", " ").split())


# Inverted index so this stays fast rather than 585 x 1415 pairwise.
index = defaultdict(list)
tr_toks = [toks(r["body"]) for r in train]
for i, ts in enumerate(tr_toks):
    for t in ts:
        index[t].append(i)

buckets = Counter()
worked, examples = 0, []
for r in test:
    ts = toks(r["body"])
    cand = Counter()
    for t in ts:
        for i in index[t]:
            cand[i] += 1
    best, best_i = 0.0, None
    for i, _ in cand.most_common(60):
        j = len(ts & tr_toks[i]) / max(1, len(ts | tr_toks[i]))
        if j > best:
            best, best_i = j, i
    buckets[min(int(best * 10), 9)] += 1
    if best >= 0.8:
        worked += 1
        if len(examples) < 6 and best < 1.0:
            examples.append((round(best, 2), r["body"], train[best_i]["body"],
                             r["label"], train[best_i]["label"]))

n = len(test)
print("Jaccard token similarity of each TEST row to its nearest TRAIN row\n")
print("  %-14s %6s %7s" % ("similarity", "rows", "share"))
for b in range(9, -1, -1):
    if buckets[b]:
        print("  %.1f - %.1f      %6d %6.1f%%" % (b / 10, (b + 1) / 10, buckets[b], 100 * buckets[b] / n))
print("\n  test rows >= 0.8 similar to a training row: %d / %d (%.1f%%)" % (worked, n, 100 * worked / n))
print("\nnear-duplicate examples (test vs its closest train row):")
for sim, a, b, la, lb in examples:
    print("  sim %.2f  [%s] %r\n            [%s] %r" % (sim, la, a[:64], lb, b[:64]))
