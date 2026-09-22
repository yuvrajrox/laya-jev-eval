"""Metrics for the reply-intent harness. Pure Python, no numpy needed.

Two of these decide whether the project ships, and neither is accuracy:

  ECE / reliability  -- Laya's pitch is calibrated probabilities. Untuned it ships
                        over-confident (mean ECE 0.466 on the English checkpoint per
                        the model card). If ECE is not low after temperature refit,
                        every downstream threshold is meaningless.
  risk-coverage      -- the production design is a cascade: serve Laya when confident,
                        fall through to Claude otherwise. What matters is "accuracy on
                        the 85% it is most sure about", not accuracy on everything.
"""
import math
from collections import Counter, defaultdict


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def per_class(y_true, y_pred, labels):
    """Precision / recall / F1 / support per label."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    rows = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows[l] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[l] + fn[l]}
    return rows


def macro_f1(y_true, y_pred, labels):
    rows = per_class(y_true, y_pred, labels)
    present = [l for l in labels if rows[l]["support"]]
    return sum(rows[l]["f1"] for l in present) / len(present) if present else 0.0


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def reweighted_accuracy(y_true, y_pred, prior):
    """Accuracy as it would be on the true production mix.

    The eval set is stratified -- roughly equal counts per class -- because you cannot
    measure a class you have 8 examples of. Production is not: it is ~70% ooo/bounce.
    Pass the real class frequencies as `prior` (estimated from an unstratified sample)
    to recover the number that predicts production behaviour. Report both; quoting only
    the stratified number overstates how hard the job is, and only the reweighted one
    hides that unsubscribe recall is broken.
    """
    per = defaultdict(lambda: [0, 0])
    for t, p in zip(y_true, y_pred):
        per[t][1] += 1
        per[t][0] += (t == p)
    total = sum(prior.get(l, 0.0) for l in per)
    if not total:
        return None
    return sum((prior.get(l, 0.0) / total) * (c / n) for l, (c, n) in per.items() if n)


def ece(confidences, correct, bins=15):
    """Expected calibration error, equal-width binning (the model card's convention)."""
    if not confidences:
        return 0.0
    buckets = [[] for _ in range(bins)]
    for c, ok in zip(confidences, correct):
        buckets[min(bins - 1, int(c * bins))].append((c, ok))
    n = len(confidences)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b))
               for b in buckets if b)


def reliability_table(confidences, correct, bins=10):
    rows = []
    buckets = [[] for _ in range(bins)]
    for c, ok in zip(confidences, correct):
        buckets[min(bins - 1, int(c * bins))].append((c, ok))
    for i, b in enumerate(buckets):
        if not b:
            continue
        rows.append({"bin": "%.1f-%.1f" % (i / bins, (i + 1) / bins), "n": len(b),
                     "mean_conf": sum(c for c, _ in b) / len(b),
                     "accuracy": sum(ok for _, ok in b) / len(b)})
    return rows


def risk_coverage(confidences, correct, points=(1.0, .95, .9, .85, .8, .7, .6, .5)):
    """Accuracy on the most-confident X% -- the cascade economics table.

    Read it as: "if we auto-serve the top 80% by confidence, we are right 96% of the
    time there, and the remaining 20% escalates to Claude." That pair of numbers is
    the entire business case, and it is what you tune the threshold against.
    """
    order = sorted(zip(confidences, correct), key=lambda x: -x[0])
    n, out = len(order), []
    for cov in points:
        k = max(1, int(round(n * cov)))
        head = order[:k]
        out.append({"coverage": cov, "n": k,
                    "accuracy": sum(ok for _, ok in head) / k,
                    "threshold": head[-1][0]})
    return out


def brier(prob_vectors, y_true_idx):
    """Multiclass Brier score -- a strictly proper rule, which is what RLCD optimises."""
    if not prob_vectors:
        return 0.0
    total = 0.0
    for p, y in zip(prob_vectors, y_true_idx):
        total += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
    return total / len(prob_vectors)


def fit_temperature(prob_vectors, y_true_idx, lo=0.05, hi=5.0, iters=60):
    """Refit one temperature by minimising NLL via ternary search on a convex-ish curve.

    Non-negotiable before trusting any probability. The model card reports mean ECE
    0.466 -> 0.081 from exactly this step, fitted per (question type, option count).
    Fit on a calibration split, never on the frozen eval set.
    """
    def nll(T):
        total = 0.0
        for p, y in zip(prob_vectors, y_true_idx):
            logits = [math.log(max(pi, 1e-12)) / T for pi in p]
            mx = max(logits)
            lse = mx + math.log(sum(math.exp(l - mx) for l in logits))
            total -= logits[y] - lse
        return total / max(1, len(prob_vectors))

    if len(prob_vectors) < 200:
        # On a small set NLL is minimised by sharpening toward certainty, so the fit
        # collapses toward lo and the "calibrated" model becomes maximally confident.
        import warnings
        warnings.warn("temperature fitted on %d samples; use >=200 per bucket or the "
                      "fit degenerates" % len(prob_vectors))
    for _ in range(iters):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


def apply_temperature(p, T):
    logits = [math.log(max(pi, 1e-12)) / T for pi in p]
    mx = max(logits)
    exps = [math.exp(l - mx) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def format_confusion(m, labels, width=14):
    short = [l[:8] for l in labels]
    head = " " * width + "".join("%8s" % s for s in short) + "     (rows = true)"
    lines = [head]
    for i, l in enumerate(labels):
        lines.append("%-*s" % (width, l[:width - 1]) + "".join("%8d" % v for v in m[i]))
    return "\n".join(lines)
