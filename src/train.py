"""Fine-tune a Laya checkpoint on a labelled split.

Objective. Laya is trained with RLCD: the policy reports a distribution, exploration
adds Gaussian noise to the logits, and the reward is a strictly proper scoring rule
(log + spherical). Expected reward is maximised only by reporting honest probabilities.

This trainer optimises that same reward -- it imports `proper_reward` from the library
rather than reimplementing it -- but directly, by gradient ascent on the reward of the
reported distribution, instead of through the REINFORCE estimator. The reason is that
REINFORCE's noise exists to explore an answer space you cannot differentiate through.
Here the targets are hard single-turn labels and the distribution is fully
differentiable, so sampling buys no exploration and only adds variance, which on 1,415
examples is the difference between a clean result and a noisy one. Same objective,
lower-variance estimator.

A validation slice is carved out of TRAIN for picking the best epoch. The test split is
never touched during training -- selecting an epoch on test is the most common way these
comparisons quietly cheat.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from laya.common import collate_items, proper_reward

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import email_intent as TASK  # noqa: E402
from layakit import encode, load_checkpoint, predict_probs, snapshot_checkpoint  # noqa: E402


def make_items(recs, tok, cfg):
    items = []
    for r in recs:
        y = TASK.LABELS.index(r["label"])
        target = [1.0 if i == y else 0.0 for i in range(len(TASK.LABELS))]
        it = encode(tok, cfg, TASK.state_for(r), TASK.QUESTIONS["intent"], target=target, label=y)
        items.append(it)
    return items


def evaluate(model, tok, cfg, recs, device, batch_size):
    probs = predict_probs(model, tok, cfg, [TASK.state_for(r) for r in recs],
                          TASK.QUESTIONS["intent"], device, batch_size)
    pred = [TASK.LABELS[int(p.argmax())] for p in probs]
    return sum(p == r["label"] for p, r in zip(pred, recs)) / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    recs = [json.loads(l) for l in open(args.train, encoding="utf-8")]
    # Validation groups are split off by `group`, so duplicate texts cannot straddle
    # train and val either -- the same leak, one level down.
    groups = sorted({r["group"] for r in recs})
    random.Random(args.seed).shuffle(groups)
    val_groups = set(groups[:max(1, int(len(groups) * args.val_frac))])
    tr = [r for r in recs if r["group"] not in val_groups]
    va = [r for r in recs if r["group"] in val_groups]

    model, tok, cfg, dev = load_checkpoint(args.model_dir, args.device)
    print("device=%s | train=%d val=%d | epochs=%d bs=%d lr=%g"
          % (dev, len(tr), len(va), args.epochs, args.batch_size, args.lr))

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    enc_p = [p for p in model.encoder.parameters() if p.requires_grad]
    head_p = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": args.lr},
                             {"params": head_p, "lr": args.head_lr}], weight_decay=0.01)

    items_tr = make_items(tr, tok, cfg)
    steps = args.epochs * ((len(items_tr) + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr],
                                                total_steps=steps, pct_start=0.1)

    ckpt_dir = os.path.join(args.run_dir, "checkpoints", "finetuned")
    log_path = os.path.join(args.run_dir, "logs", "train_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")

    base_val = evaluate(model, tok, cfg, va, dev, args.eval_batch_size)
    print("epoch 0 (zero-shot)  val_acc=%.4f" % base_val)
    log.write(json.dumps({"epoch": 0, "val_acc": base_val, "note": "zero-shot"}) + "\n")

    best, best_epoch, history = base_val, 0, []
    order = list(range(len(items_tr)))
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(order)
        total, nb = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = [items_tr[i] for i in order[start:start + args.batch_size]]
            b = collate_items([batch], tok.pad_token_id)
            logits, _ = model(b["input_ids"].to(dev), b["attention_mask"].to(dev),
                              b["marker_pos"].to(dev), b["marker_mask"].to(dev),
                              b["qtype"].to(dev))
            q = torch.softmax(logits, -1)
            r = proper_reward(q, b["target"].to(dev), b["qtype"].to(dev), b["marker_mask"].to(dev))
            loss = -r.mean()                     # maximise the strictly proper score
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            total += float(loss.item())
            nb += 1
            if nb % 25 == 0:
                print("  ep%d %4d/%d  loss %.4f" % (epoch, start + len(batch), len(order), total / nb), flush=True)

        val = evaluate(model, tok, cfg, va, dev, args.eval_batch_size)
        rec = {"epoch": epoch, "train_loss": total / max(1, nb), "val_acc": val,
               "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("epoch %d  loss=%.4f  val_acc=%.4f%s"
              % (epoch, rec["train_loss"], val, "  <- best" if val > best else ""))
        if val > best:
            best, best_epoch = val, epoch
            snapshot_checkpoint(args.model_dir, ckpt_dir, model, cfg)

    if best_epoch == 0:
        print("\n! no epoch beat the zero-shot validation score; saving the final model anyway")
        snapshot_checkpoint(args.model_dir, ckpt_dir, model, cfg)

    meta = {"base_model": args.model_dir, "best_epoch": best_epoch, "best_val_acc": best,
            "zero_shot_val_acc": base_val, "history": history, "args": vars(args),
            "train_n": len(tr), "val_n": len(va), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(meta, open(os.path.join(args.run_dir, "checkpoints", "train_meta.json"), "w"), indent=2)
    log.close()
    print("\nbest epoch %d (val %.4f) -> %s" % (best_epoch, best, ckpt_dir))


if __name__ == "__main__":
    main()
