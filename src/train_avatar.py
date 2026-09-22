"""Train a masked transformer to draw 16x16 pixel pets from a caption.

This is MaskGIT's recipe with ModernBERT as the backbone. The image is 256 discrete
cells, flattened into the sequence after the caption, one token per cell. Because all 256
cells sit in ONE sequence, every cell attends to every other cell -- which is precisely
what per-cell independent prediction could not do, and why that experiment produced a red
rectangle.

Training detail that matters: the mask ratio is sampled per example from ~30% to 100%,
not fixed at BERT's 15%. Generation starts from a fully masked grid, so the model has to
have seen that regime during training. Train only at 15% and it can refine an almost
finished image but cannot start one.
"""
import json, math, os, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "answerdotai/ModernBERT-base"
LETTERS = "ABCDEFGHIJKLMNOP"
N = 16


def build(tok, caption, grid):
    text = caption + " | " + " ".join(grid)
    return tok(text, return_tensors=None)["input_ids"]


def main():
    ap_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/avatar-mlm"
    rng = random.Random(11); torch.manual_seed(11)

    tok = AutoTokenizer.from_pretrained(MODEL)
    resume = os.environ.get("RESUME_FROM")
    mdl = AutoModelForMaskedLM.from_pretrained(resume or MODEL)
    if resume: print("resuming from %s" % resume)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mdl.to(dev)

    letter_ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in LETTERS]
    first_id = tok(LETTERS[0], add_special_tokens=False)["input_ids"][0]

    rows = [json.loads(l) for l in open("data/avatars/train.jsonl")]
    rng.shuffle(rows)
    val, train = rows[:300], rows[300:]
    print("train %d / val %d | device %s" % (len(train), len(val), dev))

    enc = [build(tok, r["caption"], r["grid"]) for r in train]
    venc = [build(tok, r["caption"], r["grid"]) for r in val]
    L = max(len(e) for e in enc)
    print("sequence length %d (caption + 256 cells)" % L)

    def batchify(items, idxs):
        ids = torch.full((len(idxs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idxs), L), dtype=torch.long)
        for j, i in enumerate(idxs):
            e = items[i]; ids[j, :len(e)] = torch.tensor(e); att[j, :len(e)] = 1
        return ids, att

    def cell_mask(ids):
        """True where the token is one of the 16 cell letters."""
        m = torch.zeros_like(ids, dtype=torch.bool)
        for lid in letter_ids:
            m |= (ids == lid)
        m |= (ids == first_id)
        return m

    opt = torch.optim.AdamW(mdl.parameters(), lr=5e-5, weight_decay=0.01)
    steps = ap_epochs * math.ceil(len(enc) / 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, total_steps=steps, pct_start=0.1)

    def run_batch(items, idxs, train_mode=True, fixed_ratio=None):
        ids, att = batchify(items, idxs)
        ids, att = ids.to(dev), att.to(dev)
        cm = cell_mask(ids)
        # MaskGIT's cosine schedule: ratio = cos(pi/2 * u) for u ~ U(0,1). Uniform sampling
        # puts only ~14% of examples above 90% masked, which is far too few to learn
        # generating from a blank canvas -- the regime every generation actually starts in.
        # The cosine puts substantially more mass near 100%.
        if fixed_ratio:
            ratio = fixed_ratio
        else:
            u = torch.rand(len(idxs), 1, device=dev)
            ratio = torch.cos(math.pi / 2 * u).clamp(0.15, 1.0)
        keep = torch.rand(ids.shape, device=dev) < (ratio if torch.is_tensor(ratio) else ratio)
        mask_here = cm & keep
        inp = ids.clone(); inp[mask_here] = tok.mask_token_id
        logits = mdl(input_ids=inp, attention_mask=att).logits
        if mask_here.sum() == 0:
            return None, 0.0
        loss = F.cross_entropy(logits[mask_here], ids[mask_here])
        acc = (logits[mask_here].argmax(-1) == ids[mask_here]).float().mean().item()
        return loss, acc

    t0 = time.time()
    order = list(range(len(enc)))
    for ep in range(1, ap_epochs + 1):
        mdl.train(); rng.shuffle(order); tot = n = 0
        for s in range(0, len(order), 16):
            loss, acc = run_batch(enc, order[s:s + 16], True)
            if loss is None: continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); n += 1
            if n % 60 == 0:
                print("  ep%d %4d/%d loss %.4f acc %.3f" % (ep, s, len(order), tot / n, acc), flush=True)
        mdl.eval()
        with torch.no_grad():
            va = [run_batch(venc, list(range(i, min(i + 16, len(venc)))), False, 1.0)[1]
                  for i in range(0, len(venc), 16)]
        print("epoch %d  loss %.4f  val cell-accuracy @100%% masked: %.4f  (%.1f min)"
              % (ep, tot / max(1, n), sum(va) / len(va), (time.time() - t0) / 60), flush=True)
        # Save every epoch, not just at the end -- a long run that can only be stopped by
        # throwing away all of its progress is a bad run.
        os.makedirs(out_dir, exist_ok=True)
        mdl.save_pretrained(out_dir); tok.save_pretrained(out_dir)
        print("  checkpoint saved after epoch %d" % ep, flush=True)

    os.makedirs(out_dir, exist_ok=True)
    mdl.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    print("saved -> %s  (%.1f min)" % (out_dir, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
