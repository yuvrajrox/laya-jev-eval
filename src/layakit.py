"""Batched encode / predict / train over a Laya checkpoint.

The SDK answers one state per call, which is fine for a request handler and far too slow
for 2,000 evaluation rows or a fine-tuning loop. These helpers go straight at the
underlying DecisionModel using the library's own primitives, so tokenisation and option
markers stay byte-identical to what the SDK does at inference.
"""
import json
import os

import numpy as np
import torch
from laya.common import (QTYPES, amp_dtype, build_model, build_sequence, collate_items,
                         render_options)


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(model_dir, device=None):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    dev = pick_device(device)
    model.to(dev)
    model.encoder.config.reference_compile = False
    return model, tok, cfg, dev


def snapshot_checkpoint(src_dir, dst_dir, model, cfg):
    """Write a fine-tuned checkpoint that `laya.load()` can open directly."""
    import shutil
    from safetensors.torch import save_file
    os.makedirs(dst_dir, exist_ok=True)
    for sub in ("tokenizer", "encoder"):
        dst = os.path.join(dst_dir, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(src_dir, sub), dst)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(dst_dir, "model.safetensors"))
    json.dump(cfg, open(os.path.join(dst_dir, "rl_agent_config.json"), "w"), indent=2)
    return dst_dir


def to_internal(qdef):
    crit = qdef.get("criteria")
    if qdef["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": qdef["type"], "ins": qdef["instructions"], "crit": crit}


def encode(tok, cfg, state, qdef, target=None, label=-1):
    q = to_internal(qdef)
    ids, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
    k = len(render_options(q))
    if len(markers) != k:
        raise ValueError("options do not fit in head_max_len=%d" % cfg["head_max_len"])
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]],
            "target": target if target is not None else [0.0] * k,
            "label": label, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "kit"}


@torch.no_grad()
def predict_probs(model, tok, cfg, states, qdef, device, batch_size=32, use_amp=True,
                  progress=None):
    """-> np.ndarray [n, k] of probabilities over the question's options."""
    model.eval()
    dtype = amp_dtype(cfg.get("amp_dtype", "fp16"))
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
        dtype = torch.float16
    if device.type == "mps":
        dtype = torch.float16          # MPS autocast supports fp16/bf16 only
    out = []
    for start in range(0, len(states), batch_size):
        chunk = states[start:start + batch_size]
        items = [encode(tok, cfg, s, qdef) for s in chunk]
        b = collate_items([items], tok.pad_token_id)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp and device.type != "cpu"):
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                              b["marker_pos"].to(device), b["marker_mask"].to(device),
                              b["qtype"].to(device))
        p = torch.softmax(logits.float(), -1).cpu().numpy()
        if not np.isfinite(p).all():
            # fp16 autocast on MPS can blow up a whole batch, and a NaN row silently
            # argmaxes to label 0 -- which looks like a wrong prediction, not a bug.
            # Redo this batch in fp32 rather than let it through.
            with torch.autocast(device_type=device.type, enabled=False):
                logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                  b["marker_pos"].to(device), b["marker_mask"].to(device),
                                  b["qtype"].to(device))
            p = torch.softmax(logits.float(), -1).cpu().numpy()
            if not np.isfinite(p).all():
                raise RuntimeError("non-finite logits at batch offset %d even in fp32" % start)
        out.append(p[:, :len(items[0]["markers"])])
        if progress and (start // batch_size) % progress == 0:
            print("    %d/%d" % (min(start + batch_size, len(states)), len(states)), flush=True)
    return np.concatenate(out, 0)
