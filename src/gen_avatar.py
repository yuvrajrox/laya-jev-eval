"""Draw a 16x16 pet from a caption, by iterative mask-predict.

Start from a fully masked grid. Each round: predict every remaining cell at once, keep the
most confident fraction, re-mask the rest. A cosine schedule decides how many to keep per
round, so early rounds commit only to the cells the model is sure about (usually the
background and outline) and later rounds fill the detail those constrain.

Predictions are restricted to the 16 palette tokens, so the model cannot emit a word where
a pixel belongs -- the same trick that makes typed decisions unhallucinatable.
"""
import json, math, sys
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

LETTERS = "ABCDEFGHIJKLMNOP"


class Painter:
    def __init__(self, ckpt="checkpoints/avatar-mlm", device=None):
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.mdl = AutoModelForMaskedLM.from_pretrained(ckpt).eval()
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.mdl.to(self.dev)
        self.ids = [self.tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in LETTERS]
        self.ids[0] = self.tok(" A", add_special_tokens=False)["input_ids"][0]
        self.allow = torch.tensor(self.ids, device=self.dev)

    @torch.no_grad()
    def draw(self, caption, rounds=8, temperature=0.0):
        tok, dev = self.tok, self.dev
        text = caption + " | " + " ".join(["A"] * 256)
        enc = tok(text, return_tensors="pt").to(dev)
        seq = enc["input_ids"][0].clone()
        # cell positions = the last 256 non-special tokens
        cells = [i for i in range(len(seq)) if seq[i] in self.ids][-256:]
        for p in cells:
            seq[p] = tok.mask_token_id
        todo = set(cells)
        for r in range(rounds):
            logits = self.mdl(input_ids=seq.unsqueeze(0), attention_mask=enc["attention_mask"]).logits[0]
            sub = logits[:, self.allow]
            if temperature > 0:
                probs = torch.softmax(sub / temperature, -1)
            else:
                probs = torch.softmax(sub, -1)
            conf, choice = probs.max(-1)
            # cosine schedule: reveal progressively more each round
            frac = math.cos(math.pi / 2 * (r + 1) / rounds)
            n_keep_masked = int(len(cells) * frac)
            ranked = sorted(todo, key=lambda p: -float(conf[p]))
            reveal = ranked if r == rounds - 1 else ranked[:max(1, len(ranked) - n_keep_masked)]
            for p in reveal:
                if temperature > 0:
                    k = int(torch.multinomial(probs[p], 1))
                else:
                    k = int(choice[p])
                seq[p] = self.ids[k]
                todo.discard(p)
            if not todo:
                break
        inv = {v: LETTERS[i] for i, v in enumerate(self.ids)}
        return "".join(inv.get(int(seq[p]), "A") for p in cells)


if __name__ == "__main__":
    p = Painter(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/avatar-mlm")
    for c in ["a purple cat with yellow eyes on a mint background",
              "a blue dog with green eyes wearing a hat on a cream background"]:
        g = p.draw(c)
        print("\n%s" % c)
        for r in range(16):
            print("   " + g[r*16:(r+1)*16])
