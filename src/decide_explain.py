"""Decision + reason on the 100-email holdout.

Two outputs per email:
  decision  -- fine-tuned Laya, the intent label (we have ground truth for this)
  reason    -- two competing methods:
                 (a) generated:  ModernBERT mask-predict, novel text
                 (b) retrieved:  one Laya choice question over 12 pre-written sentences

There is no ground truth for the reasons, so scoring them needs a proxy. The one used
here is FAITHFULNESS by round-trip: feed the reason back into the classifier on its own,
and check it yields the same intent as the decision. If an explanation cannot reproduce
the decision it is explaining, it is decoration rather than a reason. It is not a
measure of whether the sentence is TRUE of the email -- that still needs a human or an
LLM judge -- so it is reported as what it is.
"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tasks"))
import laya, email_intent as TASK
from transformers import AutoTokenizer, AutoModelForMaskedLM

REASON_BANK = {
 "repeat_contact": "the customer has contacted us about this more than once",
 "broken_product": "a product or service arrived faulty, damaged or wrong",
 "slow_response":  "we failed to reply or act within the time we promised",
 "billing_error":  "the customer was charged incorrectly or unexpectedly",
 "rude_service":   "the customer was treated poorly by a member of staff",
 "wants_action":   "the customer is asking us to change or do something specific",
 "wants_document": "the customer is asking us to send a document or record",
 "wants_info":     "the customer is asking a question and wants an answer",
 "wants_policy":   "the customer is asking about our terms, policy or availability",
 "praise_staff":   "the customer is praising a specific person or team",
 "praise_product": "the customer is happy with the product or service itself",
 "suggestion":     "the customer likes it but is suggesting an improvement",
}

class Explainer:
    def __init__(self, ckpt, device=None):
        self.agent = laya.load(ckpt) if os.path.isdir(ckpt) else laya.load("convaiinnovations/laya", subfolder=ckpt)
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        self.mlm = AutoModelForMaskedLM.from_pretrained("answerdotai/ModernBERT-large").eval().to(self.dev)

    def decide(self, body):
        r = self.agent.predict({"body": body}, TASK.QUESTIONS)["answers"]["intent"]
        return r["choice"], max(r["probabilities"].values())

    def retrieve_reason(self, body):
        q = {"reason": {"type": "choice",
                        "instructions": "Which single statement best explains this customer email?",
                        "criteria": REASON_BANK}}
        r = self.agent.predict({"body": body}, q)["answers"]["reason"]
        return REASON_BANK[r["choice"]], max(r["probabilities"].values())

    @torch.no_grad()
    def generate_reason(self, body, intent, n=8, rounds=3):
        """Mask-predict. Conditioned on the decision, which grounds it."""
        M, tok = self.tok.mask_token, self.tok
        prompt = ('Customer email: "%s"\nThis email is a %s.\nIn one sentence: the customer'
                  % (body.replace("\n", " ")[:300], intent))
        enc = tok(prompt + " " + " ".join([M] * n) + ".", return_tensors="pt").to(self.dev)
        seq = enc["input_ids"][0].clone()
        mask_pos = (seq == tok.mask_token_id).nonzero().flatten().tolist()
        frozen = {}
        for r in range(rounds):
            logits = self.mlm(input_ids=seq.unsqueeze(0), attention_mask=enc["attention_mask"]).logits[0]
            p = torch.softmax(logits, -1)
            cand = sorted(((float(p[i].max()), i, int(p[i].argmax())) for i in mask_pos if i not in frozen), reverse=True)
            if not cand: break
            keep = len(cand) if r == rounds - 1 else max(1, len(cand) // 2)
            for c, i, t in cand[:keep]:
                seq[i] = t; frozen[i] = c
            for i in mask_pos:
                if i not in frozen: seq[i] = tok.mask_token_id
        words = tok.decode([seq[i] for i in sorted(mask_pos)], skip_special_tokens=True)
        return ("the customer " + words.strip()).strip()

    def classify_text(self, text):
        return self.agent.predict({"body": text}, TASK.QUESTIONS)["answers"]["intent"]["choice"]


def main():
    run = sys.argv[1]
    recs = [json.loads(l) for l in open(os.path.join(run, "data/04_holdout100.jsonl"))]
    ck = os.path.join(run, "checkpoints/finetuned")
    ex = Explainer(ck if os.path.exists(os.path.join(ck, "model.safetensors")) else "typed-decisions")

    out, t0 = [], time.time()
    for i, r in enumerate(recs):
        intent, conf = ex.decide(r["body"])
        retr, rconf = ex.retrieve_reason(r["body"])
        gen = ex.generate_reason(r["body"], intent)
        out.append({"id": r["id"], "body": r["body"], "label": r["label"], "pred": intent,
                    "conf": conf, "reason_retrieved": retr, "reason_generated": gen})
        if (i + 1) % 25 == 0: print("  %d/100" % (i + 1), flush=True)
    elapsed = time.time() - t0

    # faithfulness: does the reason alone reproduce the decision?
    for o in out:
        o["faithful_retrieved"] = ex.classify_text(o["reason_retrieved"]) == o["pred"]
        o["faithful_generated"] = ex.classify_text(o["reason_generated"]) == o["pred"]

    p = os.path.join(run, "reports/60_decide_explain.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")

    acc = sum(o["pred"] == o["label"] for o in out) / len(out)
    fr = sum(o["faithful_retrieved"] for o in out) / len(out)
    fg = sum(o["faithful_generated"] for o in out) / len(out)
    print("\n" + "=" * 70)
    print("  decision accuracy            %.3f  (%d/100)" % (acc, int(acc * 100)))
    print("  reason faithful — retrieved  %.3f" % fr)
    print("  reason faithful — generated  %.3f" % fg)
    print("  %.0f ms per email (decision + both reasons)" % (1000 * elapsed / len(recs)))
    print("=" * 70)
    print("\n  SAMPLES:\n")
    for o in out[:3] + out[25:27] + out[50:52] + out[75:77]:
        print("  %s" % o["body"].replace("\n", " ")[:88])
        print("    decision : %s (%.2f)%s" % (o["pred"], o["conf"], "" if o["pred"] == o["label"] else "   <- WRONG, true=%s" % o["label"]))
        print("    retrieved: %s" % o["reason_retrieved"])
        print("    generated: %s\n" % o["reason_generated"])
    print("  -> %s" % p)


if __name__ == "__main__":
    main()
