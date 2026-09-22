"""Local demo server. Loads the models once and serves a page that drives them live.

Runs on the stdlib http.server so there is nothing to install. Two endpoints:

  POST /api/draw      {caption}      -> {frames: [...]}  one grid per mask-predict round,
                                        so the UI can animate the image resolving.
  POST /api/classify  {email}        -> decision + retrieved reason + generated reason

Models load lazily on first use, so the page comes up instantly and each demo only pays
for what it touches.
"""
import json, math, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "tasks"))
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

LETTERS = "ABCDEFGHIJKLMNOP"
_lock = threading.Lock()
_painter = None
_explainer = None


def painter():
    global _painter
    with _lock:
        if _painter is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            ck = os.path.join(ROOT, "checkpoints/avatar-mlm")
            tok = AutoTokenizer.from_pretrained(ck)
            mdl = AutoModelForMaskedLM.from_pretrained(ck).eval()
            dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            mdl.to(dev)
            ids = [tok(" " + c, add_special_tokens=False)["input_ids"][0] for c in LETTERS]
            _painter = (tok, mdl, dev, ids, torch.tensor(ids, device=dev))
    return _painter


def draw_frames(caption, rounds=12):
    import torch
    tok, mdl, dev, ids, allow = painter()
    text = caption + " | " + " ".join(["A"] * 256)
    enc = tok(text, return_tensors="pt").to(dev)
    seq = enc["input_ids"][0].clone()
    cells = [i for i in range(len(seq)) if int(seq[i]) in ids][-256:]
    for p in cells:
        seq[p] = tok.mask_token_id
    todo, frames = set(cells), []
    inv = {v: LETTERS[i] for i, v in enumerate(ids)}
    with torch.no_grad():
        for r in range(rounds):
            logits = mdl(input_ids=seq.unsqueeze(0), attention_mask=enc["attention_mask"]).logits[0]
            probs = torch.softmax(logits[:, allow], -1)
            conf, choice = probs.max(-1)
            frac = math.cos(math.pi / 2 * (r + 1) / rounds)
            ranked = sorted(todo, key=lambda p: -float(conf[p]))
            n = int(len(cells) * frac)
            reveal = ranked if r == rounds - 1 else ranked[:max(1, len(ranked) - n)]
            for p in reveal:
                seq[p] = ids[int(choice[p])]
                todo.discard(p)
            frames.append("".join(inv.get(int(seq[p]), ".") for p in cells))
            if not todo:
                break
    return frames


def explainer():
    global _explainer
    with _lock:
        if _explainer is None:
            from decide_explain import Explainer
            ck = os.path.join(ROOT, "runs/20260921-222527-email-intent/checkpoints/finetuned")
            _explainer = Explainer(ck if os.path.exists(os.path.join(ck, "model.safetensors")) else "typed-decisions")
    return _explainer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = "index.html" if self.path in ("/", "") else self.path.lstrip("/").split("?")[0]
        f = os.path.join(HERE, path)
        if not os.path.isfile(f):
            return self._send(404, "not found", "text/plain")
        ctype = {"html": "text/html", "js": "text/javascript", "css": "text/css"}.get(path.rsplit(".", 1)[-1], "text/plain")
        self._send(200, open(f, "rb").read(), ctype + "; charset=utf-8")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        t0 = time.time()
        try:
            if self.path == "/api/draw":
                frames = draw_frames(req.get("caption", "a purple cat with yellow eyes on a mint background"),
                                     int(req.get("rounds", 12)))
                return self._send(200, json.dumps({"frames": frames, "ms": int(1000 * (time.time() - t0))}))
            if self.path == "/api/classify":
                ex = explainer()
                body = req.get("email", "")
                intent, conf = ex.decide(body)
                retr, _ = ex.retrieve_reason(body)
                gen = ex.generate_reason(body, intent)
                return self._send(200, json.dumps({"intent": intent, "conf": round(conf, 3),
                                                   "retrieved": retr, "generated": gen,
                                                   "ms": int(1000 * (time.time() - t0))}))
        except Exception as e:                                   # noqa: BLE001
            import traceback; traceback.print_exc()
            return self._send(500, json.dumps({"error": str(e)}))
        self._send(404, json.dumps({"error": "no route"}))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("\n  demo running -> http://localhost:%d\n  (models load on first use)\n" % port)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
