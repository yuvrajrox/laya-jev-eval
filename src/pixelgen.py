"""Ask Laya to choose a colour for every cell of a grid, then render it.

The user's proposal, tested literally. Each cell is one `choice` question over a small
palette; the cell's (row, col) goes into the state so the answers can differ by position.
Batched through layakit so 1,024 cells is one sweep rather than 1,024 round trips.
"""
import sys, time
import numpy as np
sys.path.insert(0, "src")
from layakit import load_checkpoint, predict_probs

PALETTE = {  # name -> terminal colour, for rendering
    "white": 231, "black": 16, "red": 196, "green": 46,
    "blue": 21, "yellow": 226, "brown": 130, "grey": 244,
}
NAMES = list(PALETTE)

def generate(prompt, N=32, model_dir="/Users/yuvraj/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/1c5edc17a7acd8701df6fc341c0d179f1c62c982/typed-decisions"):
    model, tok, cfg, dev = load_checkpoint(model_dir)
    q = {"type": "choice",
         "instructions": ("An image is being drawn cell by cell on a %dx%d grid. "
                          "The picture is: %s. What colour is this cell?" % (N, N, prompt)),
         "criteria": {c: None for c in NAMES}}
    states = [{"picture": prompt, "grid": "%dx%d" % (N, N), "row": r, "col": c,
               "position": "row %d of %d, column %d of %d" % (r, N, c, N)}
              for r in range(N) for c in range(N)]
    t0 = time.time()
    probs = predict_probs(model, tok, cfg, states, q, dev, batch_size=64)
    el = time.time() - t0
    idx = probs.argmax(1).reshape(N, N)
    return idx, probs, el

def render(idx):
    for row in idx:
        print("    " + "".join("\x1b[48;5;%dm  \x1b[0m" % PALETTE[NAMES[c]] for c in row))

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "a red apple on a white background"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    idx, probs, el = generate(prompt, N)
    print('\n  prompt: "%s"   %dx%d = %d cells\n' % (prompt, N, N, N * N))
    render(idx)
    uniq, counts = np.unique(idx, return_counts=True)
    print("\n  %.1fs total, %.1f ms/cell" % (el, 1000 * el / (N * N)))
    print("  distinct colours used: %d of %d" % (len(uniq), len(NAMES)))
    for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1]):
        print("     %-8s %4d cells (%.0f%%)" % (NAMES[u], c, 100 * c / (N * N)))
    print("  mean confidence: %.3f" % probs.max(1).mean())
