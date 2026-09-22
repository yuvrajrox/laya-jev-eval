# -*- coding: utf-8 -*-
"""Procedural 16x16 pixel-art pet avatars, with exact captions.

Why procedural rather than scraped: we need thousands of (caption, image) pairs where the
caption is guaranteed correct. Scraped pixel art comes with licensing questions and vague
alt-text; here we KNOW the sprite is a purple cat with yellow eyes because we drew it that
way. Perfect labels, unlimited volume, zero licence risk.

Encoding for training: the 16x16 grid flattens to 256 characters, one per cell, drawn from
a 16-colour palette mapped to A-P. 256 tokens fits inside ModernBERT's context, which is
the whole reason this is trainable at all -- every cell can attend to every other cell,
which is exactly what per-cell independent prediction could not do.

Sprites are templates with ROLE slots, recoloured per sample:
    . background   O outline   B body   S belly/snout   E eye   N nose   A accessory
"""
import json
import random

PALETTE = {  # name -> hex. Index order fixes the A-P letter mapping; do not reorder.
    "black": "#1a1a1a", "white": "#f5f5f5", "grey": "#9aa0a6", "red": "#d92b2b",
    "orange": "#e8760c", "yellow": "#f0c000", "lime": "#7cc000", "green": "#2f9e44",
    "mint": "#5fd0a8", "teal": "#159b9b", "blue": "#1c6fd4", "navy": "#26407a",
    "purple": "#8a4fd0", "pink": "#e86ab0", "brown": "#8a5a2b", "cream": "#f2dfc0",
}
NAMES = list(PALETTE)
LETTERS = "ABCDEFGHIJKLMNOP"
assert len(NAMES) == 16

SPRITES = {
"cat": [
"................","................","..OO........OO..","..OBO......OBO..",
"..OBBO....OBBO..","...OBBOOOOBBO...","...OBBBBBBBBO...","..OBBBBBBBBBBO..",
"..OBEBBBBBBEBO..","..OBBBBBBBBBBO..","..OBBBSNNSBBBO..","..OBBBSSSSBBBO..",
"...OBBBBBBBBO...","....OOOOOOOO....","................","................"],
"dog": [
"................","..OO........OO..","..OBBO....OBBO..","..OBBBO..OBBBO..",
"..OBBBBOOBBBBO..","...OBBBBBBBBO...","..OBBBBBBBBBBO..","..OBEBBBBBBEBO..",
"..OBBBBBBBBBBO..","..OBBBSSSSBBBO..","..OBBBSNNSBBBO..","...OBBSSSSBBO...",
"...OBBBBBBBBO...","....OOOOOOOO....","................","................"],
"bear": [
"................","...OO......OO...","..OBBO....OBBO..","..OBBBOOOOBBBO..",
"...OBBBBBBBBO...","..OBBBBBBBBBBO..","..OBBBBBBBBBBO..","..OBEBBBBBBEBO..",
"..OBBBBBBBBBBO..","..OBBBSSSSBBBO..","..OBBBSNNSBBBO..","..OBBBSSSSBBBO..",
"...OBBBBBBBBO...","....OOOOOOOO....","................","................"],
"bunny": [
"....OO....OO....","....OBO..OBO....","....OBO..OBO....","....OBO..OBO....",
"....OBBOOBBO....","...OBBBBBBBBO...","..OBBBBBBBBBBO..","..OBEBBBBBBEBO..",
"..OBBBBBBBBBBO..","..OBBBSNNSBBBO..","..OBBBSSSSBBBO..","...OBBBBBBBBO...",
"....OBBBBBBO....",".....OOOOOO.....","................","................"],
"frog": [
"................","................","...OO......OO...","..OBEO....OBEO..",
"..OBBBOOOOBBBO..","..OBBBBBBBBBBO..",".OBBBBBBBBBBBBO.",".OBBBBBBBBBBBBO.",
".OBBSSSSSSSSBBO.",".OBBSNNNNNNSBBO.","..OBSSSSSSSSBO..","..OBBBBBBBBBBO..",
"...OBBBBBBBBO...","....OOOOOOOO....","................","................"],
"owl": [
"................","...OO......OO...","..OBBOOOOOOBBO..","..OBBBBBBBBBBO..",
".OBBBBBBBBBBBBO.",".OBBSSBBBBSSBBO.",".OBSSESBBSESSBO.",".OBBSSBBBBSSBBO.",
".OBBBBBNNBBBBBO.",".OBBBBBBBBBBBBO.","..OBBBBBBBBBBO..","..OBBBBBBBBBBO..",
"...OBBBBBBBBO...","....OOOOOOOO....","................","................"],
"fox": [
"................","..OO........OO..","..OBO......OBO..","..OBBO....OBBO..",
"...OBBOOOOBBO...","..OBBBBBBBBBBO..","..OBEBBBBBBEBO..","..OBBBBBBBBBBO..",
"..OBBSSSSSSBBO..","...OBSSNNSSBO...","...OBSSSSSSBO...","....OBSSSSBO....",
".....OBBBBO.....","......OOOO......","................","................"],
"panda": [
"................","..OO........OO..","..OBO......OBO..","..OBBOOOOOOBBO..",
"...OBBBBBBBBO...","..OBBBBBBBBBBO..","..OBSSBBBBSSBO..","..OBSESBBSESBO..",
"..OBBSSBBBBSSO..","..OBBBBNNBBBBO..","..OBBBBBBBBBBO..","...OBBBBBBBBO...",
"....OBBBBBBO....",".....OOOOOO.....","................","................"],
}

ACCESSORIES = {
    "": None,
    " wearing a hat": [(1, c, "A") for c in range(5, 11)] + [(2, c, "A") for c in range(6, 10)],
    " wearing a bow": [(0, 6, "A"), (0, 7, "A"), (0, 8, "A"), (0, 9, "A"), (1, 7, "A"), (1, 8, "A")],
    " wearing a scarf": [(12, c, "A") for c in range(4, 12)] + [(13, c, "A") for c in range(5, 11)],
}

# Body colours a pet can plausibly be; keeps captions sensible.
BODY = ["white", "grey", "orange", "yellow", "brown", "cream", "pink", "purple",
        "blue", "mint", "teal", "green", "red", "navy", "lime", "black"]
EYE = ["black", "green", "blue", "teal", "purple", "red", "orange", "yellow"]
BG = ["white", "cream", "mint", "grey", "pink", "teal", "lime", "yellow", "blue", "purple"]
ACC_COLOR = ["red", "blue", "green", "purple", "pink", "orange", "yellow", "navy", "teal", "black"]


def render(species, body, eye, bg, acc="", acc_color="red", outline="black", belly="white"):
    grid = [list(row) for row in SPRITES[species]]
    if acc and ACCESSORIES[acc]:
        for r, c, ch in ACCESSORIES[acc]:
            if 0 <= r < 16 and 0 <= c < 16:
                grid[r][c] = ch
    role = {".": bg, "O": outline, "B": body, "S": belly, "E": eye, "N": outline, "A": acc_color}
    return [[role[ch] for ch in row] for row in grid]


def encode(colour_grid):
    idx = {n: LETTERS[i] for i, n in enumerate(NAMES)}
    return "".join(idx[c] for row in colour_grid for c in row)


def caption(species, body, eye, bg, acc):
    return "a %s %s with %s eyes%s on a %s background" % (body, species, eye, acc, bg)


def sample(rng):
    species = rng.choice(list(SPRITES))
    body = rng.choice(BODY)
    eye = rng.choice([e for e in EYE if e != body])
    bg = rng.choice([b for b in BG if b != body])
    acc = rng.choice(list(ACCESSORIES))
    acc_color = rng.choice([a for a in ACC_COLOR if a != body]) if acc else "red"
    belly = "white" if body not in ("white", "cream") else "cream"
    g = render(species, body, eye, bg, acc, acc_color, "black", belly)
    return {"caption": caption(species, body, eye, bg, acc), "grid": encode(g),
            "species": species, "body": body, "eye": eye, "bg": bg, "accessory": acc.strip() or "none"}


if __name__ == "__main__":
    import collections, os, sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    rng = random.Random(7)
    seen, rows = set(), []
    while len(rows) < n:
        s = sample(rng)
        k = (s["caption"],)
        if k in seen:
            continue
        seen.add(k)
        rows.append(s)
    out = "data/avatars/train.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d unique avatars -> %s" % (len(rows), out))
    print("  grid length: %d chars (= tokens for the model)" % len(rows[0]["grid"]))
    print("  species: %s" % dict(collections.Counter(r["species"] for r in rows)))
    print("  theoretical combos: %d" % (len(SPRITES) * len(BODY) * len(EYE) * len(BG) * len(ACCESSORIES)))
    print("\n  sample caption: %s" % rows[0]["caption"])
