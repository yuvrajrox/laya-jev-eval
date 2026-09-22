"""Word-by-word generation by repeated Laya choice calls. An honest test of whether a
decision model can be made to write a sentence.

Key correction to the earlier estimate: you do not binary-search 50,000 English words.
A reasoning sentence about email triage uses a few dozen. If the candidate set fits in
ONE choice question, that is 1 call per word, not 16 -- which changes the arithmetic
completely. This measures whether the output is actually coherent at that cost.
"""
import sys, time
import laya

VOCAB = [
    # subjects / determiners
    "the", "a", "this", "customer", "sender", "message", "email", "person", "they",
    # verbs
    "is", "was", "has", "wants", "asks", "reports", "complains", "praises", "requests",
    "received", "needs", "says", "contacted", "thanks",
    # objects / nouns
    "refund", "invoice", "delivery", "order", "account", "staff", "service", "product",
    "problem", "issue", "delay", "charge", "reply", "information", "status", "appointment",
    "us", "help", "action", "answer", "team", "time",
    # modifiers
    "angry", "happy", "frustrated", "unhappy", "satisfied", "urgent", "twice", "again",
    "not", "very", "still", "already", "faulty", "wrong", "late", "missing", "good",
    # connectives / prepositions
    "and", "but", "so", "because", "about", "for", "with", "to", "of", "an", "their", "no",
    "<END>",
]

def generate(agent, email, max_words=14, vocab=VOCAB, verbose=True):
    words, timings = [], []
    for step in range(max_words):
        so_far = " ".join(words) if words else "(nothing yet)"
        q = {"next": {
            "type": "choice",
            "instructions": ("A one-sentence summary of the customer email is being written "
                             "one word at a time. Sentence so far: '%s'. Which word comes next? "
                             "Choose <END> if the sentence is complete." % so_far),
            "criteria": {w: None for w in vocab},
        }}
        t0 = time.time()
        r = agent.predict({"email": email, "sentence_so_far": so_far}, q)["answers"]["next"]
        timings.append(time.time() - t0)
        w = r["choice"]
        if w == "<END>":
            break
        words.append(w)
        if verbose:
            print("    step %2d  %-14s p=%.2f  (%.0f ms)" % (step + 1, w, max(r["probabilities"].values()), timings[-1] * 1000))
    return " ".join(words), timings


if __name__ == "__main__":
    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    print("vocab: %d options | head_max_len=%d -> %d tokens per option\n"
          % (len(VOCAB), agent.cfg["head_max_len"], (agent.cfg["head_max_len"] - 16) // len(VOCAB)))
    emails = [
        "I was charged twice for the same booking and nobody has come back to me in eight days.",
        "Just wanted to say the engineer who came out yesterday, Danny, was excellent.",
    ]
    for e in emails:
        print("EMAIL: %s" % e[:78])
        sent, t = generate(agent, e)
        print("  -> %r" % sent)
        print("  %d words, %.0f ms total, %.0f ms/word\n" % (len(t), sum(t) * 1000, 1000 * sum(t) / len(t)))
