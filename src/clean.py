"""Reply cleaning: pull the person's actual words out of a raw email body.

This matters more than it looks. The English Laya checkpoint has a 512 token budget
and roughly 320 tokens survive for the state after the option head. A raw reply is
usually 5% new text and 95% quoted thread, signature and legal disclaimer -- so an
uncleaned corpus feeds the model the *sender's own pitch* and truncates away the one
sentence that carries the intent. Cleaning is the single highest-leverage preprocessing
step in the pipeline.

Everything here is deliberately regex-and-rules. It runs on every reply at ingest time
and must never be the slow part.
"""
import re

# --------------------------------------------------------------------------- quoted thread
# Ordered: the earliest match in the body wins, since everything after it is history.
QUOTE_MARKERS = [
    r"^\s*On .{0,120}\bwrote:\s*$",                 # Gmail / Apple Mail
    r"^\s*On .{0,80}\bat\b.{0,60},.{0,80}\bwrote:", # Gmail, wrapped
    r"^\s*-{2,}\s*Original Message\s*-{2,}",        # Outlook
    r"^\s*_{5,}\s*$",                               # Outlook horizontal rule
    r"^\s*From:\s*.+$",                             # Outlook header block
    r"^\s*(>\s?){1,}",                              # plain quote prefix
    r"^\s*Sent from Mail for Windows",
    r"^\s*El .{0,120}\bescribió:\s*$",              # es
    r"^\s*Le .{0,120}\ba écrit\s*:\s*$",            # fr
    r"^\s*Am .{0,120}\bschrieb\b.{0,40}:\s*$",      # de
]
_QUOTE_RE = [re.compile(p, re.I | re.M) for p in QUOTE_MARKERS]

# --------------------------------------------------------------------------- signature
SIG_MARKERS = [
    r"^\s*--\s*$",                                  # RFC 3676 sig delimiter
    r"^\s*Sent from my (iPhone|iPad|Android|Samsung|mobile)",
    r"^\s*Get Outlook for (iOS|Android)",
    r"^\s*Best regards?,?\s*$",
    r"^\s*(Kind|Warm|Many) (regards|thanks),?\s*$",
    r"^\s*(Thanks|Cheers|Regards|Sincerely|Best),?\s*$",
]
_SIG_RE = [re.compile(p, re.I | re.M) for p in SIG_MARKERS]

# Legal boilerplate: if a line matches, drop it and everything after.
DISCLAIMER_RE = re.compile(
    r"^\s*(this (e-?mail|message)( and any attachments)? (is|are|may be) "
    r"(confidential|intended|privileged)"
    r"|the information (contained|transmitted) in this"
    r"|if you (are not|have received) (the|this) (intended|e-?mail|message) in error"
    r"|please consider the environment before printing"
    r"|disclaimer\s*:)", re.I | re.M)

UNSUB_FOOTER_RE = re.compile(
    r"^\s*(unsubscribe|opt[- ]out|manage (your )?(email )?preferences|"
    r"to stop receiving these emails)\b.{0,120}$", re.I | re.M)

# --------------------------------------------------------------------------- automated-mail signals
# These are RFC headers and subject conventions, not content. When present they are
# near-certain, so ingest.py can use them as rule labels for `ooo` / `bounce` and skip
# the teacher entirely -- which is how Apollo reached >99% precision on Out of Office.
AUTO_HEADERS = ["auto-submitted", "x-autoreply", "x-autorespond", "x-auto-response-suppress",
                "precedence: auto_reply", "x-failed-recipients"]

OOO_SUBJECT_RE = re.compile(
    r"\b(out of (the )?office|automatic reply|auto(matic)?[- ]?(reply|response)|"
    r"autoreply|away from (my )?(desk|office)|on (annual |parental |maternity )?leave|"
    r"abwesenheit|abwesend|réponse automatique|absence du bureau|respuesta automática|"
    r"fuera de la oficina|ooto|o\.o\.o)\b", re.I)

BOUNCE_SUBJECT_RE = re.compile(
    r"\b(undeliverable|undelivered mail|delivery (status notification|failure|has failed)|"
    r"returned mail|mail delivery (failed|subsystem)|failure notice|"
    r"address not found|recipient address rejected)\b", re.I)

BOUNCE_SENDER_RE = re.compile(
    r"^(mailer-daemon|postmaster|no-?reply|nobody)@", re.I)


def strip_html(text):
    """Crude but adequate: these are plaintext parts most of the time."""
    if "<" not in text:
        return text
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    for a, b in [("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&rsquo;", "'")]:
        text = text.replace(a, b)
    return text


def _earliest(text, regexes, min_offset=0):
    """Offset of the earliest match at or after min_offset, or None."""
    best = None
    for rx in regexes:
        m = rx.search(text, min_offset)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def split_thread(text):
    """Split a raw body into (newest reply, [older quoted turns])."""
    text = strip_html(text or "").replace("\r\n", "\n")
    cut = _earliest(text, _QUOTE_RE)
    if cut is None:
        return text.strip(), []
    head, tail = text[:cut], text[cut:]
    # Older turns, newest first in the raw text; reverse so `thread[-1]` is most recent.
    turns = [t.strip() for t in re.split(r"(?m)^\s*On .{0,120}\bwrote:\s*$", tail) if t.strip()]
    turns = [re.sub(r"(?m)^\s*>\s?", "", t)[:1500] for t in turns]
    return head.strip(), list(reversed(turns))[:5]


def _last_match(text, regexes):
    """Offset of the latest match across all regexes, or None."""
    best = None
    for rx in regexes:
        for m in rx.finditer(text):
            if best is None or m.start() > best:
                best = m.start()
    return best


def strip_signature(text, max_sig_chars=400, max_sig_lines=8):
    """Drop the sign-off, signature block and legal disclaimer.

    The sign-off patterns are anchored to a whole line, so "Thanks, but we are not
    interested" never matches and keeps the half of the sentence that carries the
    label. The remaining risk is a sign-off followed by more substance ("Best, -- and
    one more thing, can you send pricing?"), so a match is only treated as the start
    of a signature when what follows *looks* like one: short, and few lines.
    """
    if not text:
        return text
    # Footers are suffixes, so only strip when real text precedes them. Without this
    # guard a one-word "unsubscribe" reply -- the single most important thing this
    # classifier must never miss -- is cleaned away to an empty string and dropped.
    for rx in (DISCLAIMER_RE, UNSUB_FOOTER_RE):
        m = rx.search(text)
        if m and len(text[:m.start()].strip()) >= 40:
            text = text[:m.start()]
    cut = _last_match(text, _SIG_RE)
    if cut is not None:
        tail = text[cut:]
        if len(tail) <= max_sig_chars and tail.count("\n") <= max_sig_lines:
            text = text[:cut]
    return text.strip()


def normalise_whitespace(text):
    text = re.sub(r"[ \t]+", " ", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(body_raw, max_chars=4000):
    """Raw body -> (cleaned reply, older thread turns)."""
    reply, thread = split_thread(body_raw)
    reply = normalise_whitespace(strip_signature(reply))
    if not reply and thread:
        # Reply was pure quote -- e.g. a forward with no new text. Keep something.
        reply = normalise_whitespace(thread[-1])[:max_chars]
    return reply[:max_chars], [normalise_whitespace(t)[:1200] for t in thread]


def detect_automated(subject="", sender="", headers="", body=""):
    """Rule label for the two classes that rules genuinely solve.

    Returns "ooo", "bounce" or None. Deliberately conservative: it only fires on
    header / subject / envelope evidence, never on body phrasing, because a human
    writing "I'll be out next week, but yes let's talk" is `interested`, not `ooo`.
    """
    h = (headers or "").lower()
    blob = " ".join([subject or "", sender or ""])
    if BOUNCE_SENDER_RE.search(sender or "") or BOUNCE_SUBJECT_RE.search(blob) \
            or "x-failed-recipients" in h:
        return "bounce"
    if OOO_SUBJECT_RE.search(subject or "") or any(k in h for k in AUTO_HEADERS[:5]):
        return "ooo"
    return None


if __name__ == "__main__":
    sample = """Thanks for reaching out, but we just signed with a competitor last month.

Best,
Dana Whitfield
VP Revenue Operations | Northwind
m: +1 555 0142

On Tue, Sep 15, 2026 at 9:02 AM Alex Rivera <alex@vendor.com> wrote:
> Hi Dana, noticed Northwind is hiring 12 AEs this quarter...

This e-mail is confidential and intended solely for the addressee.
"""
    reply, thread = clean(sample)
    print("REPLY :", repr(reply))
    print("THREAD:", [t[:60] for t in thread])
    print("AUTO  :", detect_automated(subject="Automatic reply: quick question"))
    print("AUTO  :", detect_automated(sender="MAILER-DAEMON@mx.acme.com", subject="Undeliverable"))
