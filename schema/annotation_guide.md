# Annotation guide — reply intent

For the ~300 eval rows you label by hand. Read it once before starting, and keep it
open. Consistency matters more than any individual call: a coherently-wrong rule still
produces a learnable model, while an incoherent one puts a ceiling on everything.

## The seven labels

| label | it means |
|---|---|
| `interested` | wants to continue: asks for a call, pricing, a demo, more detail, or says yes |
| `not_interested` | declines: not a fit, no budget, already solved, stop pitching, or not right now |
| `ooo` | automatic out-of-office, holiday, parental leave, vacation autoresponder |
| `unsubscribe` | asks to be removed from the list or never contacted again |
| `referral` | redirects to a different person or team as the right contact |
| `bounce` | delivery failure from a mail server, not written by a person |
| `other` | none of the above: unclear, identity questions, unrelated |

Pick exactly one. When torn, apply the rules below; if still torn, mark it `hard=True`
and pick the better of the two. The `hard` flag is not a cop-out — hard-case accuracy is
reported separately and is where the gap between Laya and Claude actually lives.

## The rules that decide the ambiguous cases

**1. Removal beats refusal.** "Not interested, please remove me" is `unsubscribe`, not
`not_interested`. The removal request is the one with legal consequences; missing it
costs far more than mislabelling a decline.

**2. "Not right now" is `not_interested`.** "Try us in Q3", "budget's gone", "we're in a
hiring freeze" — all `not_interested`, with `revisit_later=True`. It is tempting to call
these `interested` because they are warm, but they must not trigger the interested
workflow today. The warmth is captured by `revisit_later`, which is what that field is for.

**3. The human wins over the wrapper.** An autoresponder that also contains a real human
reply — "I'm out until the 14th but yes, let's talk when I'm back" — is `interested`,
`is_human=True`. The automated envelope is not the intent.

**4. Left-the-company messages split by whether a successor is named.** "I've left Acme,
contact hello@acme.com" is `referral`. "This mailbox is no longer monitored" with no
forward is `bounce`. If it names a person or a team, it is a referral.

**5. `is_human` is about authorship, not content.** A bounce and an OOO are both
`is_human=False` regardless of how conversational they read.

**6. Acknowledgements are `other`.** "Thanks!", "Noted.", "Received." — these are not
interest. Only mark `interested` when there is a forward-looking ask.

**7. "How did you get my email?" is `other`** unless removal is also requested, in which
case `unsubscribe`. Set `hostility` to 1 or 2 as the tone warrants.

**8. Delay warnings are `bounce`.** "Your message has not been delivered yet, delivery
will be retried" is still a delivery failure signal. Real, and genuinely debatable —
this one is marked `hard`.

**9. Mistaken-identity replies are `other`, not `referral`.** "Wrong Dana, you want the
one at Northwind" is not a referral to the right contact for the pitch; it is a data
quality signal.

## The co-questions

- `meeting_booked` — **true only for a specific time or a booking link.** "Let's chat
  sometime" is false. "Does Thursday at 2pm work?" is true. A calendly link is true.
- `is_human` — see rule 5.
- `revisit_later` — true when the reply asks to be contacted again at a later date, even
  vaguely ("circle back in January", "maybe in a couple of years").
- `hostility` — `0` neutral or polite · `1` annoyed, curt, irritated · `2` angry, abusive,
  or threatening legal / compliance action. Curtness alone is not hostility; "pass" is 0.

## Working method

Label in one sitting per class-block if you can, and **do not look at any model output
while labelling.** Once you have seen a prediction you cannot unsee it, and your eval
set quietly becomes a measure of agreement with the model rather than with reality.

If you disagree with a rule above, change the rule *and re-label everything it touched*.
Do not make a silent exception — that is how a corpus becomes unlearnable.
