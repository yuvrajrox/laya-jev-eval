"""Task spec for the supplied email_intent.csv (4 generic intent labels).

This is NOT the Rox reply-intent task. It is a generic customer-email intent set --
request / inquiry / complaint / feedback -- so it measures how well Laya learns a
4-way text classification from ~1,400 short examples. That transfers as evidence about
the *method*, not about reply classification specifically.
"""

LABELS = ["request", "inquiry", "complaint", "feedback"]

CRITERIA = {
    "request":   "asks the company to perform an action or change something",
    "inquiry":   "asks for information or the status of something",
    "complaint": "expresses dissatisfaction about a problem or poor service",
    "feedback":  "gives praise, thanks, or an opinion without asking for anything",
}

INSTRUCTIONS = "Classify the intent of this customer email."

QUESTIONS = {
    "intent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA},
}


def state_for(rec):
    """The supplied CSV has no subject line, so the state is the body alone."""
    return {"body": rec["body"]}
