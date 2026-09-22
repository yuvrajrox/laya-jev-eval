# -*- coding: utf-8 -*-
"""Hand-written seed replies for the cold-outbound reply-intent task.

Written by hand rather than sampled, for one reason: a random sample of real replies
is ~70% out-of-office and bounce, so it contains almost no unsubscribes, referrals or
hostile replies -- the classes that matter most and that you cannot measure without
examples. These fill those cells and, deliberately, the hard cells between them.

`hard=True` marks a case where reasonable annotators disagree. Keep the hard cases
over-represented here and track accuracy on them separately: the easy cases are solved
by any model, and the gap between Laya and Claude lives entirely in the hard ones.

Fields: subject, body, intent, meeting_booked, is_human, hostility (0-2),
revisit_later, hard, lang.
"""

def R(subject, body, intent, meeting_booked=False, is_human=True, hostility=0,
      revisit_later=False, hard=False, lang="en"):
    return dict(subject=subject, body=body, intent=intent, meeting_booked=meeting_booked,
                is_human=is_human, hostility=hostility, revisit_later=revisit_later,
                hard=hard, lang=lang)

SEED = [
    # ------------------------------------------------------------------ interested
    R("Re: quick question", "Sure, happy to chat. Does Thursday at 2pm ET work?", "interested", meeting_booked=True),
    R("Re: Northwind + inventory forecasting", "This is timely actually. Send over pricing and I'll loop in our CFO.", "interested"),
    R("Re: 12 open AE roles", "Interesting. What does implementation look like for a team our size?", "interested"),
    R("Re: quick question", "yes", "interested", hard=True),
    R("Re: saw your Series B", "Go ahead and send a calendar invite for next week, mornings are better for me.", "interested", meeting_booked=True),
    R("Re: idea for Q4", "I'd like to understand the pricing model before we go further, but broadly yes this is relevant.", "interested"),
    R("Re: intro", "Can you share a deck? If it looks right I'll set something up with my team.", "interested"),
    R("Re: forecasting accuracy", "We're actively evaluating vendors in this space right now. Good timing.", "interested"),
    R("Re: quick question", "Sounds good, let's talk. I'm in London so afternoons your time.", "interested"),
    R("Re: hi from Vendor", "Sure -- but I should say up front our budget for this is very small.", "interested", hard=True),
    R("Re: 15 minutes?", "ok", "interested", hard=True),
    R("Re: pipeline coverage", "Book it. https://calendly.com/d-whitfield/30min", "interested", meeting_booked=True),
    R("Re: quick question", "Yeah I'll bite. What makes you different from the four other vendors who emailed me this week?", "interested", hostility=1, hard=True),
    R("Re: RevOps at Acme", "Let me know what times you have open Tuesday or Wednesday.", "interested", meeting_booked=True),
    R("Re: following up", "Yes please. We tried to build this internally last year and it went badly.", "interested"),

    # ------------------------------------------------------------------ not_interested
    R("Re: quick question", "Thanks for reaching out, but we just signed with a competitor last month.", "not_interested"),
    R("Re: intro", "Not a fit for us, we're a services business and don't run outbound.", "not_interested"),
    R("Re: forecasting", "No thanks.", "not_interested"),
    R("Re: quick question", "We're in a hiring freeze and everything non-essential is frozen with it. Try us in Q3.", "not_interested", revisit_later=True, hard=True),
    R("Re: hi", "Already have a solution for this, and we're happy with it.", "not_interested"),
    R("Re: 15 minutes?", "I'm not the right person and I don't think anyone here is. Good luck.", "not_interested", hard=True),
    R("Re: Acme + Vendor", "Budget's gone for the year. Circle back in January if you want.", "not_interested", revisit_later=True, hard=True),
    R("Re: quick question", "We evaluated you two years ago and passed. Nothing's changed on our side.", "not_interested"),
    R("Re: revenue ops", "pass", "not_interested"),
    R("Re: intro", "Appreciate the note but this isn't a priority for us right now.", "not_interested"),
    R("Re: quick question", "We're too small for this. Maybe in a couple of years.", "not_interested", revisit_later=True, hard=True),
    R("Re: demo?", "Thanks, but we build everything in house.", "not_interested"),
    R("Re: Q4 planning", "Not now. I'll reach out if that changes.", "not_interested", revisit_later=True),
    R("Re: hello", "This is not something we would ever buy.", "not_interested"),
    R("Re: quick question", "We're being acquired so nothing is getting bought here for at least six months.", "not_interested", revisit_later=True, hard=True),

    # ------------------------------------------------------------------ ooo
    R("Automatic reply: quick question",
      "I am out of the office until 29 September with limited access to email. For urgent matters please contact Priya Raman (priya@acme.com).",
      "ooo", is_human=False, hard=True),
    R("Out of Office: intro", "On annual leave, back Monday. I will reply then.", "ooo", is_human=False),
    R("Automatic reply: forecasting",
      "Thank you for your message. I am currently on parental leave until March 2027. Please contact the RevOps team at revops@acme.com.",
      "ooo", is_human=False, hard=True),
    R("Re: quick question", "I'm travelling this week with patchy signal -- I'll come back to you properly next Monday.", "ooo", hard=True),
    R("Automatic reply: hi", "I'm away from my desk and will respond on my return.", "ooo", is_human=False),
    R("AUTOMATIC REPLY", "OOO until 10/03. Not monitoring email.", "ooo", is_human=False),
    R("Re: intro", "Heads up, I'm out until the 14th but yes, interested -- let's set something up when I'm back.", "interested", is_human=True, hard=True),
    R("Automatische Antwort: kurze Frage", "Ich bin bis zum 5. Oktober nicht im Büro und habe keinen Zugriff auf meine E-Mails.", "ooo", is_human=False, lang="de"),
    R("Réponse automatique : question rapide", "Je suis absent du bureau jusqu'au 12 octobre. Je répondrai à mon retour.", "ooo", is_human=False, lang="fr"),
    R("Respuesta automática: hola", "Estoy fuera de la oficina hasta el lunes. Para asuntos urgentes, contacte a soporte@acme.com.", "ooo", is_human=False, lang="es"),
    R("Automatic reply: quick question", "自動返信：10月5日まで不在にしております。", "ooo", is_human=False, lang="ja"),
    R("Out of office", "Maternity leave. Back in six months. Please remove me from any active threads.", "ooo", is_human=False, hard=True),
    R("Automatic reply", "I have left Acme. Please direct enquiries to hello@acme.com.", "referral", is_human=False, hard=True),
    R("Auto: Re: pipeline", "Working reduced hours Thursday and Friday. Responses may be slow.", "ooo", is_human=False),
    R("Automatic reply: intro", "Out sick. Will review when back.", "ooo", is_human=False),

    # ------------------------------------------------------------------ unsubscribe
    R("Re: quick question", "Remove me from your list.", "unsubscribe"),
    R("Re: intro", "unsubscribe", "unsubscribe"),
    R("Re: forecasting", "Please take me off this list and do not contact me again.", "unsubscribe"),
    R("Re: hi", "STOP EMAILING ME", "unsubscribe", hostility=2),
    R("Re: quick question", "Not interested, and please remove me from your database.", "unsubscribe", hard=True),
    R("Re: Acme", "I did not opt in to this. Delete my data under GDPR Article 17 and confirm in writing.", "unsubscribe", hostility=2),
    R("Re: 15 minutes?", "opt out", "unsubscribe"),
    R("Re: following up", "This is the third email. Stop.", "unsubscribe", hostility=2),
    R("Re: hello", "Please unsubscribe me from all future communications. Thank you.", "unsubscribe"),
    R("Re: quick question", "Take me off. Also how did you get my email?", "unsubscribe", hard=True),
    R("Re: intro", "Do not contact me or anyone at this company again. We will report this as spam.", "unsubscribe", hostility=2),
    R("Re: Q4", "remove", "unsubscribe"),
    R("Re: pipeline coverage", "Kindly remove my address from your marketing lists.", "unsubscribe"),
    R("Re: hi", "No. Unsubscribe.", "unsubscribe"),
    R("Re: quick question", "I've marked this as spam and I'm unsubscribing. Please respect that.", "unsubscribe", hostility=1),

    # ------------------------------------------------------------------ referral
    R("Re: quick question", "I don't own this any more -- Sam Okoye does. Copying him here.", "referral"),
    R("Re: intro", "You want our RevOps lead, not me. Try priya@acme.com.", "referral"),
    R("Re: forecasting", "Wrong person. Procurement handles vendor evaluations.", "referral", hard=True),
    R("Re: Acme + Vendor", "I've left Northwind. My replacement is Dana Whitfield.", "referral"),
    R("Re: hi", "Forwarding to our Head of Sales Ops who'd be the right contact.", "referral"),
    R("Re: 15 minutes?", "Adding Marcus who runs this area. I'll step back.", "referral"),
    R("Re: quick question", "Not my remit but I'll pass it on internally.", "referral", hard=True),
    R("Re: pipeline", "Please direct this to legal@acme.com, all vendor outreach goes through them.", "referral"),
    R("Re: intro", "I moved to the product org last quarter. Ping Elena on the GTM side.", "referral"),
    R("Re: Q4 planning", "cc'ing Tom, he owns the tooling budget.", "referral"),
    R("Re: hello", "That'd be a question for our CRO. Happy to intro if it's genuinely relevant.", "referral", hard=True),
    R("Re: quick question", "Try our IT team, they gatekeep all software purchases here.", "referral"),
    R("Re: demo", "I'm an IC, not a buyer. My manager is copied.", "referral"),
    R("Re: revenue", "Sending this over to the team that would actually use it.", "referral"),
    R("Re: intro", "Wrong Dana. You want Dana Whitfield at Northwind, I'm at Southgate.", "other", hard=True),

    # ------------------------------------------------------------------ bounce
    R("Undeliverable: quick question",
      "Your message to dana.whitfield@northwind.com couldn't be delivered. The email address you entered couldn't be found. 550 5.1.1 User unknown",
      "bounce", is_human=False),
    R("Mail delivery failed: returning message to sender",
      "This is the mail system at host mx.acme.com. I'm sorry to have to inform you that your message could not be delivered to one or more recipients. 550 5.1.1 <sam@acme.com>: Recipient address rejected: User unknown in virtual mailbox table",
      "bounce", is_human=False),
    R("Delivery Status Notification (Failure)",
      "Address not found. Your message wasn't delivered to priya@acme.com because the address couldn't be found, or is unable to receive mail.",
      "bounce", is_human=False),
    R("Delivery Status Notification (Delay)",
      "This is an automatically generated Delivery Status Notification. THIS IS A WARNING MESSAGE ONLY. Your message has not been delivered yet. Delivery will be retried for 24 hours.",
      "bounce", is_human=False, hard=True),
    R("Undelivered Mail Returned to Sender",
      "The mail system: host mail.northwind.com said: 552 5.2.2 Mailbox full (in reply to RCPT TO command)",
      "bounce", is_human=False),
    R("failure notice",
      "Hi. This is the qmail-send program at acme.com. I'm afraid I wasn't able to deliver your message. Sorry it didn't work out. Permanent failure.",
      "bounce", is_human=False),
    R("Undeliverable: intro",
      "Your message wasn't delivered because the recipient's email provider rejected it. 554 5.7.1 Message rejected due to content restrictions",
      "bounce", is_human=False, hard=True),
    R("Automatic reply: quick question",
      "dana.whitfield@northwind.com is no longer with the company. This mailbox is not monitored.",
      "bounce", is_human=False, hard=True),
    R("Returned mail: see transcript for details",
      "The original message was received at Tue, 15 Sep 2026 09:02:11. ----- The following addresses had permanent fatal errors -----",
      "bounce", is_human=False),
    R("Delivery has failed to these recipients or groups",
      "The recipient's mailbox is full and can't accept messages now. Microsoft Exchange Server 2019.",
      "bounce", is_human=False),

    # ------------------------------------------------------------------ other
    R("Re: quick question", "Who is this?", "other"),
    R("Re: intro", "How did you get my email address?", "other", hostility=1, hard=True),
    R("Re: hi", "?", "other"),
    R("Re: forecasting", "Is this a real company or is this AI generated?", "other", hostility=1),
    R("Re: Acme", "Thanks!", "other", hard=True),
    R("Re: quick question", "Received.", "other"),
    R("Re: intro", "Hi -- are you the same vendor we spoke to in 2023? The name is familiar.", "other", hard=True),
    R("Re: 15 minutes?", "sorry, sent that to the wrong thread", "other"),
    R("Re: hello", "Do you have a partner programme? We're a consultancy, not a buyer.", "other", hard=True),
    R("Re: pipeline coverage", "Please stop using my company logo in your emails without permission.", "other", hostility=2, hard=True),
    R("Re: quick question", "Noted.", "other"),
    R("Re: demo", "What's your data retention policy? Asking before I engage at all.", "other", hard=True),
    R("Re: intro", "I'll forward this to my spam folder where it belongs.", "not_interested", hostility=2, hard=True),
    R("Re: Q4", "Can you resend? The link in your email is broken.", "other", hard=True),
    R("Re: hi", "unsubscribe from this thread only please, keep me on the newsletter", "unsubscribe", hard=True),
]

if __name__ == "__main__":
    import collections, csv, sys, os
    counts = collections.Counter(r["intent"] for r in SEED)
    out = os.path.join(os.path.dirname(__file__), "seed_replies.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(SEED[0].keys()))
        w.writeheader()
        w.writerows(SEED)
    print("%d rows -> %s" % (len(SEED), out))
    for k, v in counts.most_common():
        print("  %-16s %3d" % (k, v))
    print("  hard cases: %d" % sum(1 for r in SEED if r["hard"]))
    print("  non-English: %d" % sum(1 for r in SEED if r["lang"] != "en"))
