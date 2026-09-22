# -*- coding: utf-8 -*-
"""100 hand-written customer emails — an honest generalisation test.

Why this exists: the supplied email_intent.csv is template-generated, so 39% of its
test rows were >=0.8 token-similar to a training row and the fine-tuned model scored
1.0000 on it. That number measures pattern recall, not generalisation. This set is
written from scratch to break every one of those patterns.

Deliberate differences from the training distribution:
  - natural, grammatical English (the source data is often not: "please correction my
    prescription immediately")
  - length from 4 to ~70 words, against a source median of 9
  - greetings, sign-offs, order numbers, multi-sentence bodies, typos, lowercase,
    non-native phrasing
  - 12 industries, and vocabulary chosen to avoid the source's noun list where possible
  - mixed-signal emails, which the source almost never has

LABELLING CONVENTION — inferred from the source data, not invented:
  request    an ask for an ACTION, including "send me <document>"
  inquiry    an ask for INFORMATION, including "update me on the status of X"
  complaint  dissatisfaction present -> complaint WINS over any co-occurring ask
  feedback   praise or opinion with nothing asked for

`hard=True` marks cases where the boundary is genuinely arguable. Scoring is reported
separately for those, because a model losing only on hard cases is a different finding
from one losing everywhere.
"""


def E(body, label, hard=False, domain=""):
    return dict(body=body, label=label, hard=hard, domain=domain)


HOLDOUT = [
    # ---------------------------------------------------------------- request (25)
    E("Could you please close my savings account? I've moved banks and no longer need it.", "request", domain="banking"),
    E("hi - need to add my wife as an authorised driver on the policy before friday. she's had her licence 11 years, clean. what do you need from me", "request", domain="insurance"),
    E("Please cancel the auto-renewal on order #88213. I'd rather re-subscribe manually next year.", "request", domain="retail"),
    E("Dear Support Team,\n\nFollowing our call this morning, I would be grateful if you could reset the two-factor authentication on my account. I no longer have access to the old handset.\n\nKind regards,\nMargaret Ellison", "request", domain="saas"),
    E("Move my appointment from the 14th to any morning the following week please.", "request", domain="healthcare"),
    E("Can you send across the VAT invoice for last month? Our accountant is chasing.", "request", domain="finance"),
    E("Swap the seats to aisle if there's anything left, two passengers, booking QJ4471.", "request", domain="travel"),
    E("I'd like to downgrade from the Pro tier to Starter at the end of the billing cycle. Not unhappy with it, we just aren't using the extra seats.", "request", hard=True, domain="saas"),
    E("please remove my old address from the account, it keeps autofilling at checkout", "request", domain="retail"),
    E("Would you be able to raise the daily transfer limit on my current account to £5,000? Happy to provide whatever verification you need.", "request", domain="banking"),
    E("Kindly arrange a replacement fob. Mine stopped working over the weekend and I can't get into the building.", "request", hard=True, domain="facilities"),
    E("Add my new number to the file: 07700 900381. The old one is dead.", "request", domain="telecom"),
    E("Requesting a copy of my full medical record under a subject access request. Please confirm what ID you need.", "request", domain="healthcare"),
    E("Hey, could someone re-issue the certificate? I managed to delete the PDF before saving it. Sorry.", "request", domain="education"),
    E("We need the contract amended to reflect the new entity name — Harbourline Logistics Ltd, not Harbourline Ltd. Legal is holding signature until it's fixed.", "request", domain="b2b"),
    E("Book me onto the November cohort instead of October. Work travel got in the way.", "request", domain="education"),
    E("Please transfer my number to the new SIM. PAC code attached.", "request", domain="telecom"),
    E("Set up a direct debit for the monthly premium rather than the annual lump sum, if that's possible.", "request", domain="insurance"),
    E("Attaching the signed form. Please process the beneficiary change when you can.", "request", domain="finance"),
    E("Can you unlock my account? I locked myself out trying to remember the security answer.", "request", domain="banking"),
    E("I'd like the parcel redirected to my office — 4 Bower Street, not the home address. It's out for delivery tomorrow so this is a bit urgent.", "request", domain="retail"),
    E("Extend the trial by two weeks please, our procurement process is slower than expected.", "request", domain="saas"),
    E("Please delete my account and everything associated with it.", "request", hard=True, domain="saas"),
    E("Could you arrange an interpreter for the appointment on the 3rd? Cantonese.", "request", domain="healthcare"),
    E("send the statement for the last 6 months to this email pls", "request", domain="banking"),

    # ---------------------------------------------------------------- inquiry (25)
    E("What happens to my unused data allowance at the end of the month — does it roll over?", "inquiry", domain="telecom"),
    E("Hi, wondering how long the underwriting usually takes once the medical is done? No rush, just planning.", "inquiry", domain="insurance"),
    E("Do you deliver to the Isle of Skye, and if so is there a surcharge?", "inquiry", domain="retail"),
    E("Dear Sir or Madam,\n\nI am writing to ask whether the postgraduate diploma is recognised by the professional body in Ireland. I could not find this on your website.\n\nYours faithfully,\nA. Nwachukwu", "inquiry", domain="education"),
    E("any update on the claim? submitted it on the 2nd and haven't heard anything", "inquiry", hard=True, domain="insurance"),
    E("Which documents count as proof of address for you? I don't have a utility bill in my name.", "inquiry", domain="banking"),
    E("Is the lounge access included on the business fare or is that an add-on?", "inquiry", domain="travel"),
    E("Could you clarify the difference between the Standard and Enhanced checks? The pricing page lists both but not what changes.", "inquiry", domain="b2b"),
    E("what time does the pharmacy counter close on saturdays", "inquiry", domain="healthcare"),
    E("Where has my order got to? It said 3-5 days and we're on day 9.", "inquiry", hard=True, domain="retail"),
    E("Is there a student rate? I'm enrolled full time until June.", "inquiry", domain="education"),
    E("Can you confirm whether data is held in the EU? Our DPO needs this before we can sign.", "inquiry", domain="b2b"),
    E("How many people can I add to a family plan, and do they all need to live at the same address?", "inquiry", domain="telecom"),
    E("Hello — quick question. If I pause the subscription, do I keep the history or is it wiped?", "inquiry", domain="saas"),
    E("Am I covered for accidental damage abroad, or is that only the home contents part of the policy?", "inquiry", domain="insurance"),
    E("Who do I speak to about a bulk order? Around 400 units.", "inquiry", domain="retail"),
    E("Just checking — is the appointment on the 12th at the Lister site or the main hospital? The letter says both.", "inquiry", domain="healthcare"),
    E("Do you have a status page? Trying to work out if the slowness this morning is us or you.", "inquiry", hard=True, domain="saas"),
    E("Could you tell me what the early repayment charge would be if I settled the loan in March?", "inquiry", domain="finance"),
    E("Tell me what your refund window is for opened items.", "inquiry", domain="retail"),
    E("Is there parking on site and does it need booking in advance?", "inquiry", domain="facilities"),
    E("hi does the warranty still apply if i bought it through a reseller rather than direct", "inquiry", domain="retail"),
    E("What's the notice period on the annual contract? Trying to work out our renewal timeline.", "inquiry", domain="b2b"),
    E("Could you provide a breakdown of the fees on the last statement? Line 4 isn't clear to me.", "inquiry", hard=True, domain="banking"),
    E("When are the results published, and will we be notified by email or post?", "inquiry", domain="education"),

    # ---------------------------------------------------------------- complaint (25)
    E("This is the third engineer visit for the same fault and it is still not fixed. I have taken three days off work for this.", "complaint", domain="telecom"),
    E("I was charged twice for the same booking and nobody has come back to me in eight days. Frankly this is not good enough for a company of your size.", "complaint", domain="travel"),
    E("The dress arrived with a tear along the seam. Clearly it was not checked before dispatch.", "complaint", domain="retail"),
    E("Dear Complaints Team,\n\nI wish to raise a formal complaint regarding the handling of my claim, reference CL-40921. I was told on three separate occasions that a decision was imminent, and on each occasion no decision followed. I have now been without a vehicle for six weeks.\n\nI expect a written response within the statutory timeframe.\n\nYours sincerely,\nD. Okonjo", "complaint", domain="insurance"),
    E("your app has logged me out every single day this week. losing patience with it", "complaint", domain="saas"),
    E("I'm disappointed. The room was not the one shown in the photographs and the air conditioning did not work for two of the three nights.", "complaint", domain="travel"),
    E("Nobody told me the fee had gone up. I only found out when the direct debit came out higher. That should have been communicated.", "complaint", domain="banking"),
    E("Waited 50 minutes past my appointment time and then was seen for four minutes. It felt rushed and I left without my questions answered.", "complaint", domain="healthcare"),
    E("The replacement you sent is the wrong colour. Again. This is the second time.", "complaint", domain="retail"),
    E("I have been passed between four different departments this morning and had to repeat my details each time. Please can someone take ownership of this.", "complaint", hard=True, domain="telecom"),
    E("Honestly the new interface is a step backwards. Things that took one click now take four and there was no warning it was changing.", "complaint", hard=True, domain="saas"),
    E("Still no refund. It has been 21 working days and your policy says 10.", "complaint", domain="retail"),
    E("The course materials were out of date — half the screenshots don't match the current software. For the price, I expected better.", "complaint", domain="education"),
    E("I was assured the engineer would call before arriving. He didn't, I wasn't in, and now I'm told the next slot is in a fortnight.", "complaint", domain="facilities"),
    E("unacceptable. my payment failed because of your outage and now i've been charged a late fee by my landlord", "complaint", domain="finance"),
    E("The delivery driver left the parcel in the recycling bin, in the rain, without ringing the bell. It was destroyed.", "complaint", domain="retail"),
    E("I am not happy that the premium increased by 40% with no change in my circumstances and no explanation offered.", "complaint", domain="insurance"),
    E("Your chatbot is useless and there is no way to reach a human. I have spent 25 minutes trying to find a phone number.", "complaint", domain="saas"),
    E("The seats we paid extra to reserve were reassigned at the gate with no explanation and no refund offered.", "complaint", domain="travel"),
    E("Third time asking. Please stop sending marketing texts. I have unsubscribed twice already.", "complaint", hard=True, domain="telecom"),
    E("The invoice is wrong again — you've billed us for 40 licences and we have 25. This happens most months and it wastes an hour of my time each time.", "complaint", domain="b2b"),
    E("Reception was rude when I asked about the delay. There was no need for that tone.", "complaint", domain="healthcare"),
    E("It broke after nine days. Nine. I'd like to know what you intend to do about it.", "complaint", hard=True, domain="retail"),
    E("Very poor communication throughout. Nobody called back when they said they would, on any of the four occasions.", "complaint", domain="b2b"),
    E("I've been overcharged, underpaid and ignored. At this point I'm considering the ombudsman.", "complaint", domain="finance"),

    # ---------------------------------------------------------------- feedback (25)
    E("Just wanted to say the engineer who came out yesterday, Danny, was excellent. Explained everything and cleaned up after himself.", "feedback", domain="telecom"),
    E("Genuinely the smoothest checkout I've used. No account required, no upsells. Refreshing.", "feedback", domain="retail"),
    E("Dear Team,\n\nI wanted to pass on my thanks to the ward staff on Elm. My mother was there for eleven days and the care she received was exceptional, particularly from the night team who I know are stretched.\n\nWith gratitude,\nH. Pereira", "feedback", domain="healthcare"),
    E("the new dark mode is lovely. easy on the eyes at 1am", "feedback", domain="saas"),
    E("Worth every penny. We've cut our monthly close from six days to two.", "feedback", domain="b2b"),
    E("Loved the course. The tutor was patient with the slower ones of us and never made anyone feel stupid.", "feedback", domain="education"),
    E("Fast, friendly, no fuss. Will use again.", "feedback", domain="retail"),
    E("Hotel was spotless and the breakfast was far better than we expected for the price. Small thing, but the staff remembered our names.", "feedback", domain="travel"),
    E("Honestly I was sceptical about switching but it's been painless. Credit where it's due.", "feedback", domain="banking"),
    E("The claims process was much less painful than I'd braced for. Sorted in four days.", "feedback", domain="insurance"),
    E("Great app overall. If I could suggest one thing, a widget for the home screen would be brilliant.", "feedback", hard=True, domain="saas"),
    E("Thank you for sorting that so quickly yesterday. Real relief.", "feedback", domain="finance"),
    E("Big improvement on last year's event. The venue change was the right call.", "feedback", domain="education"),
    E("Your support person — I think her name was Riya — went well beyond what I expected. Please pass that on.", "feedback", domain="saas"),
    E("Product is good. Packaging is excessive though, four boxes for one small item.", "feedback", hard=True, domain="retail"),
    E("Been with you eleven years and never had a reason to leave. That says something.", "feedback", domain="insurance"),
    E("Really appreciate that you publish the changelog properly. Most companies don't bother.", "feedback", domain="saas"),
    E("The physio was brilliant, gave me exercises I could actually do at home. Much better than last time.", "feedback", domain="healthcare"),
    E("food was decent, service was quick, no complaints at all", "feedback", domain="travel"),
    E("Impressed that someone actually read my last email rather than sending a template. Rare these days.", "feedback", domain="b2b"),
    E("The onboarding call was genuinely useful rather than a sales pitch in disguise. Thanks.", "feedback", domain="b2b"),
    E("Lovely to deal with a company that answers the phone. That alone.", "feedback", domain="telecom"),
    E("Delivery was three days early which never happens. Made my week.", "feedback", domain="retail"),
    E("I like the new statements — much clearer than the old format.", "feedback", domain="banking"),
    E("Ten out of ten. Nothing to add.", "feedback", domain="retail"),
]

if __name__ == "__main__":
    import collections, csv, os
    counts = collections.Counter(r["label"] for r in HOLDOUT)
    out = os.path.join(os.path.dirname(__file__), "holdout_100.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["email", "intent", "hard", "domain"])
        w.writeheader()
        for r in HOLDOUT:
            w.writerow({"email": r["body"], "intent": r["label"],
                        "hard": r["hard"], "domain": r["domain"]})
    words = [len(r["body"].split()) for r in HOLDOUT]
    print("%d emails -> %s" % (len(HOLDOUT), out))
    for k, v in counts.most_common():
        print("  %-12s %3d" % (k, v))
    print("  hard cases : %d" % sum(1 for r in HOLDOUT if r["hard"]))
    print("  domains    : %d" % len({r["domain"] for r in HOLDOUT}))
    print("  words      : min %d  median %d  max %d" % (min(words), sorted(words)[len(words)//2], max(words)))
