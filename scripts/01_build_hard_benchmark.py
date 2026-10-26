"""
Build the harder fact-pattern benchmark (BNS-Facts-135).

Design rules, applied to every item:
  1. The query describes a factual situation as a lay complainant would report
     it. It never contains the gold offence label, nor any term from the
     original LEGAL_SYNONYMS dictionary.
  2. Every item is grounded in the conduct a specific BNS section describes;
     the section is recorded so the gold label is auditable against statute.
  3. Each category carries "plain" items plus "boundary" items that sit close
     to a neighbouring offence (theft/robbery/extortion, cheating/theft,
     rape/sexual harassment), so the benchmark tests legal discrimination and
     not vocabulary matching.

Provenance: these items were drafted from the statutory text, not collected
from real users. They still require review by a qualified legal reader before
publication; `reviewed_by` is left blank for that purpose.

Writes: data/bns_facts_135.csv
"""

import os
import csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")

# (expected_offence, difficulty, grounding_section, query)
ITEMS = [
    # ---------------- THEFT (BNS 303/304/305/306/314) ----------------
    ("theft", "plain", "BNS_303", "My bicycle was removed from outside my house while I was asleep."),
    ("theft", "plain", "BNS_306", "A colleague opened my bag at work and took my cash without asking."),
    ("theft", "plain", "BNS_305", "Someone entered my shop at night and carried away goods from the shelves."),
    ("theft", "plain", "BNS_303", "My mobile phone disappeared from my pocket while I was in a crowded bus."),
    ("theft", "plain", "BNS_306", "A domestic helper took jewellery from my cupboard without permission."),
    ("theft", "plain", "BNS_305", "Someone drove away my motorcycle that was parked outside the market."),
    ("theft", "plain", "BNS_303", "A person picked up my laptop from the library table and walked out with it."),
    ("theft", "plain", "BNS_306", "An employee removed money from the shop till on several occasions."),
    ("theft", "plain", "BNS_303", "My neighbour carried away construction material from my plot without telling me."),
    ("theft", "plain", "BNS_305", "A passenger took my suitcase from the overhead rack and left the train with it."),
    ("theft", "boundary", "BNS_304", "A man on a motorcycle grabbed the chain from my neck and sped away without striking me."),
    ("theft", "boundary", "BNS_303", "I left my wallet on a shop counter and another customer quietly put it in his pocket."),
    ("theft", "boundary", "BNS_314", "A delivery agent kept the parcel for himself instead of handing it over to me."),
    ("theft", "boundary", "BNS_303", "A tenant removed the water pump from the house when vacating, without my consent."),
    ("theft", "boundary", "BNS_305", "Someone broke the lock of my godown and took away bags of grain."),

    # ---------------- ROBBERY (BNS 309/310/311/312) ----------------
    ("robbery", "plain", "BNS_309", "Two men held a knife to me on the street and took my wallet."),
    ("robbery", "plain", "BNS_309", "A man beat me and pulled the bag from my hand before running away."),
    ("robbery", "plain", "BNS_311", "An armed man injured my arm and took the cash I was carrying to the bank."),
    ("robbery", "plain", "BNS_309", "Someone put a hand over my mouth, threatened to hurt me, and took my phone."),
    ("robbery", "plain", "BNS_310", "Six men surrounded our vehicle, assaulted the driver and took everything inside."),
    ("robbery", "plain", "BNS_309", "A man pushed me to the ground and snatched my purse while I resisted."),
    ("robbery", "plain", "BNS_311", "Two persons attacked a shopkeeper with an iron rod and emptied his cash drawer."),
    ("robbery", "plain", "BNS_309", "I was threatened with immediate harm and made to hand over my gold ring on the spot."),
    ("robbery", "plain", "BNS_312", "The accused was carrying a firearm when he tried to take the cash van's money."),
    ("robbery", "plain", "BNS_310", "A group of seven persons jointly attacked a house and carried away valuables."),
    ("robbery", "boundary", "BNS_309", "A man demanded my wallet and said he would stab me right then if I refused, so I gave it."),
    ("robbery", "boundary", "BNS_309", "He took my bag only after knocking me down; no weapon was used."),
    ("robbery", "boundary", "BNS_311", "During the incident the accused caused a serious head injury while taking the money."),
    ("robbery", "boundary", "BNS_309", "The accused used force to prevent me from resisting while removing my belongings."),
    ("robbery", "boundary", "BNS_310", "Five men acting together attempted to take our goods by force but fled when police arrived."),

    # ---------------- EXTORTION (BNS 308) ----------------
    ("extortion", "plain", "BNS_308", "A man said he would harm my family unless I handed over money every month."),
    ("extortion", "plain", "BNS_308", "Someone threatened to damage my shop if I did not pay him a weekly amount."),
    ("extortion", "plain", "BNS_308", "A caller said he would release private photographs of me unless I transferred money."),
    ("extortion", "plain", "BNS_308", "I was told my son would be hurt if I did not sign over the property papers."),
    ("extortion", "plain", "BNS_308", "A local group demanded payment and warned of consequences for my business if I refused."),
    ("extortion", "plain", "BNS_308", "He threatened to make false allegations against me unless I gave him my savings."),
    ("extortion", "plain", "BNS_308", "A man put me in fear of injury and made me transfer a valuable security to him."),
    ("extortion", "plain", "BNS_308", "Someone said my house would be set on fire if I did not pay the amount demanded."),
    ("extortion", "plain", "BNS_308", "I was frightened into handing over my cheque book by a threat of future violence."),
    ("extortion", "plain", "BNS_308", "A person demanded payment and said he would injure me later if I did not comply."),
    ("extortion", "boundary", "BNS_308", "He threatened future harm and I paid him the next day at the bank."),
    ("extortion", "boundary", "BNS_308", "The threat was to harm my brother, not me, and I delivered the money because of it."),
    ("extortion", "boundary", "BNS_308", "No weapon was shown and nothing was taken by force; I handed it over out of fear of what he would do."),
    ("extortion", "boundary", "BNS_308", "A man threatened to expose a private matter unless I signed a document transferring money."),
    ("extortion", "boundary", "BNS_308", "He put me in fear of injury and induced me to deliver a valuable item to a third person."),

    # ---------------- CHEATING (BNS 318/319) ----------------
    ("cheating", "plain", "BNS_318", "A person took an advance payment promising goods he never intended to supply."),
    ("cheating", "plain", "BNS_318", "A builder collected booking money for flats that were never going to be constructed."),
    ("cheating", "plain", "BNS_318", "Someone sold me land by showing papers for a plot he did not own."),
    ("cheating", "plain", "BNS_318", "An agent promised a government job in return for money and then disappeared."),
    ("cheating", "plain", "BNS_318", "A trader delivered inferior material after taking payment for a superior grade."),
    ("cheating", "plain", "BNS_319", "A man came to my door claiming to be a bank officer and collected my card and cash."),
    ("cheating", "plain", "BNS_318", "I was induced to invest by false promises of guaranteed monthly returns."),
    ("cheating", "plain", "BNS_318", "Someone used a false identity online and persuaded me to transfer money to his account."),
    ("cheating", "plain", "BNS_318", "A shopkeeper knowingly gave me counterfeit currency as change."),
    ("cheating", "plain", "BNS_318", "A travel agent took payment for tickets he never booked."),
    ("cheating", "boundary", "BNS_318", "By deceiving me he induced me to deliver my property to him voluntarily."),
    ("cheating", "boundary", "BNS_318", "Nothing was taken from my possession; I handed it over because I was misled about the facts."),
    ("cheating", "boundary", "BNS_319", "He pretended to be a police officer and collected a fine from me that did not exist."),
    ("cheating", "boundary", "BNS_318", "A contractor was paid in full after falsely certifying that work had been completed."),
    ("cheating", "boundary", "BNS_318", "I signed a document believing it was a receipt because the other party misrepresented its contents."),

    # ---------------- FORGERY (BNS 336/340/341) ----------------
    ("forgery", "plain", "BNS_336", "Someone created a fake property document carrying my signature."),
    ("forgery", "plain", "BNS_336", "A false mark sheet was prepared in my institution's name."),
    ("forgery", "plain", "BNS_340", "A man presented a document he knew to be false as if it were genuine in court."),
    ("forgery", "plain", "BNS_341", "A person made a duplicate rubber stamp of our office seal."),
    ("forgery", "plain", "BNS_336", "My employee prepared false invoices to support a claim against the company."),
    ("forgery", "plain", "BNS_336", "A false electronic record was created to show a payment that never happened."),
    ("forgery", "plain", "BNS_340", "He used a fabricated experience certificate to obtain employment."),
    ("forgery", "plain", "BNS_336", "Someone altered the figures on a cheque after I had signed it."),
    ("forgery", "plain", "BNS_336", "A will was prepared in my late father's name after his death."),
    ("forgery", "plain", "BNS_341", "An instrument was manufactured for the purpose of making false impressions of a government seal."),
    ("forgery", "boundary", "BNS_336", "A false document was made with intent to support a claim to property."),
    ("forgery", "boundary", "BNS_340", "He did not make the document himself but knowingly used the false one as genuine."),
    ("forgery", "boundary", "BNS_336", "Part of a genuine agreement was altered without authority to change its meaning."),
    ("forgery", "boundary", "BNS_336", "An electronic record was fabricated to cause injury to another person's reputation."),
    ("forgery", "boundary", "BNS_341", "Counterfeit plates were kept in his possession for producing false official impressions."),

    # ---------------- MURDER (BNS 101/103/105) ----------------
    ("murder", "plain", "BNS_103", "A man attacked his neighbour with an axe intending to end his life, and the neighbour died."),
    ("murder", "plain", "BNS_103", "The accused shot the victim deliberately and the victim died at the scene."),
    ("murder", "plain", "BNS_103", "Poison was administered with the intention of causing death and the person died."),
    ("murder", "plain", "BNS_103", "He stabbed the deceased repeatedly in the chest knowing it would cause death."),
    ("murder", "plain", "BNS_105", "A quarrel turned violent and one man struck another, causing a death he did not intend."),
    ("murder", "plain", "BNS_103", "The accused planned and carried out an attack that resulted in the victim's death."),
    ("murder", "plain", "BNS_110", "He attacked the victim intending to cause a fatal injury, but the victim survived."),
    ("murder", "plain", "BNS_103", "A group acting together caused the death of a man because of his caste."),
    ("murder", "plain", "BNS_105", "The accused caused death by an act done with knowledge that it was likely to cause death."),
    ("murder", "plain", "BNS_103", "A husband set his wife on fire and she succumbed to her injuries."),
    ("murder", "boundary", "BNS_105", "Death was caused in a sudden fight without premeditation."),
    ("murder", "boundary", "BNS_103", "What punishment applies when one person deliberately ends another person's life?"),
    ("murder", "boundary", "BNS_110", "The act would have caused death had it succeeded, but the victim recovered."),
    ("murder", "boundary", "BNS_105", "He intended to cause bodily injury likely to cause death and the person died."),
    ("murder", "boundary", "BNS_103", "Five persons acting in concert caused a death on communal grounds."),

    # ---------------- RAPE (BNS 63/64/65/69/70) ----------------
    ("rape", "plain", "BNS_64", "A woman reports that a man had sexual intercourse with her without her consent."),
    ("rape", "plain", "BNS_65", "The complainant is a girl aged fourteen and the accused is an adult man."),
    ("rape", "plain", "BNS_70", "A woman was subjected to non-consensual intercourse by several men acting together."),
    ("rape", "plain", "BNS_69", "A man obtained intercourse by promising marriage he never intended to honour."),
    ("rape", "plain", "BNS_64", "Consent was obtained by putting the woman in fear of death."),
    ("rape", "plain", "BNS_64", "The woman was unable to communicate consent because she was unconscious."),
    ("rape", "plain", "BNS_65", "The victim was below sixteen years of age at the time of the incident."),
    ("rape", "plain", "BNS_64", "The accused had intercourse with the complainant against her will."),
    ("rape", "plain", "BNS_69", "Intercourse was obtained by deceitful means as to the man's identity."),
    ("rape", "plain", "BNS_70", "Multiple accused acted in furtherance of a common intention against one woman."),
    ("rape", "boundary", "BNS_64", "Consent was given but only because the woman was threatened with harm to her child."),
    ("rape", "boundary", "BNS_69", "The relationship was consensual but based on a false promise of marriage."),
    ("rape", "boundary", "BNS_64", "The woman was intoxicated and unable to understand what she was consenting to."),
    ("rape", "boundary", "BNS_65", "The accused claims consent, but the complainant was fifteen years old."),
    ("rape", "boundary", "BNS_64", "There was penetration without the woman's willing agreement."),

    # ---------------- SEXUAL HARASSMENT (BNS 75) ----------------
    ("sexual harassment", "plain", "BNS_75", "A manager repeatedly asked a female employee for favours of a sexual nature in exchange for promotion."),
    ("sexual harassment", "plain", "BNS_75", "A man made unwelcome physical advances towards a woman at her workplace."),
    ("sexual harassment", "plain", "BNS_75", "A colleague showed pornographic images to a woman against her will."),
    ("sexual harassment", "plain", "BNS_75", "A man made repeated remarks of a sexual nature to a woman in the office."),
    ("sexual harassment", "plain", "BNS_75", "A woman was subjected to unwelcome explicit overtures by a senior at work."),
    ("sexual harassment", "plain", "BNS_75", "A man demanded favours of a sexual kind from a woman seeking his help."),
    ("sexual harassment", "plain", "BNS_75", "Unwanted physical contact of a sexual nature was made by a co-worker."),
    ("sexual harassment", "plain", "BNS_75", "A woman received repeated obscene comments from a man at her place of work."),
    ("sexual harassment", "plain", "BNS_75", "A shopkeeper made explicit suggestions to a woman customer that she did not welcome."),
    ("sexual harassment", "plain", "BNS_75", "A teacher made inappropriate advances of a sexual nature towards a student."),
    ("sexual harassment", "boundary", "BNS_75", "There was no physical contact, only repeated remarks of a sexual nature."),
    ("sexual harassment", "boundary", "BNS_75", "A request for favours of a sexual kind was made but refused and nothing further occurred."),
    ("sexual harassment", "boundary", "BNS_75", "A man showed explicit material to a woman who had not asked to see it."),
    ("sexual harassment", "boundary", "BNS_75", "Unwelcome advances were made but the woman was able to leave immediately."),
    ("sexual harassment", "boundary", "BNS_75", "The conduct was verbal only and occurred in a professional setting."),

    # ---------------- STALKING (BNS 78) ----------------
    ("stalking", "plain", "BNS_78", "A man keeps following a woman to her workplace despite being told to stop."),
    ("stalking", "plain", "BNS_78", "A woman is repeatedly contacted by a man even after she made her disinterest clear."),
    ("stalking", "plain", "BNS_78", "A man monitors a woman's internet activity and messages her constantly."),
    ("stalking", "plain", "BNS_78", "A woman is followed home every evening by the same person."),
    ("stalking", "plain", "BNS_78", "A man waits outside a woman's college daily and attempts to speak to her against her wishes."),
    ("stalking", "plain", "BNS_78", "Repeated attempts at personal contact continue despite a clear refusal by the woman."),
    ("stalking", "plain", "BNS_78", "A man tracks a woman's online accounts and turns up wherever she goes."),
    ("stalking", "plain", "BNS_78", "A woman reports being watched and pursued by a man over several weeks."),
    ("stalking", "plain", "BNS_78", "A man sends messages continuously to a woman who has blocked him repeatedly."),
    ("stalking", "plain", "BNS_78", "A woman is persistently pursued by a man seeking personal interaction she has refused."),
    ("stalking", "boundary", "BNS_78", "The man never approached her physically but monitored her electronic communication."),
    ("stalking", "boundary", "BNS_78", "Contact was attempted repeatedly online rather than in person."),
    ("stalking", "boundary", "BNS_78", "He followed her on only two occasions but continues to message her daily."),
    ("stalking", "boundary", "BNS_78", "The woman had clearly indicated disinterest and the attempts at contact continued."),
    ("stalking", "boundary", "BNS_78", "A man observed a woman's movements over time without ever speaking to her."),
]


def main():
    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "bns_facts_135.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query", "expected_offence", "difficulty",
                    "grounding_section", "reviewed_by"])
        for i, (off, diff, sec, q) in enumerate(ITEMS, start=1):
            w.writerow([f"F{i:03d}", q, off, diff, sec, ""])

    print(f"wrote {out}  ({len(ITEMS)} items)")
    from collections import Counter
    c = Counter(o for o, _, _, _ in ITEMS)
    d = Counter(df for _, df, _, _ in ITEMS)
    print("\nper category:")
    for k in sorted(c):
        print(f"  {k:<20} {c[k]}")
    print("\nper difficulty:")
    for k in sorted(d):
        print(f"  {k:<20} {d[k]}")


if __name__ == "__main__":
    main()
