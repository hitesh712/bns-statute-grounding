"""
Phase 1c -- build DEFINES and PUNISHED_BY from the statute.

The defect being fixed: the original edges were inferred from offence words
appearing anywhere in a section's BODY, so BNS_111 (organised crime),
BNS_112 and BNS_317 (stolen property) attached to three or four offences each,
`murder` attached to BNS_93 (abandoning a child), and `cheating` attached to
BNS_112/BNS_317 but never to BNS_318, which actually defines it.

The fix: a section is linked to an offence only when its own MARGINAL TITLE --
the statutory statement of what the section is about -- names that offence.
Punishments are read from the body using the formulaic sentencing language.

Input :  data/bns_act_titled.csv
Output:  data/graph_defines.csv       section_id, offence, title
         data/graph_punished_by.csv   offence, punishment
         data/graph_stats.txt
"""

import os
import re
import pandas as pd
from collections import defaultdict

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Title-level surface forms per coarse offence category. Matched against the
# marginal title only, never the body.
CATEGORY_TERMS = {
    "theft":             [r"\btheft\b", r"\bsnatching\b"],
    "robbery":           [r"\brobbery\b", r"\bdacoity\b"],
    "extortion":         [r"\bextortion\b"],
    "cheating":          [r"\bcheating\b"],
    "forgery":           [r"\bforgery\b", r"\bforged\b", r"\bforging\b",
                          r"\bfalse document\b", r"\bfalse electronic record\b"],
    "murder":            [r"\bmurder\b", r"\bculpable homicide\b"],
    "rape":              [r"\brape\b"],
    "sexual harassment": [r"\bsexual harassment\b"],
    "stalking":          [r"\bstalking\b"],
}

# Chapter headings and artefacts that trail the marginal title after extraction.
TITLE_NOISE = re.compile(
    r"\s*(?:\.\s*)?(?:Of\s+[A-Za-z ,'\-]+|\((?:[ivxlc]+|[0-9]+)\)|CHAPTER\s+[IVXLC]+)\s*$")


def clean_title(t):
    t = " ".join(str(t).split())
    for _ in range(3):
        t2 = TITLE_NOISE.sub("", t).strip().rstrip(".")
        if t2 == t:
            break
        t = t2
    return t.strip().rstrip(".")


def extract_punishments(body):
    """Structured sentencing elements from the formulaic drafting."""
    b = " ".join(str(body).lower().split())
    out = set()
    if re.search(r"punished with death|with death\b", b):
        out.add("death")
    if "imprisonment for life" in b:
        out.add("imprisonment for life")
    if re.search(r"\bcommunity service\b", b):
        out.add("community service")
    if re.search(r"\bliable to fine\b|\bwith fine\b|\band fine\b", b):
        out.add("fine")
    for m in re.finditer(
            r"which may extend to (\w+|\d+)\s+years", b):
        out.add(f"up to {m.group(1)} years imprisonment")
    for m in re.finditer(
            r"shall not be less than (\w+|\d+)\s+years", b):
        out.add(f"minimum {m.group(1)} years imprisonment")
    if re.search(r"which may extend to (\w+|\d+)\s+months", b):
        out.add("imprisonment up to a term of months")
    return out


def main():
    df = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    df["title_clean"] = df["title"].map(clean_title)

    defines = []
    for _, r in df.iterrows():
        t = str(r["title_clean"]).lower()
        if not t:
            continue
        for cat, pats in CATEGORY_TERMS.items():
            if any(re.search(p, t) for p in pats):
                defines.append({
                    "section_id": r["section_id"],
                    "offence": cat,
                    "title": r["title_clean"],
                })

    d = pd.DataFrame(defines).drop_duplicates()
    d.to_csv(os.path.join(DATA, "graph_defines.csv"), index=False, encoding="utf-8")

    # PUNISHED_BY: union of sentencing elements over the sections of an offence
    body_by_id = dict(zip(df["section_id"], df["body"]))
    pun = defaultdict(set)
    for _, r in d.iterrows():
        pun[r["offence"]] |= extract_punishments(body_by_id.get(r["section_id"], ""))
    pb = pd.DataFrame(
        [{"offence": o, "punishment": p} for o in sorted(pun) for p in sorted(pun[o])])
    pb.to_csv(os.path.join(DATA, "graph_punished_by.csv"), index=False, encoding="utf-8")

    # ---- report -----------------------------------------------------------
    lines = []
    lines.append(f"sections with a usable title : {(df['title_clean']!='').sum()}/{len(df)}")
    lines.append(f"DEFINES edges                : {len(d)}")
    lines.append(f"sections carrying an edge    : {d.section_id.nunique()}")
    lines.append(f"PUNISHED_BY edges            : {len(pb)}")
    lines.append("")
    lines.append("DEFINES per offence:")
    for cat, g in d.groupby("offence"):
        lines.append(f"  {cat:<20} {len(g):3d}  "
                     f"{', '.join(sorted(g.section_id, key=lambda s:int(s.split('_')[1])))}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(DATA, "graph_stats.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\nsection titles behind each edge:")
    for cat, g in d.groupby("offence"):
        print(f"\n  [{cat}]")
        for _, r in g.sort_values("section_id").iterrows():
            print(f"    {r.section_id:<9} {r.title[:66]}")

    print("\nPUNISHED_BY:")
    for o, g in pb.groupby("offence"):
        print(f"  {o:<20} {'; '.join(g.punishment)}")


if __name__ == "__main__":
    main()
