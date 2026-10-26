"""
Phase 1a -- parse the official BNS 2023 text into sections with marginal titles.

The marginal title is what makes a statute-grounded graph possible: it states
what a section is ABOUT ("103. Punishment for murder"), whereas the original
implementation inferred offence links from whichever offence words happened to
appear anywhere in a section's body. That is the defect this replaces.

Input :  corpus/bns_prs.txt   (official text, 137 pp)
Output:  data/bns_sections.csv   section_no, title, body
"""

import os
import re

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus", "bns_prs.txt")
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CHAPTER_LINE = re.compile(
    r"^\s*(CHAPTER\s+[IVXLC]+|OF\s+[A-Z ,'\-]+|[A-Z][A-Z ,'\-]{6,})\s*$")


def clean(s):
    return " ".join(s.replace("\u2019", "'").split())


def split_numbered(text):
    """Yield (number, chunk) for entries beginning '<n>.' at a line start."""
    starts = [(int(m.group(1)), m.start(), m.end())
              for m in re.finditer(r"(?m)^\s*(\d{1,3})\.\s*", text)]
    for i, (num, s, e) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
        yield num, text[e:end]


def parse_titles(arrangement):
    """section number -> marginal title, from the Arrangement of Clauses."""
    titles = {}
    for num, chunk in split_numbered(arrangement):
        lines = [l for l in chunk.splitlines()
                 if l.strip() and not CHAPTER_LINE.match(l)]
        if not lines:
            continue
        title = clean(" ".join(lines)).rstrip(".")
        # The arrangement is strictly ascending; a repeat means we have run
        # into the body, so keep the first (authoritative) occurrence.
        if num not in titles and 0 < len(title) < 300:
            titles[num] = title
    return titles


def parse_bodies(body):
    bodies = {}
    for num, chunk in split_numbered(body):
        txt = clean(chunk)
        if num not in bodies and len(txt) > 40:
            bodies[num] = txt
    return bodies


def main():
    os.makedirs(DATA, exist_ok=True)
    text = open(CORPUS, encoding="utf-8").read()

    # The body restarts the numbering at "Short title"; the second occurrence
    # of that phrase marks the boundary.
    marks = [m.start() for m in re.finditer(r"Short title", text)]
    split_at = marks[1] if len(marks) > 1 else len(text) // 6
    # back up to the "1." that introduces it
    m = re.search(r"(?m)^\s*1\.\s*", text[max(0, split_at - 400):split_at + 50])
    if m:
        split_at = max(0, split_at - 400) + m.start()

    arrangement, body = text[:split_at], text[split_at:]
    print(f"arrangement: {len(arrangement):,} chars | body: {len(body):,} chars")

    titles = parse_titles(arrangement)
    bodies = parse_bodies(body)
    print(f"titles parsed : {len(titles)}")
    print(f"bodies parsed : {len(bodies)}")

    nums = sorted(set(titles) | set(bodies))
    rows = []
    for n in nums:
        if n < 1 or n > 358:
            continue
        rows.append({
            "section_id": f"BNS_{n}",
            "section_no": n,
            "title": titles.get(n, ""),
            "body": bodies.get(n, ""),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "bns_sections.csv")
    df.to_csv(out, index=False, encoding="utf-8")

    miss_t = (df["title"] == "").sum()
    miss_b = (df["body"] == "").sum()
    print(f"\nrows: {len(df)}   missing title: {miss_t}   missing body: {miss_b}")

    print("\nspot checks:")
    for n in (63, 64, 75, 78, 101, 103, 105, 303, 304, 308, 309, 318, 319, 336):
        r = df[df.section_no == n]
        if len(r):
            print(f"  BNS_{n:<4} {r.iloc[0]['title'][:70]}")
        else:
            print(f"  BNS_{n:<4} MISSING")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
