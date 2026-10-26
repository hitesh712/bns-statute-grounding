"""
Extract the statutory text of every section that carries a DEFINES edge.

READ-ONLY with respect to the original implementation: original_data/ is
only ever read from. All output goes to data/.

Purpose: the harder benchmark must be grounded in the conduct the statute
actually describes, not in invented scenarios. This dumps the text we write
fact-pattern queries from.
"""

import os
import pandas as pd

OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")


def main():
    os.makedirs(DATA, exist_ok=True)

    chunks = pd.read_csv(os.path.join(OLD, "chunks", "legal_chunks.csv"))
    links = pd.read_csv(os.path.join(OLD, "results", "offence_links.csv"))

    chunks["chunk_id"] = chunks["chunk_id"].astype(str).str.strip()
    links["section_id"] = links["section_id"].astype(str).str.strip()
    links["offence"] = links["offence"].astype(str).str.strip().str.lower()

    merged = links.merge(
        chunks[["chunk_id", "act", "section", "content"]],
        left_on="section_id", right_on="chunk_id", how="left",
    )

    missing = merged["content"].isna().sum()
    print(f"DEFINES edges            : {len(links)}")
    print(f"distinct linked sections : {links['section_id'].nunique()}")
    print(f"offence categories       : {links['offence'].nunique()}")
    if missing:
        print(f"WARNING: {missing} linked sections have no text in the corpus")
        print(merged[merged["content"].isna()][["section_id", "offence"]].to_string())

    merged = merged.drop(columns=["chunk_id"])
    out = os.path.join(DATA, "linked_sections.csv")
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"\nwrote {out}")

    # Readable dump for drafting queries against the real statutory conduct.
    txt = os.path.join(DATA, "linked_sections.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for off, grp in merged.groupby("offence"):
            f.write("=" * 78 + "\n")
            f.write(f"OFFENCE: {off}   ({len(grp)} linked sections)\n")
            f.write("=" * 78 + "\n\n")
            for _, r in grp.iterrows():
                body = str(r["content"]).replace("\r", " ")
                body = " ".join(body.split())
                f.write(f"--- {r['section_id']} ---\n{body[:1800]}\n\n")
    print(f"wrote {txt}")

    print("\nsections per offence:")
    print(links.groupby("offence")["section_id"].nunique().to_string())


if __name__ == "__main__":
    main()
