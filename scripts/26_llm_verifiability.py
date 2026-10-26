"""
Phase 6 -- verifiability: does a frontier LLM cite the CORRECT BNS provision?

Opus 5.5 classifies BNS-Facts offences almost perfectly zero-shot (97.8% on
v1.0). Offence labels are not where an advisory system earns trust, though.
The user needs the section that applies and the punishment it prescribes, and
the BNS (2023) renumbered every provision of the IPC (1860). A model trained on
decades of IPC-era text may answer with IPC numbers, or with BNS numbers that
do not correspond to the conduct. A statute-grounded retriever cannot: every
section it cites is a real BNS section, quoted from the Act.

Two request types, one batch, BNS-Facts-135 v1.1:
  cls-<id>  pure offence classification (same prompt as 23_llm_baseline.py)
  cit-<id>  offence + BNS section + maximum imprisonment + death possible

The citation prompt states explicitly that the BNS replaced the IPC and asks
for BNS numbers -- every chance is given to the model, so any IPC citation is a
knowledge gap, not a misunderstanding of the question.

Scoring against the statute:
  section (family)  cited section defines the gold offence (statute-grounded
                    graph) or is the item's grounding section
  section (exact)   cited section == the item's grounding section
  IPC confusion     cited number is the IPC section for that offence
  nonexistent       cited number > 358 (the BNS has 358 sections)
  max imprisonment  matches a sentencing tier the grounding section prescribes
  death possible    scored only on sections where it is unambiguous

Usage:
  python 26_llm_verifiability.py pilot
  python 26_llm_verifiability.py submit
  python 26_llm_verifiability.py collect [--wait]
"""

import os
import re
import sys
import json
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
SCRIPTS = os.path.join(HERE, "scripts")

_spec = importlib.util.spec_from_file_location("b", os.path.join(SCRIPTS, "23_llm_baseline.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)          # shares client(), usd(), ledger, LABELS, prompts

MODEL, EFFORT, LABELS = B.MODEL, B.EFFORT, B.LABELS
BENCH = os.path.join(DATA, "bns_facts_135_v1_1.csv")
PILOT = os.path.join(RES, "verif_pilot_stats.json")
STATE = os.path.join(RES, "verif_batch_state.json")
BUDGET = 1.67
BNS_MAX_SECTION = 358

CIT_SYSTEM = ("You are an expert on Indian criminal law. Answer under the Bharatiya "
              "Nyaya Sanhita, 2023 (BNS). The BNS replaced the Indian Penal Code (IPC) "
              "from 1 July 2024 and renumbered its provisions: give BNS section numbers, "
              "not IPC section numbers.")

CIT_SCHEMA = {
    "type": "object",
    "properties": {
        "offence": {"type": "string", "enum": LABELS},
        "bns_section": {"type": "integer"},
        "max_imprisonment": {"type": "string"},
        "death_possible": {"type": "boolean"},
    },
    "required": ["offence", "bns_section", "max_imprisonment", "death_possible"],
    "additionalProperties": False,
}

# IPC numbers for each coarse offence (definition and punishment sections).
# Disjoint from the valid BNS sections of the same offence, so a hit is unambiguous.
IPC = {
    "murder": {299, 300, 302, 304, 307, 308},
    "rape": {375, 376},
    "theft": {378, 379, 380, 381},
    "robbery": {390, 391, 392, 393, 394, 395, 396, 397, 398},
    "extortion": {383, 384, 385, 386, 387, 388, 389},
    "cheating": {415, 416, 417, 418, 419, 420},
    "forgery": set(range(463, 475)),
    "sexual harassment": {354},       # IPC 354A
    "stalking": {354},                # IPC 354D
}

# Maximum-imprisonment tiers each grounding section prescribes, read from the
# Act's sentencing clauses (see journal/RESULTS.md, Phase 6). "life" = imprisonment for life.
TIERS = {
    "BNS_64": {"life"}, "BNS_65": {"life"}, "BNS_70": {"life"},
    "BNS_75": {"3", "1"}, "BNS_78": {"3", "5"},
    "BNS_103": {"life"}, "BNS_105": {"life", "10"}, "BNS_110": {"3", "7"},
    "BNS_303": {"3", "5"}, "BNS_304": {"3"}, "BNS_305": {"7"}, "BNS_306": {"7"},
    "BNS_308": {"2", "7", "10"},
    "BNS_309": {"7", "10", "14", "life"}, "BNS_310": {"life", "10"},
    "BNS_311": {"life", "10", "14"}, "BNS_312": {"7", "10", "14", "life"},
    "BNS_318": {"3", "5", "7"}, "BNS_319": {"5"},
    "BNS_336": {"2", "3", "7"}, "BNS_340": {"2", "3", "7", "10", "life"},
    "BNS_341": {"life", "7", "3"},
    # Sections a model may legitimately cite instead of the grounding section.
    "BNS_66": {"life"}, "BNS_74": {"5"}, "BNS_79": {"3"}, "BNS_109": {"10", "life"},
    "BNS_337": {"7"}, "BNS_338": {"life", "10"}, "BNS_339": {"7", "life"},
    "BNS_351": {"2", "7"},
}
# Death penalty: scored only where the grounding section settles it for the item.
DEATH = {"BNS_103": True}
for s in ("BNS_64", "BNS_75", "BNS_78", "BNS_105", "BNS_110", "BNS_303", "BNS_304",
          "BNS_305", "BNS_306", "BNS_308", "BNS_309", "BNS_311", "BNS_312",
          "BNS_318", "BNS_319", "BNS_336", "BNS_340", "BNS_341"):
    DEATH[s] = False


def cit_user(q):
    return (f"Situation: {q}\n\n"
            "Under the BNS, which single section most specifically makes this conduct "
            "punishable, and what does that section prescribe? Give:\n"
            f"- offence: the best-fitting category from: {', '.join(LABELS)}\n"
            "- bns_section: the BNS section number\n"
            "- max_imprisonment: the maximum term of imprisonment that section prescribes "
            "for this conduct, as a number of years in digits, or \"life\"\n"
            "- death_possible: whether that section permits the death penalty for this conduct")


def cit_params(q, max_tokens):
    return dict(model=MODEL, max_tokens=max_tokens, system=CIT_SYSTEM,
                output_config={"effort": EFFORT,
                               "format": {"type": "json_schema", "schema": CIT_SCHEMA}},
                messages=[{"role": "user", "content": cit_user(q)}])


def parse_max(s):
    t = str(s).lower()
    if "life" in t:
        return "life"
    m = re.search(r"\d+", t)
    return m.group(0) if m else "?"


def load_bench():
    b = pd.read_csv(BENCH)
    b["query"] = b["query"].astype(str).str.strip()
    b["gold"] = b["expected_offence"].astype(str).str.strip().str.lower()
    return b


def defining_sections():
    d = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    out = {}
    for _, r in d.iterrows():
        out.setdefault(str(r.offence).strip().lower(), set()).add(
            int(str(r.section_id).split("_")[1]))
    return out


# -------------------------------------------------------------------- pilot --

def cmd_pilot(a):
    import anthropic
    os.makedirs(RES, exist_ok=True)
    c = B.client()
    b = load_bench()
    sample = b.groupby("gold", sort=True).head(1).reset_index(drop=True)
    print(f"PILOT: {len(sample)} citation calls to {MODEL} (effort={EFFORT})\n")
    rows, spent = [], 0.0
    for _, r in sample.iterrows():
        try:
            msg = c.messages.create(**cit_params(r["query"], max_tokens=4000))
        except anthropic.BadRequestError as e:
            sys.exit(f"Request rejected: {e.message}")
        except anthropic.AuthenticationError:
            sys.exit("Authentication failed.")
        u = msg.usage
        cost = B.usd(u.input_tokens, u.output_tokens)
        spent += cost
        txt = B.text_of(msg)
        try:
            j = json.loads(txt)
        except json.JSONDecodeError:
            j = {}
        rows.append({"in": u.input_tokens, "out": u.output_tokens, "usd": cost})
        print(f"  {r['query_id']} gold={r['gold']:<18} {r['grounding_section']:<8} -> "
              f"{json.dumps(j)}   out={u.output_tokens} ${cost:.4f} stop={msg.stop_reason}")
    total = B.ledger_add(spent, f"verifiability pilot {len(rows)} calls")
    d = pd.DataFrame(rows)
    stats = {"mean_in": float(d["in"].mean()), "mean_out": float(d["out"].mean()),
             "max_out": int(d["out"].max())}
    json.dump(stats, open(PILOT, "w"), indent=2)
    est = 135 * B.usd(stats["mean_in"], 1.5 * stats["mean_out"], batch=True)
    print(f"\npilot ${spent:.4f} (ledger ${total:.4f}); mean {stats['mean_in']:.0f} in / "
          f"{stats['mean_out']:.0f} out; projected citation batch ${est:.3f}")


# ------------------------------------------------------------------- submit --

def cmd_submit(a):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if not os.path.exists(PILOT):
        sys.exit("Run the pilot first.")
    if os.path.exists(STATE) and not a.force:
        sys.exit("Batch already submitted; use collect.")
    st = json.load(open(PILOT))
    b = load_bench()
    cls_cost = len(b) * B.usd(160, 40, batch=True)
    cit_cost = len(b) * B.usd(st["mean_in"], 1.5 * st["mean_out"], batch=True)
    spent = B.ledger_get()["spent_usd"]
    avail = BUDGET - spent - B.SAFETY_MARGIN
    print(f"projected ${cls_cost + cit_cost:.3f}  available ${avail:.3f}")
    if cls_cost + cit_cost > avail:
        sys.exit("REFUSED: over budget.")
    mt_cit = int(min(8000, max(1500, 3 * st["max_out"])))
    reqs = []
    for r in b.itertuples():
        reqs.append(Request(custom_id=f"cls-{r.query_id}",
                            params=MessageCreateParamsNonStreaming(**B.request_params(r.query, 1500))))
        reqs.append(Request(custom_id=f"cit-{r.query_id}",
                            params=MessageCreateParamsNonStreaming(**cit_params(r.query, mt_cit))))
    batch = B.client().messages.batches.create(requests=reqs)
    json.dump({"batch_id": batch.id, "n": len(reqs), "max_tokens_cit": mt_cit,
               "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}, open(STATE, "w"), indent=2)
    print(f"submitted {batch.id} ({len(reqs)} requests)")


# ------------------------------------------------------------------ collect --

def system_citations(b):
    """Strict single citation from the retrieval system: the highest-ranked
    retrieved section that carries a DEFINES edge (BGE encoder, BNS-scoped)."""
    import faiss
    from sentence_transformers import SentenceTransformer
    syn = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
           "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
           "abduction": "kidnapping", "kidnap": "kidnapping"}

    def norm(q):
        s = q.lower()
        for x, y in syn.items():
            s = s.replace(x, y)
        return s
    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    has_edge = {}
    for _, r in dfn.iterrows():
        has_edge.setdefault(str(r.section_id), set()).add(str(r.offence).lower())
    m = SentenceTransformer("BAAI/bge-base-en-v1.5")
    qe = np.asarray(m.encode([norm(q) for q in b["query"]], batch_size=32)).astype("float32")
    emb = np.asarray(m.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
    _, I = idx.search(qe, 50)
    cites, offs = [], []
    for row in I:
        hit = next((ids[j] for j in row if ids[j] in has_edge), None)
        cites.append(int(hit.split("_")[1]) if hit else -1)
        offs.append(sorted(has_edge[hit])[0] if hit else "none")
    return cites, offs


def cmd_collect(a):
    if not os.path.exists(STATE):
        sys.exit("No batch submitted.")
    import anthropic
    st = json.load(open(STATE))
    c = B.client()
    while True:
        try:
            bt = c.messages.batches.retrieve(st["batch_id"])
        except anthropic.APIConnectionError:
            # Local network drop; the batch keeps running server-side. Retry.
            print(f"[{time.strftime('%H:%M:%S')}] connection error - retrying in 30s")
            time.sleep(30)
            continue
        rc = bt.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {bt.processing_status} processing={rc.processing} "
              f"succeeded={rc.succeeded} errored={rc.errored}")
        if bt.processing_status == "ended":
            break
        if not a.wait:
            sys.exit("Not finished; re-run with --wait.")
        time.sleep(30)

    recs, cost = {}, 0.0
    for res in c.messages.batches.results(st["batch_id"]):
        kind, qid = res.custom_id.split("-", 1)
        r = recs.setdefault(qid, {"query_id": qid})
        if res.result.type != "succeeded":
            r[f"{kind}_status"] = res.result.type
            continue
        msg = res.result.message
        u = msg.usage
        cost += B.usd(u.input_tokens, u.output_tokens, batch=True)
        txt = B.text_of(msg)
        r[f"{kind}_stop"] = msg.stop_reason
        if kind == "cls":
            r["cls_pred"] = "REFUSED" if msg.stop_reason == "refusal" else B.parse_label(txt)
            r["cls_raw"] = txt[:160]
        else:
            try:
                j = json.loads(txt)
            except json.JSONDecodeError:
                j = {}
            r["cit_offence"] = j.get("offence", "PARSE_ERROR")
            r["cit_section"] = j.get("bns_section", -1)
            r["cit_max"] = parse_max(j.get("max_imprisonment", ""))
            r["cit_death"] = j.get("death_possible", None)
            r["cit_raw"] = txt[:200]
    total = B.ledger_add(cost, f"verifiability batch {st['batch_id']}")

    b = load_bench().set_index("query_id")
    df = pd.DataFrame(recs.values()).set_index("query_id").loc[b.index].reset_index()
    df["gold"] = b["gold"].values
    df["grounding"] = b["grounding_section"].values
    df["difficulty"] = b["difficulty"].values
    fam = defining_sections()

    df["cls_ok"] = df["cls_pred"] == df["gold"]
    df["cit_offence_ok"] = df["cit_offence"] == df["gold"]
    df["sec_family_ok"] = [int(s) in fam.get(g, set()) | {int(gr.split("_")[1])}
                           for s, g, gr in zip(df.cit_section, df.gold, df.grounding)]
    df["sec_exact_ok"] = [int(s) == int(gr.split("_")[1]) for s, gr in zip(df.cit_section, df.grounding)]
    df["ipc_confusion"] = [int(s) in IPC.get(g, set()) for s, g in zip(df.cit_section, df.gold)]
    df["nonexistent"] = df["cit_section"].astype(int) > BNS_MAX_SECTION
    def max_allowed(gr, cited, gold):
        # A stated maximum is right if the grounding section prescribes it, or if
        # the model cited a different section of the same offence family that does.
        allowed = set(TIERS.get(gr, set()))
        if int(cited) in fam.get(gold, set()) and f"BNS_{int(cited)}" in TIERS:
            allowed |= TIERS[f"BNS_{int(cited)}"]
        return allowed

    df["max_ok"] = [m in max_allowed(gr, cs, g) for m, gr, cs, g in
                    zip(df.cit_max, df.grounding, df.cit_section, df.gold)]
    df["max_ok_strict"] = [m in TIERS.get(gr, set()) for m, gr in zip(df.cit_max, df.grounding)]
    df["death_scored"] = df["grounding"].isin(DEATH)
    df["death_ok"] = [(bool(d) == DEATH[gr]) if gr in DEATH else np.nan
                      for d, gr in zip(df.cit_death, df.grounding)]

    sc, so = system_citations(b.reset_index())
    df["sys_section"] = sc
    df["sys_offence"] = so
    df["sys_sec_family_ok"] = [s in fam.get(g, set()) for s, g in zip(df.sys_section, df.gold)]
    df.to_csv(os.path.join(RES, "verifiability_outputs.csv"), index=False, encoding="utf-8")

    n = len(df)
    pct = lambda x: 100 * np.nanmean(np.asarray(x, dtype=float))
    print(f"\nbatch cost ${cost:.4f}   ledger total ${total:.4f}")
    print("=" * 84)
    print(f"BNS-Facts-135 v1.1 (n={n}), {MODEL}, effort={EFFORT}")
    print("=" * 84)
    print(f"  offence, classification prompt     {pct(df.cls_ok):6.2f}%")
    print(f"  offence, citation prompt           {pct(df.cit_offence_ok):6.2f}%")
    print(f"  BNS section correct (family)       {pct(df.sec_family_ok):6.2f}%")
    print(f"  BNS section correct (exact)        {pct(df.sec_exact_ok):6.2f}%")
    print(f"  cited the IPC number instead       {pct(df.ipc_confusion):6.2f}%")
    print(f"  cited a nonexistent BNS section    {pct(df.nonexistent):6.2f}%")
    print(f"  maximum imprisonment correct       {pct(df.max_ok):6.2f}%   "
          f"(vs grounding section only: {pct(df.max_ok_strict):.2f}%)")
    print(f"  death-penalty answer correct       {pct(df.death_ok[df.death_scored]):6.2f}%  "
          f"(n={int(df.death_scored.sum())})")
    print(f"\n  retrieval system (BGE, strict single citation):")
    print(f"  BNS section correct (family)       {pct(df.sys_sec_family_ok):6.2f}%   "
          f"IPC citations: 0 by construction")

    print("\nper offence -- LLM section correct (family) / IPC confusion / max correct:")
    g = df.groupby("gold").agg(sec=("sec_family_ok", "mean"), ipc=("ipc_confusion", "mean"),
                               mx=("max_ok", "mean"), sys=("sys_sec_family_ok", "mean")).mul(100).round(1)
    print(g.to_string())

    print("\nmost common LLM citations per offence (section: count):")
    for off, gg in df.groupby("gold"):
        top = gg["cit_section"].value_counts().head(4)
        print(f"  {off:<18} " + ", ".join(f"{int(k)}:{v}" for k, v in top.items()))

    from scipy.stats import binomtest
    a_, b_ = df.sec_family_ok.values, df.sys_sec_family_ok.values
    n10, n01 = int((a_ & ~b_).sum()), int((~a_ & b_).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    print(f"\nsection citation, LLM vs retrieval system: LLM-only={n10} system-only={n01} p={p:.3g}")
    print("wrote results/verifiability_outputs.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    s = sub.add_parser("submit"); s.add_argument("--force", action="store_true")
    c = sub.add_parser("collect"); c.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
