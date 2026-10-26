"""
Phase 5 -- zero-shot LLM baseline (Claude Opus 5.5), budget-guarded.

The question a reviewer will ask: does a frontier LLM, given only the nine
offence names and no retrieval or graph, already solve BNS-Facts-135?

Same closed label set as the trivial embedding baseline, raw queries (the
hand-built synonym map is not applied -- an LLM does not need it, and it only
ever rewrites toward label words). No definitions, no statute text.

Built to fit a small API balance:
  pilot    9 synchronous calls (one per category) to MEASURE real cost per call
  submit   the full run via the Message Batches API (50% off), refused if the
           measured projection exceeds the remaining budget
  collect  fetch results, score, and test against the retrieval systems

Server-side refusal fallbacks are deliberately NOT used: a fallback would answer
some queries with a different model and contaminate an "Opus 5.5 zero-shot"
baseline. A refusal is recorded as a refusal and reported. (The parameter is
also rejected by the Batches API.)

The API key is read from .env in the repository root and never printed.

Usage:
  python 23_llm_baseline.py pilot
  python 23_llm_baseline.py submit [--with-original] [--budget 1.67]
  python 23_llm_baseline.py collect
"""

import os
import re
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
OLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "original_data")
ENV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")

MODEL = "claude-opus-5-5"
EFFORT = "medium"                     # Opus 5.5 default, set explicitly for reporting
PRICE_IN, PRICE_OUT = 4.00, 20.00     # USD per million tokens, claude-opus-5-5
BATCH_DISCOUNT = 0.5
SAFETY_MARGIN = 0.15                  # USD kept in reserve

LABELS = ["cheating", "extortion", "forgery", "murder", "rape", "robbery",
          "sexual harassment", "stalking", "theft"]
SYSTEM = ("You classify short descriptions of situations under Indian criminal law "
          "(the Bharatiya Nyaya Sanhita, 2023) into exactly one offence category.")

LEDGER = os.path.join(RES, "llm_ledger.json")
PILOT = os.path.join(RES, "llm_pilot_stats.json")
STATE = os.path.join(RES, "llm_batch_state.json")


# ---------------------------------------------------------------- utilities --

def load_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if not os.path.exists(ENV):
        sys.exit(f"No API key. Create {ENV} containing one line:\n"
                 f"  ANTHROPIC_API_KEY=sk-ant-...")
    raw = open(ENV, "rb").read()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    found = False
    for line in text.splitlines():
        line = line.strip()
        val = line.split("=", 1)[1].strip().strip('"').strip("'") if "=" in line else ""
        if line.startswith("ANTHROPIC_API_KEY="):
            if not val.startswith("sk-ant-"):
                sys.exit("The key in .env does not look like an Anthropic key (sk-ant-...).")
            os.environ["ANTHROPIC_API_KEY"] = val
            found = True
        elif line.startswith("ANTHROPIC_WORKSPACE_ID=") and val:
            os.environ["ANTHROPIC_WORKSPACE_ID"] = val
    if not found:
        sys.exit(f"No 'ANTHROPIC_API_KEY=' line found in {ENV}")


def client():
    import anthropic
    load_key()
    # An organisation-level key that is not scoped to a workspace must name one
    # on every request via the anthropic-workspace-id header.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    return anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)


def user_msg(q):
    return (f"Situation: {q}\n\n"
            f"Which ONE of these offence categories best fits? {', '.join(LABELS)}\n\n"
            "Reply with only the category name, exactly as written above.")


def request_params(q, max_tokens):
    # thinking omitted: on Opus 5.5 it is adaptive by default and cannot be disabled
    return dict(model=MODEL, max_tokens=max_tokens, system=SYSTEM,
                output_config={"effort": EFFORT},
                messages=[{"role": "user", "content": user_msg(q)}])


def parse_label(text):
    t = " ".join(re.sub(r"[^a-z ]+", " ", str(text).lower()).split())
    if t in LABELS:
        return t
    hits = []
    for lab in LABELS:
        m = re.search(r"\b" + re.escape(lab) + r"\b", t)
        if m:
            hits.append((m.start(), -len(lab), lab))
    return sorted(hits)[0][2] if hits else "UNPARSED"


def text_of(msg):
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def usd(inp, out, batch=False):
    c = (inp * PRICE_IN + out * PRICE_OUT) / 1e6
    return c * (BATCH_DISCOUNT if batch else 1.0)


def ledger_get():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"spent_usd": 0.0, "entries": []}


def ledger_add(amount, note):
    lg = ledger_get()
    lg["spent_usd"] = round(lg["spent_usd"] + amount, 6)
    lg["entries"].append({"usd": round(amount, 6), "note": note,
                          "time": time.strftime("%Y-%m-%d %H:%M:%S")})
    json.dump(lg, open(LEDGER, "w"), indent=2)
    return lg["spent_usd"]


def load_benchmarks():
    f = pd.read_csv(os.path.join(DATA, "bns_facts_135.csv"))
    f["query"] = f["query"].astype(str).str.strip()
    f["gold"] = f["expected_offence"].astype(str).str.strip().str.lower()
    o = pd.read_csv(os.path.join(OLD, "evaluation_queries_v2.csv"))
    o.columns = o.columns.str.strip()
    o["query"] = o["query"].astype(str).str.strip()
    o["gold"] = o["expected_offence"].astype(str).str.strip().str.lower()
    o["query_id"] = [f"O{i:03d}" for i in range(1, len(o) + 1)]
    return f, o


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    return n10, n01, (1.0 if n == 0 else binomtest(n10, n, 0.5).pvalue)


# -------------------------------------------------------------------- pilot --

def cmd_pilot(args):
    import anthropic
    os.makedirs(RES, exist_ok=True)
    c = client()
    facts, _ = load_benchmarks()
    sample = facts.groupby("gold", sort=True).head(1).reset_index(drop=True)

    print(f"PILOT: {len(sample)} synchronous calls to {MODEL} (effort={EFFORT})\n")
    rows, spent = [], 0.0
    for _, r in sample.iterrows():
        try:
            msg = c.messages.create(**request_params(r["query"], max_tokens=4000))
        except anthropic.AuthenticationError:
            sys.exit("Authentication failed - the key in .env is invalid or revoked.")
        except anthropic.PermissionDeniedError as e:
            sys.exit(f"Permission denied: {e.message}")
        except anthropic.BadRequestError as e:
            sys.exit(f"Request rejected: {e.message}")
        except anthropic.RateLimitError:
            sys.exit("Rate limited - wait a minute and re-run the pilot.")
        except anthropic.APIConnectionError:
            sys.exit("Network error reaching the API.")
        u = msg.usage
        cost = usd(u.input_tokens, u.output_tokens)
        spent += cost
        txt = text_of(msg)
        pred = "REFUSED" if msg.stop_reason == "refusal" else parse_label(txt)
        rows.append({"query_id": r["query_id"], "gold": r["gold"], "pred": pred,
                     "stop": msg.stop_reason, "in": u.input_tokens,
                     "out": u.output_tokens, "usd": cost, "raw": txt[:60]})
        print(f"  {r['query_id']} gold={r['gold']:<18} pred={pred:<18} "
              f"in={u.input_tokens:<4} out={u.output_tokens:<5} ${cost:.4f}  "
              f"stop={msg.stop_reason}")

    total = ledger_add(spent, f"pilot {len(rows)} sync calls")
    d = pd.DataFrame(rows)
    stats = {"n": len(d), "mean_in": float(d["in"].mean()),
             "mean_out": float(d["out"].mean()), "max_out": int(d["out"].max()),
             "mean_usd_sync": float(d["usd"].mean()), "pilot_usd": spent,
             "correct": int((d.pred == d.gold).sum())}
    json.dump(stats, open(PILOT, "w"), indent=2)

    def proj(n, out_tokens):
        return n * usd(stats["mean_in"], out_tokens, batch=True)

    print(f"\npilot cost              : ${spent:.4f}   (ledger total ${total:.4f})")
    print(f"pilot accuracy          : {stats['correct']}/{stats['n']}")
    print(f"mean tokens / call      : {stats['mean_in']:.0f} in, {stats['mean_out']:.0f} out "
          f"(max {stats['max_out']})")
    print("\nPROJECTED BATCH COST (50% off):")
    for name, n in (("BNS-Facts-135", 135), ("original 110", 110), ("both", 245)):
        print(f"  {name:<15} expected ${proj(n, stats['mean_out']):.3f}   "
              f"pessimistic ${proj(n, 1.5 * stats['mean_out']):.3f}")


# ------------------------------------------------------------------- submit --

def cmd_submit(args):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if not os.path.exists(PILOT):
        sys.exit("Run the pilot first - the budget check needs measured costs.")
    if os.path.exists(STATE) and not args.force:
        sys.exit(f"A batch was already submitted ({STATE}). Use 'collect', or --force.")

    stats = json.load(open(PILOT))
    facts, orig = load_benchmarks()
    jobs = [("facts", r.query_id, r.query) for r in facts.itertuples()]
    if args.with_original:
        jobs += [("orig", r.query_id, r.query) for r in orig.itertuples()]

    pessimistic = len(jobs) * usd(stats["mean_in"], 1.5 * stats["mean_out"], batch=True)
    spent = ledger_get()["spent_usd"]
    remaining = args.budget - spent - SAFETY_MARGIN
    print(f"jobs: {len(jobs)}   pessimistic cost ${pessimistic:.3f}   "
          f"budget ${args.budget:.2f} - spent ${spent:.3f} - reserve ${SAFETY_MARGIN:.2f} "
          f"= ${remaining:.3f} available")
    if pessimistic > remaining:
        sys.exit("REFUSED: projected cost exceeds remaining budget. "
                 "Drop --with-original, or top up.")

    max_tokens = int(min(8000, max(1500, 3 * stats["max_out"])))
    reqs = [Request(custom_id=f"{tag}-{qid}",
                    params=MessageCreateParamsNonStreaming(**request_params(q, max_tokens)))
            for tag, qid, q in jobs]

    c = client()
    batch = c.messages.batches.create(requests=reqs)
    json.dump({"batch_id": batch.id, "n": len(reqs), "max_tokens": max_tokens,
               "model": MODEL, "effort": EFFORT, "with_original": bool(args.with_original),
               "submitted": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(STATE, "w"), indent=2)
    print(f"submitted batch {batch.id}  ({len(reqs)} requests, max_tokens={max_tokens})")
    print(f"status: {batch.processing_status}.  Next: python 23_llm_baseline.py collect")


# ------------------------------------------------------------------ collect --

def hybrid_outcomes(facts):
    """Recompute hybrid (BM25 + BGE, RRF) + graph per-query outcomes, locally."""
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    syn = {"sexual assault": "rape", "homicide": "murder", "killing a person": "murder",
           "forging documents": "forgery", "fake document": "forgery", "fraud": "cheating",
           "abduction": "kidnapping", "kidnap": "kidnapping"}

    def norm(q):
        s = q.lower()
        for a, b in syn.items():
            s = s.replace(a, b)
        return s

    sec = pd.read_csv(os.path.join(DATA, "bns_act_titled.csv"))
    sec["body"] = sec["body"].fillna("").astype(str)
    ids = sec["section_id"].astype(str).values
    dfn = pd.read_csv(os.path.join(DATA, "graph_defines.csv"))
    defines = {}
    for _, r in dfn.iterrows():
        defines.setdefault(str(r.section_id).strip(), set()).add(str(r.offence).strip().lower())

    qs = [norm(q) for q in facts["query"]]
    tok = lambda s: re.findall(r"[a-z]+", str(s).lower())
    bm = BM25Okapi([tok(t) for t in sec["body"]])
    bm_rank = [[ids[j] for j in np.argsort(-bm.get_scores(tok(q)))[:20]] for q in qs]

    m = SentenceTransformer("BAAI/bge-base-en-v1.5")
    qe = np.asarray(m.encode(qs, batch_size=32)).astype("float32")
    emb = np.asarray(m.encode(sec["body"].tolist(), batch_size=16)).astype("float32")
    idx = faiss.IndexFlatL2(emb.shape[1]); idx.add(emb)
    _, I = idx.search(qe, 20)
    dn_rank = [[ids[j] for j in row] for row in I]

    def rrf(a, b, k=60):
        s = {}
        for rl in (a, b):
            for p, sid in enumerate(rl):
                s[sid] = s.get(sid, 0) + 1 / (k + p + 1)
        return [x for x, _ in sorted(s.items(), key=lambda kv: -kv[1])]

    hy = [rrf(bm_rank[i], dn_rank[i]) for i in range(len(qs))]
    gold = facts["gold"].tolist()
    out = {}
    for k in (5, 10, 20):
        out[f"hybrid|graph_k{k}"] = np.array([gold[i] in set().union(
            *([defines.get(s, set()) for s in r[:k]] or [set()])) for i, r in enumerate(hy)])
    return out


def cmd_collect(args):
    if not os.path.exists(STATE):
        sys.exit("No batch submitted yet.")
    st = json.load(open(STATE))
    c = client()

    while True:
        b = c.messages.batches.retrieve(st["batch_id"])
        rc = b.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {b.processing_status}  processing={rc.processing} "
              f"succeeded={rc.succeeded} errored={rc.errored}")
        if b.processing_status == "ended":
            break
        if not args.wait:
            sys.exit("Not finished yet. Re-run collect later, or pass --wait.")
        time.sleep(30)

    rows, cost = [], 0.0
    for res in c.messages.batches.results(st["batch_id"]):
        tag, qid = res.custom_id.split("-", 1)
        rec = {"bench": tag, "query_id": qid, "result": res.result.type,
               "stop": "", "raw": "", "pred": "ERROR", "in": 0, "out": 0}
        if res.result.type == "succeeded":
            msg = res.result.message
            u = msg.usage
            rec.update(stop=msg.stop_reason, raw=text_of(msg)[:200],
                       **{"in": u.input_tokens, "out": u.output_tokens})
            rec["pred"] = "REFUSED" if msg.stop_reason == "refusal" else parse_label(rec["raw"])
            cost += usd(u.input_tokens, u.output_tokens, batch=True)
        rows.append(rec)
    total = ledger_add(cost, f"batch {st['batch_id']} ({len(rows)} results)")

    out = pd.DataFrame(rows)
    facts, orig = load_benchmarks()
    gold = {**dict(zip(facts.query_id, facts.gold)), **dict(zip(orig.query_id, orig.gold))}
    out["gold"] = out["query_id"].map(gold)
    out["correct"] = out["pred"] == out["gold"]
    out.to_csv(os.path.join(RES, "llm_baseline_outputs.csv"), index=False, encoding="utf-8")
    with open(os.path.join(RES, "llm_baseline_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(f"model={MODEL} effort={EFFORT} max_tokens={st['max_tokens']} thinking=adaptive(default)\n\n")
        f.write(f"SYSTEM:\n{SYSTEM}\n\nUSER (template):\n{user_msg('{query}')}\n")

    print(f"\nbatch cost ${cost:.4f}   ledger total ${total:.4f}")
    print("=" * 80)
    for tag, name in (("facts", "BNS-Facts-135"), ("orig", "ORIGINAL 110")):
        d = out[out.bench == tag]
        if not len(d):
            continue
        print(f"\n{name}: {MODEL} zero-shot = {d.correct.sum()}/{len(d)} "
              f"= {100*d.correct.mean():.2f}%   refused={(d.pred=='REFUSED').sum()} "
              f"unparsed={(d.pred=='UNPARSED').sum()} errored={(d.pred=='ERROR').sum()}")

    d = out[out.bench == "facts"].set_index("query_id").loc[facts.query_id]
    llm = d["correct"].values
    print("\nper category (BNS-Facts-135):")
    print(d.groupby("gold")["correct"].mean().mul(100).round(1).to_string())

    pq = pd.read_csv(os.path.join(RES, "per_query_outcomes.csv")).set_index("query_id").loc[facts.query_id]
    comps = {c_: pq[c_].astype(str).str.lower().eq("true").values for c_ in
             ["BGE-base|trivial", "BGE-base|graph_k5", "BGE-base|graph_k10",
              "BGE-base|graph_k20", "mpnet-base|graph_k20"] if c_ in pq.columns}
    comps.update(hybrid_outcomes(facts))

    print("\nLLM vs retrieval systems on BNS-Facts-135 (exact McNemar):")
    print(f"  {'system':<24}{'acc':>8}{'LLM-only':>10}{'sys-only':>10}{'p':>11}")
    sig_rows = []
    for name, ok in comps.items():
        n10, n01, p = mcnemar(llm, ok)
        print(f"  {name:<24}{100*ok.mean():>7.2f}%{n10:>10}{n01:>10}{p:>11.3g}")
        sig_rows.append({"system": name, "accuracy": round(100 * ok.mean(), 2),
                         "llm_accuracy": round(100 * llm.mean(), 2),
                         "llm_only": n10, "system_only": n01, "p": p})
    pd.DataFrame(sig_rows).to_csv(os.path.join(RES, "llm_vs_systems.csv"), index=False)
    print(f"\nwrote results/llm_baseline_outputs.csv, llm_vs_systems.csv, llm_baseline_prompt.txt")


# --------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pilot")
    s = sub.add_parser("submit")
    s.add_argument("--with-original", action="store_true")
    s.add_argument("--budget", type=float, default=1.67)
    s.add_argument("--force", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    {"pilot": cmd_pilot, "submit": cmd_submit, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    main()
