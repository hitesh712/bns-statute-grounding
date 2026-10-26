"""
Phase 7b -- Claude Haiku 4.5 across the three grounding conditions.

Haiku is the small, cheap commercial model a cost-sensitive legal-aid service
would actually deploy. Same citation prompt and schema as the Opus run; the
only differences are the model, no `effort` parameter (Haiku 4.5 rejects it),
and no thinking (Haiku's default).

Conditions: none | rag | rag+graph   (contexts from 27_build_contexts.py)

Usage:
  python 28_haiku_scale.py pilot
  python 28_haiku_scale.py submit
  python 28_haiku_scale.py collect [--wait]
"""

import os
import sys
import json
import time
import argparse
import importlib.util
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import verif_common as C  # noqa: E402

_s = importlib.util.spec_from_file_location("b", os.path.join(HERE, "scripts", "23_llm_baseline.py"))
B = importlib.util.module_from_spec(_s)
_s.loader.exec_module(B)       # client(), ledger, text_of

MODEL = "claude-haiku-4-5"
PRICE_IN, PRICE_OUT = 1.00, 5.00
BUDGET = 1.67
PILOT = os.path.join(C.RES, "haiku_pilot_stats.json")
STATE = os.path.join(C.RES, "haiku_batch_state.json")


def usd(i, o, batch=False):
    return (i * PRICE_IN + o * PRICE_OUT) / 1e6 * (0.5 if batch else 1.0)


def params(query, cond, ctx):
    return dict(model=MODEL, max_tokens=1024, system=C.CIT_SYSTEM,
                output_config={"format": {"type": "json_schema", "schema": C.CIT_SCHEMA}},
                messages=[{"role": "user", "content": C.user_prompt(query, cond, ctx)}])


def cmd_pilot(a):
    import anthropic
    c = B.client()
    b = C.load_bench()
    ctx = C.load_contexts()
    rows, spent = [], 0.0
    for qid in ("F001", "F046"):
        q = b.set_index("query_id").loc[qid, "query"]
        for cond in C.CONDITIONS:
            try:
                m = c.messages.create(**params(q, cond, ctx[qid]))
            except anthropic.BadRequestError as e:
                sys.exit(f"Request rejected: {e.message}")
            u = m.usage
            cost = usd(u.input_tokens, u.output_tokens)
            spent += cost
            rows.append({"cond": cond, "in": u.input_tokens, "out": u.output_tokens})
            print(f"  {qid} {cond:<10} in={u.input_tokens:<5} out={u.output_tokens:<4} "
                  f"${cost:.4f}  {B.text_of(m)[:110]}")
    total = B.ledger_add(spent, f"haiku pilot {len(rows)} calls")
    d = pd.DataFrame(rows).groupby("cond")[["in", "out"]].mean()
    json.dump(d.to_dict(orient="index"), open(PILOT, "w"), indent=2)
    est = sum(135 * usd(r["in"], 1.5 * r["out"], batch=True) for _, r in d.iterrows())
    print(f"\npilot ${spent:.4f} (ledger ${total:.4f}); projected batch ${est:.3f}")


def cmd_submit(a):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    if not os.path.exists(PILOT):
        sys.exit("Run the pilot first.")
    if os.path.exists(STATE) and not a.force:
        sys.exit("Batch already submitted; use collect.")
    st = json.load(open(PILOT))
    est = sum(135 * usd(v["in"], 1.5 * v["out"], batch=True) for v in st.values())
    avail = BUDGET - B.ledger_get()["spent_usd"] - B.SAFETY_MARGIN
    print(f"projected ${est:.3f}  available ${avail:.3f}")
    if est > avail:
        sys.exit("REFUSED: over budget.")
    b = C.load_bench()
    ctx = C.load_contexts()
    reqs = [Request(custom_id=f"{cond.replace('+', 'G')}-{r.query_id}",
                    params=MessageCreateParamsNonStreaming(**params(r.query, cond, ctx[r.query_id])))
            for r in b.itertuples() for cond in C.CONDITIONS]
    bt = B.client().messages.batches.create(requests=reqs)
    json.dump({"batch_id": bt.id, "n": len(reqs),
               "submitted": time.strftime("%Y-%m-%d %H:%M:%S")}, open(STATE, "w"), indent=2)
    print(f"submitted {bt.id} ({len(reqs)} requests)")


def cmd_collect(a):
    import anthropic
    st = json.load(open(STATE))
    c = B.client()
    while True:
        try:
            bt = c.messages.batches.retrieve(st["batch_id"])
        except anthropic.APIConnectionError:
            print("connection error - retrying"); time.sleep(30); continue
        rc = bt.request_counts
        print(f"[{time.strftime('%H:%M:%S')}] {bt.processing_status} processing={rc.processing} "
              f"succeeded={rc.succeeded} errored={rc.errored}")
        if bt.processing_status == "ended":
            break
        if not a.wait:
            sys.exit("Not finished; re-run with --wait.")
        time.sleep(30)

    b = C.load_bench().set_index("query_id")
    rows, cost = [], 0.0
    for res in c.messages.batches.results(st["batch_id"]):
        tag, qid = res.custom_id.split("-", 1)
        cond = tag.replace("G", "+")
        rec = {"model": MODEL, "condition": cond, "query_id": qid,
               "gold": b.loc[qid, "gold"], "grounding": b.loc[qid, "grounding_section"]}
        if res.result.type == "succeeded":
            m = res.result.message
            cost += usd(m.usage.input_tokens, m.usage.output_tokens, batch=True)
            j = C.parse_json(B.text_of(m))
            rec.update(offence=j.get("offence", ""), section=j.get("bns_section", -1),
                       max_imprisonment=j.get("max_imprisonment", ""),
                       death_possible=j.get("death_possible", ""), stop=m.stop_reason)
        else:
            rec.update(offence="", section=-1, max_imprisonment="", death_possible="",
                       stop=res.result.type)
        rows.append(rec)
    total = B.ledger_add(cost, f"haiku batch {st['batch_id']}")

    d = C.score(pd.DataFrame(rows))
    d.to_csv(os.path.join(C.RES, "scale_haiku_outputs.csv"), index=False, encoding="utf-8")
    print(f"\nbatch cost ${cost:.4f}   ledger total ${total:.4f}\n")
    summ = pd.DataFrame({cond: C.summarise(d[d.condition == cond]) for cond in C.CONDITIONS}).T
    print(f"{MODEL} on BNS-Facts-135 v1.1 (%):")
    print(summ.to_string())
    print("\nwrote results/scale_haiku_outputs.csv")


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
