"""Model-version regression check.

Compare metrics of a candidate Jev version against a stored baseline at the SAME
thresholds. Exit non-zero if any metric drifts beyond tolerance, so a `JEV_MODEL`
bump cannot silently shift the operating point.

Usage:
    python -m eval.regression --baseline eval/reports/answers_jev-1.13.0.parquet \
        --candidate eval/reports/answers_jev-1.14.0.parquet --t-low 0.2 --t-high 0.8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from eval.metrics import compute_metrics

DEFAULT_TOLERANCES = {
    "roc_auc": 0.02,  # absolute drop allowed
    "pr_auc": 0.03,
    "step_up_rate": 0.05,  # absolute change either direction
    "fraud_approved_missed": 0.02,  # absolute increase allowed
    "legit_declined_false_positive": 0.01,  # absolute increase allowed
}


def flatten(m: dict) -> dict[str, float | None]:
    return {
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "step_up_rate": m["step_up_rate"],
        "fraud_approved_missed": m["fraud_caught"]["approved_missed"],
        "legit_declined_false_positive": m["legit_friction"]["declined_false_positive"],
        "decline_precision": m["decline"]["precision"],
        "decline_recall": m["decline"]["recall"],
    }


def compare(base: dict, cand: dict, tol: dict[str, float]) -> tuple[list[dict], bool]:
    rows = []
    ok = True
    for k, b in base.items():
        c = cand.get(k)
        if b is None or c is None:
            rows.append({"metric": k, "baseline": b, "candidate": c, "delta": None, "status": "n/a"})
            continue
        delta = c - b
        status = "ok"
        if k in ("roc_auc", "pr_auc") and -delta > tol[k]:
            status = "FAIL"
        elif k == "step_up_rate" and abs(delta) > tol[k]:
            status = "FAIL"
        elif k in ("fraud_approved_missed", "legit_declined_false_positive") and delta > tol[k]:
            status = "FAIL"
        ok &= status == "ok"
        rows.append({"metric": k, "baseline": b, "candidate": c, "delta": delta, "status": status})
    return rows, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--t-low", type=float, default=0.2)
    ap.add_argument("--t-high", type=float, default=0.8)
    ap.add_argument("--c-min", type=float, default=0.6)
    ap.add_argument("--tolerances", type=str, default=None, help="JSON overrides")
    args = ap.parse_args()

    tol = {**DEFAULT_TOLERANCES, **(json.loads(args.tolerances) if args.tolerances else {})}
    base_df = pd.read_parquet(args.baseline)
    cand_df = pd.read_parquet(args.candidate)
    # Compare on the intersection of transaction ids so the sample is identical.
    ids = set(base_df["transaction_id"]) & set(cand_df["transaction_id"])
    base_df = base_df[base_df["transaction_id"].isin(ids)]
    cand_df = cand_df[cand_df["transaction_id"].isin(ids)]

    base = flatten(compute_metrics(base_df, args.t_low, args.t_high, args.c_min))
    cand = flatten(compute_metrics(cand_df, args.t_low, args.t_high, args.c_min))
    rows, ok = compare(base, cand, tol)

    print(f"compared {len(ids)} shared transactions")
    print(f"baseline={sorted(base_df['model'].unique())} candidate={sorted(cand_df['model'].unique())}")
    for r in rows:
        d = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        print(f"  {r['metric']:<32} {str(r['baseline']):<10.10} -> {str(r['candidate']):<10.10} {d:>9}  {r['status']}")
    print("PASS" if ok else "FAIL: candidate drifts beyond tolerance; recalibrate before bumping JEV_MODEL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
