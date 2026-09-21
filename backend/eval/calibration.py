"""Probability calibration of `is_fraud.noul`.

This is not action-threshold selection. `eval.calibrate` chooses `t_low` and
`t_high`. This module asks whether a stated fraud probability matches the
observed fraud rate: Brier score, and a reliability diagram (predicted
probability vs. fraction of fraud in each bin).

Usage:
    python -m eval.calibration --answers eval/reports/answers_jev-1.13.0.parquet \
        --out eval/reports/reliability.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def brier_score(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    """Mean squared error of the probability. 0 is perfect; 0.25 is a constant 0.5 on balanced labels."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    se = (p - y) ** 2
    if len(se) == 0:
        return 0.0
    if w is None:
        return float(se.mean())
    return float(np.average(se, weights=w))


def reliability_table(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray | None = None,
    n_bins: int = 10,
) -> list[dict]:
    """Equal-width bins of predicted probability vs. observed fraud rate."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if w is not None:
        w = np.asarray(w, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    for i in range(n_bins):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if w is None:
            weight = float(n)
            mean_p = float(p[mask].mean()) if n else None
            frac = float(y[mask].mean()) if n else None
        else:
            weight = float(w[mask].sum()) if n else 0.0
            mean_p = float(np.average(p[mask], weights=w[mask])) if weight else None
            frac = float(np.average(y[mask], weights=w[mask])) if weight else None
        rows.append(
            {
                "bin_lo": round(lo, 4),
                "bin_hi": round(hi, 4),
                "n": n,
                "weight": weight,
                "mean_predicted": mean_p,
                "fraction_positive": frac,
            }
        )
    return rows


def expected_calibration_error(rows: list[dict]) -> float | None:
    """Weight each bin's |predicted − observed| by its share of the sample."""
    total = sum(float(r["weight"]) for r in rows)
    if not total:
        return None
    err = 0.0
    for row in rows:
        if not row["weight"] or row["mean_predicted"] is None:
            continue
        err += float(row["weight"]) * abs(float(row["mean_predicted"]) - float(row["fraction_positive"]))
    return float(err / total)


def probability_calibration(df: pd.DataFrame, n_bins: int = 10) -> dict:
    from eval.metrics import sample_weights

    y = df["label"].astype(int).to_numpy()
    p = df["p_fraud"].to_numpy(dtype=float)
    w = sample_weights(df)
    rows = reliability_table(y, p, w, n_bins=n_bins)
    return {
        "note": (
            "Compares is_fraud.noul (p_fraud) to observed labels. "
            "This measures probability calibration. Action thresholds are selected by eval.calibrate."
        ),
        "n": int(len(df)),
        "brier": brier_score(y, p, w),
        "ece": expected_calibration_error(rows),
        "reliability": rows,
    }


def reliability_svg(rows: list[dict]) -> str:
    """Reliability diagram: observed fraud rate against mean predicted probability."""
    width, height = 440, 400
    left, right, top, bottom = 52, 24, 24, 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xy(px: float, py: float) -> tuple[float, float]:
        return left + px * plot_w, top + (1 - py) * plot_h

    x0, y0 = xy(0, 0)
    x1, y1 = xy(1, 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#b0b0b0" stroke-dasharray="4 3"/>',
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y0:.1f}" stroke="#333"/>',
        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x0:.1f}" y2="{y1:.1f}" stroke="#333"/>',
    ]
    occupied = [r for r in rows if r["n"] and r["mean_predicted"] is not None]
    max_n = max((r["n"] for r in occupied), default=1)
    for row in occupied:
        cx, cy = xy(float(row["mean_predicted"]), float(row["fraction_positive"]))
        radius = 4 + 8 * (row["n"] / max_n) ** 0.5
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="#1d4e89" fill-opacity="0.85"/>')
    for tick in (0, 0.5, 1):
        tx, ty = xy(tick, 0)
        parts.append(
            f'<text x="{tx:.1f}" y="{ty + 16:.1f}" text-anchor="middle" font-size="11" fill="#333">{tick:g}</text>'
        )
        lx, ly = xy(0, tick)
        parts.append(
            f'<text x="{lx - 8:.1f}" y="{ly + 4:.1f}" text-anchor="end" font-size="11" fill="#333">{tick:g}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" font-size="12" fill="#222">'
        "mean predicted fraud probability</text>"
    )
    parts.append(
        f'<text x="16" y="{top + plot_h / 2:.1f}" text-anchor="middle" font-size="12" fill="#222" '
        f'transform="rotate(-90 16 {top + plot_h / 2:.1f})">observed fraud rate</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Brier score and reliability diagram for is_fraud.noul")
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--split", default="test", help="chronological split to score (default: held-out test)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None, help="write the reliability diagram SVG here")
    args = ap.parse_args()

    from eval.metrics import frame_for

    df = frame_for(pd.read_parquet(args.answers), args.split)
    report = probability_calibration(df, n_bins=args.bins)
    report["split"] = args.split
    report["model"] = sorted(df["model"].unique().tolist()) if "model" in df.columns else []
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(reliability_svg(report["reliability"]))


if __name__ == "__main__":
    main()
