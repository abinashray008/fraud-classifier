"""Export synthetic legitimate control-failure cases for the existing Jev eval runner.

This is a targeted prompt regression set, not a representative calibration sample.
The runner calls the classifier directly, including cases live hard controls would
decline without scoring. Expected issuer declines are separate from fraud labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures" / "authorization_counterexamples.json"


def build_sample() -> pd.DataFrame:
    fixture = json.loads(FIXTURES.read_text())
    return pd.DataFrame(
        [
            {
                **fixture["base_transaction"],
                **case["transaction"],
                **fixture["history"],
                "TransactionID": case["id"],
                "isFraud": case["is_fraud"],
                "expected_hard_decline": case["expected_hard_decline"],
                "scenario_explanation": case["explanation"],
                "split": "test",
            }
            for case in fixture["cases"]
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval/reports/authorization_counterexamples.parquet"))
    args = parser.parse_args()
    sample = build_sample()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)
    print(f"Wrote {len(sample)} synthetic counterexamples to {args.out}")


if __name__ == "__main__":
    main()
