"""Load the readable subset of IEEE-CIS Fraud Detection and produce a stratified sample.

Download `train_transaction.csv` and `train_identity.csv` from
https://www.kaggle.com/competitions/ieee-fraud-detection/data into `data/ieee/`
(Kaggle terms apply; the files are not redistributed here).

Usage:
    python -m eval.load_ieee --data-dir ../data/ieee --n 5000 --fraud-frac 0.3 --out reports/sample.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from app.schemas.transaction import Transaction

TXN_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",  # used only to build a pseudo card_id
    "card4",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C13",
    "C14",
    "D1",
    "D2",
    "D4",
    "D10",
    "D15",
]
ID_COLUMNS = ["TransactionID", "DeviceType", "DeviceInfo"]


def load_readable(data_dir: Path) -> pd.DataFrame:
    txn = pd.read_csv(data_dir / "train_transaction.csv", usecols=TXN_COLUMNS)
    ident_path = data_dir / "train_identity.csv"
    if ident_path.exists():
        ident = pd.read_csv(ident_path, usecols=ID_COLUMNS)
        txn = txn.merge(ident, on="TransactionID", how="left")
    else:
        txn["DeviceType"] = None
        txn["DeviceInfo"] = None
    # Pseudo card key: card1 is a card identifier surrogate in IEEE-CIS.
    txn["card_id"] = "card_" + txn["card1"].astype("Int64").astype(str)
    txn = txn.drop(columns=["card1"])
    # Per-card average amount (a feature the real-time system would get from a store).
    txn["card_avg_amount"] = txn.groupby("card_id")["TransactionAmt"].transform("mean")
    return txn


def stratified_sample(df: pd.DataFrame, n: int, fraud_frac: float, seed: int = 42) -> pd.DataFrame:
    n_fraud = int(n * fraud_frac)
    n_legit = n - n_fraud
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]
    sample = pd.concat(
        [
            fraud.sample(min(n_fraud, len(fraud)), random_state=seed),
            legit.sample(min(n_legit, len(legit)), random_state=seed),
        ]
    ).sample(frac=1, random_state=seed)
    return sample.reset_index(drop=True)


def row_to_transaction(row: pd.Series) -> Transaction:
    payload = {}
    for k, v in row.items():
        if k == "isFraud":
            continue
        if not isinstance(v, (list, dict)) and pd.isna(v):
            continue
        if hasattr(v, "item"):  # numpy scalar -> native Python
            v = v.item()
        payload[k] = v
    payload["transaction_id"] = str(payload.pop("TransactionID"))
    return Transaction.model_validate(payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("../data/ieee"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--fraud-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("eval/reports/sample.parquet"))
    args = ap.parse_args()

    df = load_readable(args.data_dir)
    sample = stratified_sample(df, args.n, args.fraud_frac, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)
    print(f"wrote {len(sample)} rows ({sample['isFraud'].mean():.1%} fraud) to {args.out}")


if __name__ == "__main__":
    main()
