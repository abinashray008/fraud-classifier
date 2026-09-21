"""Load the readable subset of IEEE-CIS Fraud Detection and build an eval sample.

History features use only earlier transactions of the same card. The sample is
then cut on TransactionDT into development, calibration, and an untouched test
period. Sampling keeps each period's fraud rate (about 3.5% on the full train
file: 20,663 / 590,540). An explicit --fraud-frac oversamples and stores
sample_weight so precision, PR-AUC, and step-up rate can be reweighted.

Download `train_transaction.csv` and `train_identity.csv` from
https://www.kaggle.com/competitions/ieee-fraud-detection/data into `data/ieee/`
(Kaggle terms apply; the files are not redistributed here).

Usage:
    python -m eval.load_ieee --data-dir ../data/ieee --n 5000 --out eval/reports/sample.parquet
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.features.history import CARD_HISTORY_KEYS, PriorTxn, compute_card_history_features
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

# Vesta's published train file. Period-level weights use the loaded frame's own
# rate; this constant is the dataset the loader is built for.
IEEE_CIS_TRAIN_TRANSACTIONS = 590_540
IEEE_CIS_TRAIN_FRAUD = 20_663
IEEE_CIS_TRAIN_FRAUD_PREVALENCE = IEEE_CIS_TRAIN_FRAUD / IEEE_CIS_TRAIN_TRANSACTIONS

SPLIT_DEVELOPMENT = "development"
SPLIT_CALIBRATION = "calibration"
SPLIT_TEST = "test"
SPLIT_ORDER = (SPLIT_DEVELOPMENT, SPLIT_CALIBRATION, SPLIT_TEST)


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
    # Time-respecting history from prior rows of the same card — not IEEE C/D columns.
    txn = add_card_history_features(txn)
    return txn


def assign_chronological_splits(
    df: pd.DataFrame,
    *,
    dev_frac: float = 0.50,
    cal_frac: float = 0.25,
) -> pd.DataFrame:
    """Label rows by TransactionDT. Test starts only after calibration ends.

    Cuts are on the time span, not the row count, so a burst of traffic cannot
    pull future transactions into an earlier period. History features must
    already have been computed on the full card timeline.
    """
    if dev_frac <= 0 or cal_frac <= 0 or dev_frac + cal_frac >= 1:
        raise ValueError("Require dev_frac > 0, cal_frac > 0, and dev_frac + cal_frac < 1")
    out = df.copy()
    t0 = float(out["TransactionDT"].min())
    t1 = float(out["TransactionDT"].max())
    span = t1 - t0
    if span <= 0:
        out["split"] = SPLIT_DEVELOPMENT
        return out
    dev_end = t0 + dev_frac * span
    cal_end = t0 + (dev_frac + cal_frac) * span
    dt = out["TransactionDT"].astype(float)
    out["split"] = SPLIT_TEST
    out.loc[dt <= cal_end, "split"] = SPLIT_CALIBRATION
    out.loc[dt <= dev_end, "split"] = SPLIT_DEVELOPMENT
    return out


def sample_period(
    df: pd.DataFrame,
    n: int | None,
    *,
    fraud_frac: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample one time period.

    fraud_frac=None draws a simple random sample, which keeps the period's
    fraud rate. A set fraud_frac oversamples and writes sample_weight equal to
    (class count in the period) / (class count in the sample), so weighted
    metrics recover the period prevalence instead of the sample mix.
    """
    if df.empty:
        return df.copy()
    population = len(df)
    if n is None or n >= population:
        out = df.copy()
        out["sample_weight"] = 1.0
        return out.reset_index(drop=True)
    if n <= 0:
        return df.iloc[0:0].copy()

    y = df["isFraud"].astype(int)
    if fraud_frac is None:
        out = df.sample(n=n, random_state=seed).copy()
        out["sample_weight"] = population / n
        return out.reset_index(drop=True)

    if not 0 < fraud_frac < 1:
        raise ValueError("fraud_frac must be between 0 and 1")
    n_fraud_pop = int((y == 1).sum())
    n_legit_pop = int((y == 0).sum())
    n_fraud = min(int(round(n * fraud_frac)), n_fraud_pop, n)
    n_legit = min(n - n_fraud, n_legit_pop)
    parts = []
    if n_fraud:
        parts.append(df[y == 1].sample(n_fraud, random_state=seed))
    if n_legit:
        parts.append(df[y == 0].sample(n_legit, random_state=seed))
    out = pd.concat(parts).sample(frac=1, random_state=seed).copy()
    sampled_fraud = int((out["isFraud"].astype(int) == 1).sum())
    sampled_legit = int((out["isFraud"].astype(int) == 0).sum())
    fraud_weight = n_fraud_pop / sampled_fraud if sampled_fraud else 1.0
    legit_weight = n_legit_pop / sampled_legit if sampled_legit else 1.0
    out["sample_weight"] = np.where(out["isFraud"].astype(int) == 1, fraud_weight, legit_weight)
    return out.reset_index(drop=True)


def _allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    raw = {s: n * sizes[s] / total for s in SPLIT_ORDER}
    alloc = {s: min(int(raw[s]), sizes[s]) for s in SPLIT_ORDER}
    remainder = n - sum(alloc.values())
    order = sorted(SPLIT_ORDER, key=lambda s: raw[s] - alloc[s], reverse=True)
    for s in order:
        if remainder <= 0:
            break
        room = sizes[s] - alloc[s]
        take = min(room, remainder)
        alloc[s] += take
        remainder -= take
    return alloc


def build_eval_samples(
    df: pd.DataFrame,
    n: int,
    *,
    fraud_frac: float | None = None,
    seed: int = 42,
    dev_frac: float = 0.50,
    cal_frac: float = 0.25,
) -> pd.DataFrame:
    """Chronological development / calibration / test samples.

    `df` must already carry time-respecting history features. Test rows are
    strictly later than calibration rows, which are strictly later than
    development rows, unless every timestamp is identical.
    """
    labeled = assign_chronological_splits(df, dev_frac=dev_frac, cal_frac=cal_frac)
    sizes = {s: int((labeled["split"] == s).sum()) for s in SPLIT_ORDER}
    alloc = _allocate(sizes, n)
    parts = [
        sample_period(labeled[labeled["split"] == s], alloc[s], fraud_frac=fraud_frac, seed=seed)
        for s in SPLIT_ORDER
        if alloc[s] > 0
    ]
    if not parts:
        return labeled.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _none_if_nan(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        return v
    return v


def add_card_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-card history features using only earlier TransactionDT rows.

    C/D columns are left untouched (opaque). Features here are computed from
    TransactionAmt, TransactionDT, DeviceInfo, and P_emaildomain.
    """
    df = df.sort_values(["card_id", "TransactionDT"]).copy()
    feats_by_index: dict[object, dict] = {}
    has_device = "DeviceInfo" in df.columns
    has_email = "P_emaildomain" in df.columns

    for _, group in df.groupby("card_id", sort=False):
        prior: list[PriorTxn] = []
        for rec in group.itertuples():
            at = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(seconds=float(rec.TransactionDT))
            device = _none_if_nan(getattr(rec, "DeviceInfo", None)) if has_device else None
            if device is not None:
                device = str(device)
            email = _none_if_nan(getattr(rec, "P_emaildomain", None)) if has_email else None
            if email is not None:
                email = str(email)
            amount = float(rec.TransactionAmt)
            feats_by_index[rec.Index] = compute_card_history_features(
                prior=prior,
                amount=amount,
                at=at,
                device_info=device,
                email_domain=email,
            )
            prior.append(PriorTxn(amount=amount, timestamp=at, device_info=device, email_domain=email))

    feat_df = pd.DataFrame.from_dict(feats_by_index, orient="index")
    return df.join(feat_df)


def row_history(row: pd.Series) -> dict:
    """Extract history-feature columns from a sample row for `build_state`."""
    hist = {}
    for k in CARD_HISTORY_KEYS:
        if k not in row.index:
            continue
        v = row[k]
        if not isinstance(v, (list, dict)) and pd.isna(v):
            continue
        if hasattr(v, "item"):
            v = v.item()
        hist[k] = v
    return hist


def row_to_transaction(row: pd.Series) -> Transaction:
    payload = {}
    for k, v in row.items():
        if k == "isFraud" or k in CARD_HISTORY_KEYS:
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
    ap.add_argument(
        "--fraud-frac",
        type=float,
        default=None,
        help="Oversample fraud to this fraction. Default keeps each period's prevalence.",
    )
    ap.add_argument("--dev-frac", type=float, default=0.50)
    ap.add_argument("--cal-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("eval/reports/sample.parquet"))
    args = ap.parse_args()

    df = load_readable(args.data_dir)
    population_rate = float(df["isFraud"].mean())
    sample = build_eval_samples(
        df,
        args.n,
        fraud_frac=args.fraud_frac,
        seed=args.seed,
        dev_frac=args.dev_frac,
        cal_frac=args.cal_frac,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(args.out, index=False)
    print(
        f"wrote {len(sample)} rows to {args.out} "
        f"(dataset fraud {population_rate:.2%}; IEEE-CIS train {IEEE_CIS_TRAIN_FRAUD_PREVALENCE:.2%})"
    )
    for split in SPLIT_ORDER:
        part = sample[sample["split"] == split]
        if part.empty:
            print(f"  {split}: 0 rows")
            continue
        weighted = float(np.average(part["isFraud"].astype(float), weights=part["sample_weight"]))
        print(
            f"  {split}: {len(part)} rows, sample fraud {part['isFraud'].mean():.2%}, "
            f"weighted fraud {weighted:.2%}, "
            f"TransactionDT {int(part['TransactionDT'].min())}–{int(part['TransactionDT'].max())}"
        )


if __name__ == "__main__":
    main()
