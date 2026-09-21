from app.features.state_builder import bucket_distance, build_state, classify_email_domain
from app.schemas.transaction import Transaction


def test_build_state_drops_none_and_derives_features():
    tx = Transaction(
        TransactionAmt=1250.0,
        ProductCD="W",
        card4="visa",
        card6="credit",
        P_emaildomain="gmail.com",
        R_emaildomain="mailinator.com",
        dist1=812.0,
        TransactionDT=86400 * 3 + 3 * 3600,  # 03:00
        DeviceInfo="SM-G9650",
        D2=0.0,
        D15=0.0,
        card_avg_amount=50.0,
    )
    state = build_state(tx)

    t = state["transaction"]
    assert t["amount_usd"] == 1250.0
    assert t["amount_bucket"] == "large (500-2000)"
    assert t["product"] == "web purchase of physical goods"
    assert t["hour_of_day_utc"] == 3
    assert t["is_night_time"] is True
    assert t["purchaser_email_domain_type"] == "free_webmail"
    assert t["recipient_email_domain_type"] == "disposable"
    assert t["purchaser_recipient_email_mismatch"] is True
    assert t["distance_bucket"] == "far (>500)"
    assert "billing_region_code" not in t  # None pruned

    assert state["velocity"]["card_is_new"] is True
    assert state["velocity"]["amount_vs_card_average_ratio"] == 25.0
    assert state["device"]["device_is_new_for_card"] is True


def test_build_state_minimal():
    state = build_state(Transaction(amount=12.5))
    assert state == {"transaction": {"amount_usd": 12.5, "amount_bucket": "small (10-100)"}}


def test_accepts_field_names_and_aliases():
    a = Transaction(amount=10, card_network="visa")
    b = Transaction(TransactionAmt=10, card4="visa")
    assert a.amount == b.amount and a.card_network == b.card_network


def test_helpers():
    assert classify_email_domain(None) is None
    assert classify_email_domain("acme.com") == "corporate_or_isp"
    assert classify_email_domain("mit.edu") == "institutional"
    assert bucket_distance(3) == "local (<=5)"
    assert bucket_distance(None) is None
