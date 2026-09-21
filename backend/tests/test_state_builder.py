from datetime import UTC, datetime, timedelta

from app.agent.feature_store import DEMO_TRUSTED_DEVICE, FeatureStore
from app.features.history import PriorTxn, compute_card_history_features
from app.features.state_builder import bucket_dist1, build_state, classify_email_domain
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
        card_avg_amount=50.0,
        card_known_devices=["SM-G950F Build/R16NW"],
    )
    state = build_state(tx)

    t = state["transaction"]
    assert t["amount_usd"] == 1250.0
    assert t["amount_bucket"] == "large (500-2000 USD)"
    assert "product_code" not in t
    assert t["hour_of_day_utc"] == 3
    assert t["is_night_time"] is True
    assert t["purchaser_email_domain_type"] == "free_webmail"
    assert t["recipient_email_domain_type"] == "disposable"
    assert t["purchaser_recipient_email_mismatch"] is True
    assert t["dist1"] == 812.0
    assert t["dist1_bucket"] == ">500"
    assert "addr1" not in t  # None pruned
    assert state["card_history"]["amount_vs_mean_prior_ratio"] == 25.0
    assert state["device"]["device_is_new_for_card"] is True


def test_product_code_is_not_sent_to_jev():
    for code in ("W", "C", "H", "S", "R"):
        state = build_state(Transaction(amount=10, ProductCD=code))
        assert "product_code" not in state["transaction"]
        blob = str(state)
        assert "hotel" not in blob.lower()
        assert "travel" not in blob.lower()
        assert "subscription" not in blob.lower()
        assert "recurring" not in blob.lower()


def test_masked_c_d_columns_are_not_interpreted_in_state():
    tx = Transaction(amount=40, C14=1.0, D15=0.0, C1=9.0, D2=0.0, ProductCD="H")
    state = build_state(tx)
    blob = str(state)
    assert "device_txn" not in blob
    assert "days_since_device" not in blob
    assert "transactions_from_this_device" not in blob
    assert "hotel" not in blob.lower()
    assert "card_is_new" not in blob  # D2 is not treated as card age
    assert "C14" not in blob and "c14" not in blob
    assert "product_code" not in blob


def test_history_features_have_units_and_windows():
    at = datetime(2026, 1, 10, tzinfo=UTC)
    prior = [
        PriorTxn(amount=40.0, timestamp=at - timedelta(days=10), device_info="SM-A Build/1", email_domain="gmail.com"),
        PriorTxn(amount=40.0, timestamp=at - timedelta(hours=2), device_info="SM-A Build/1", email_domain="gmail.com"),
        PriorTxn(
            amount=40.0, timestamp=at - timedelta(minutes=30), device_info="SM-B Build/2", email_domain="gmail.com"
        ),
    ]
    hist = compute_card_history_features(
        prior=prior, amount=40.0, at=at, device_info="SM-B Build/2", email_domain="gmail.com"
    )
    assert hist["history_found"] is True
    assert hist["prior_transaction_count"] == 3
    assert hist["days_since_first_transaction"] == 10.0
    assert hist["transactions_last_1h"] == 1
    assert hist["transactions_last_24h"] == 2
    assert hist["transactions_last_7d"] == 2
    assert hist["amount_usd_last_24h"] == 80.0
    assert hist["distinct_devices_last_24h"] == 2
    assert hist["matching_amount_count_last_24h"] == 2
    assert hist["device_seen_before_on_this_card"] is True
    assert hist["email_domain_seen_before_on_this_card"] is True

    state = build_state(Transaction(amount=40.0, DeviceInfo="SM-B Build/2"), history=hist)
    ch = state["card_history"]
    assert ch["prior_transaction_count"] == 3
    assert ch["transactions_last_24h"] == 2
    assert ch["amount_usd_last_24h"] == 80.0
    assert state["device"]["device_is_new_for_card"] is False
    assert state["device"]["device_identifier_kind"] == "fingerprint"


def test_generic_device_string_is_not_a_fingerprint():
    at = datetime(2026, 1, 10, tzinfo=UTC)
    prior = [
        PriorTxn(amount=25.0, timestamp=at - timedelta(days=3), device_info="Windows", outcome="approved"),
        PriorTxn(amount=30.0, timestamp=at - timedelta(hours=1), device_info="Windows", outcome="approved"),
    ]
    hist = compute_card_history_features(
        prior=prior, amount=40.0, at=at, device_info="Windows", email_domain="outlook.com"
    )
    assert hist["device_seen_before_on_this_card"] is None
    assert hist["distinct_devices_last_24h"] is None
    assert hist["mean_amount_usd_prior"] == 27.5
    assert hist["prior_transaction_count"] == 2

    state = build_state(Transaction(amount=40.0, DeviceInfo="Windows"), history=hist)
    assert state["device"]["device_identifier_kind"] == "description"
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"
    assert state["device"]["device_is_new_for_card"] == "unknown"
    assert state["card_history"]["mean_amount_usd_prior"] == 27.5


def test_build_state_minimal():
    state = build_state(Transaction(amount=12.5))
    assert state == {"transaction": {"amount_usd": 12.5, "amount_bucket": "small (10-100 USD)"}}


def test_accepts_field_names_and_aliases():
    a = Transaction(amount=10, card_network="visa", dist1=3)
    b = Transaction(TransactionAmt=10, card4="visa", distance_billing_to_purchase=3)
    assert a.amount == b.amount and a.card_network == b.card_network
    assert a.dist1 == b.dist1 == 3
    assert Transaction(amount=1, C14=2.0).c14 == 2.0
    assert Transaction(amount=1, D15=0.0).d15 == 0.0


def test_helpers():
    assert classify_email_domain(None) is None
    assert classify_email_domain("acme.com") == "corporate_or_isp"
    assert classify_email_domain("mit.edu") == "institutional"
    assert bucket_dist1(3) == "<=5"
    assert bucket_dist1(None) is None


def test_feature_store_history_for_seeded_card():
    store = FeatureStore()
    store.seed_demo()
    tx = Transaction(
        amount=62.5,
        card_id="card_good_001",
        DeviceInfo=DEMO_TRUSTED_DEVICE,
        P_emaildomain="gmail.com",
    )
    hist = store.history_features(tx)
    assert hist["history_found"] is True
    assert hist["prior_transaction_count"] == 12
    assert hist["device_seen_before_on_this_card"] is True
    assert hist["mean_amount_usd_prior"] == 62.5
    assert hist["trusted_device_ids"] == [DEMO_TRUSTED_DEVICE]
    assert hist["current_device_is_trusted"] is True
    assert hist["confirmed_outcomes"] == {"approved": 12, "declined": 0, "chargeback": 0}
    assert len(hist["recent_attempts"]) == 5
    assert hist["recent_attempts"][0]["outcome"] == "approved"
    state = build_state(tx, history=hist)
    assert state["card_history"]["card_is_new"] is False
    assert state["card_history"]["mean_amount_usd_prior"] == 62.5
    assert state["device"]["device_is_new_for_card"] is False
    assert state["device"]["current_device_is_trusted"] is True


def test_windows_on_established_card_does_not_match_a_device():
    store = FeatureStore()
    store.seed_demo()
    tx = Transaction(amount=80.0, card_id="card_ato_002", DeviceInfo="Windows")
    hist = store.history_features(tx)
    assert hist["history_found"] is True
    assert hist["prior_transaction_count"] == 8
    assert hist["mean_amount_usd_prior"] is not None
    assert hist["device_seen_before_on_this_card"] is None
    assert hist["trusted_device_ids"] == []
    assert hist["current_device_is_trusted"] is None
    assert hist["confirmed_outcomes"]["approved"] == 8
    assert hist["recent_attempts"]
    assert store.device_history("Windows")["identifier_kind"] == "description"
    assert store.device_history("Windows")["distinct_cards_seen"] is None
    store.record(Transaction(amount=10, card_id="card_ato_002", DeviceInfo="Windows"), "approved")
    assert "Windows" not in store.device_index
    assert store.cards["card_ato_002"].known_devices == set()
    state = build_state(tx, history=hist)
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"
    assert state["card_history"]["mean_amount_usd_prior"] == hist["mean_amount_usd_prior"]


def test_feature_store_new_card_and_shared_device():
    store = FeatureStore()
    store.seed_demo()
    tx = Transaction(
        amount=1250,
        card_id="card_new_777",
        DeviceInfo="SM-G9650 Build/R16NW",
    )
    hist = store.history_features(tx)
    assert hist["history_found"] is False
    assert hist["prior_transaction_count"] == 0
    assert hist["device_seen_before_on_this_card"] is False
    assert hist["device_distinct_cards_seen"] == 4
    assert hist["device_chargebacks"] == 3
    state = build_state(tx, history=hist)
    assert state["card_history"]["card_is_new"] is True
    assert state["device"]["device_is_new_for_card"] is True
    assert state["device"]["current_device_is_trusted"] is False
    assert state["device"]["device_distinct_cards_seen"] == 4
    assert state["card_history"]["trusted_device_ids"] == []
    assert state["card_history"]["recent_attempts"] == []
    assert state["card_history"]["confirmed_outcomes"] == {"approved": 0, "declined": 0, "chargeback": 0}


def test_card_present_rules_reach_jev_state():
    tx = Transaction(
        amount=420,
        channel="card_present",
        card_network="visa",
        entry_mode="fallback_swipe",
        card_status="open",
        cvm_result="signature",
        pin_tries_exceeded=False,
        track_cvv="match",
        merchant_id="merch_elec_19",
        merchant_name="Centro Eletronicos",
        mcc="5732",
        merchant_country="br",
        merchant_city="Sao Paulo",
        cardholder_country="US",
        terminal_attended=True,
        available_credit_usd=5000,
        single_purchase_limit_usd=2000,
    )
    hist = {
        "merchant_seen_before_on_this_card": False,
        "distinct_merchants_last_1h": 2,
        "distinct_merchant_countries_last_24h": 2,
        "minutes_since_previous_merchant": 40.0,
        "previous_merchant_country": "US",
        "merchant_country_changed_within_2h": True,
        "mean_amount_usd_prior": 48.0,
        "amount_usd_last_1h": 46.0,
        "amount_vs_this_merchant_ratio": None,
        "matching_amount_count_last_24h": 0,
    }
    state = build_state(tx, history=hist)
    cp = state["card_present"]
    assert cp["network"] == "visa"
    assert cp["entry_mode_meaning"].startswith("chip failed")
    assert cp["merchant_country"] == "BR"
    assert "merchant terminal" in cp["authorization_path"]
    assert cp["rules"]["magstripe_fallback"] is True
    assert cp["rules"]["swipe_without_pin"] is True
    assert cp["rules"]["cross_border"] is True
    assert cp["rules"]["merchant_country_changed_within_2h"] is True
    assert cp["rules"]["cash_like_mcc"] is False
    assert cp["rules"]["card_unusable"] is False
    assert cp["rules"]["over_limit"] is False
    assert cp["rules"]["merchant_burst_1h"] is False
    assert cp["amount_usd"] == 420.0
    assert cp["mean_prior_amount_usd"] == 48.0
    assert cp["amount_vs_mean_prior_ratio"] == 8.75
    assert cp["amount_usd_last_1h_including_this"] == 466.0
    assert cp["rules"]["amount_far_above_history"] is True
    assert cp["rules"]["probing_amount"] is False
    assert "card_present" not in build_state(Transaction(amount=12.5, channel="card_not_present", merchant_id="m"))


def test_hard_authorization_controls_decline_without_a_high_fraud_score():
    from app.features.card_present import authorization_decline_reason

    stolen = build_state(
        Transaction(
            amount=48,
            channel="card_present",
            card_status="stolen",
            track_cvv="mismatch",
            entry_mode="swipe",
        )
    )
    reason = authorization_decline_reason(stolen)
    assert reason is not None
    assert "card_unusable" in reason
    assert "track_cvv_mismatch" in reason

    over = build_state(
        Transaction(amount=80, channel="card_present", available_credit_usd=50, card_status="open")
    )
    assert "over_limit" in (authorization_decline_reason(over) or "")
    assert authorization_decline_reason(build_state(Transaction(amount=40))) is None

    probe = build_state(Transaction(amount=0.5, channel="card_present", card_status="open"), history={})
    assert probe["card_present"]["amount_usd"] == 0.5
    assert probe["card_present"]["rules"]["probing_amount"] is True
    assert "amount_far_above_history" not in probe["card_present"]["rules"]
    spike = build_state(
        Transaction(amount=200, channel="card_present", card_status="open"),
        history={"mean_amount_usd_prior": 40.0, "amount_vs_this_merchant_ratio": 5.0},
    )
    assert spike["card_present"]["rules"]["amount_far_above_history"] is True
    assert spike["card_present"]["rules"]["amount_far_above_this_merchant"] is True
    assert authorization_decline_reason(spike) is None


def test_merchant_country_change_within_2h():
    at = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    prior = [
        PriorTxn(
            amount=46,
            timestamp=at - timedelta(minutes=40),
            merchant_id="merch_grocery_88",
            merchant_country="US",
        ),
        PriorTxn(amount=20, timestamp=at - timedelta(days=3), merchant_id="merch_old", merchant_country="US"),
    ]
    from app.features.history import compute_merchant_features

    feats = compute_merchant_features(
        prior=prior, at=at, merchant_id="merch_elec_19", merchant_country="BR"
    )
    assert feats["merchant_seen_before_on_this_card"] is False
    assert feats["distinct_merchants_last_1h"] == 2
    assert feats["previous_merchant_country"] == "US"
    assert feats["minutes_since_previous_merchant"] == 40.0
    assert feats["merchant_country_changed_within_2h"] is True

    same = compute_merchant_features(
        prior=prior, at=at, merchant_id="merch_grocery_88", merchant_country="US"
    )
    assert same["merchant_seen_before_on_this_card"] is True
    assert same["merchant_country_changed_within_2h"] is False
    assert same["distinct_merchants_last_1h"] == 1


def test_questions_tell_jev_to_read_card_present_rules():
    from app.jev.questions import build_questions

    qs = build_questions()
    assert "card_present.rules" in qs["is_fraud"].instructions
    assert "magstripe_fallback" in qs["is_fraud"].instructions
    assert "amount_usd" in qs["is_fraud"].instructions
    assert "amount_far_above_history" in qs["is_fraud"].instructions
    assert "amount_far_above_history" in qs["amount_anomalous"].instructions
    assert "card_present" in qs["risk"].instructions
    assert "presentment_invalid" in qs
    assert "merchant_anomaly" in qs


def test_missing_card_id_is_unknown_not_a_new_card():
    store = FeatureStore()
    store.seed_demo()
    tx = Transaction(amount=40.0, DeviceInfo="Windows")
    hist = store.history_features(tx)
    state = build_state(tx, history=hist)
    assert state["card_history"]["history_found"] == "unknown"
    assert state["card_history"]["prior_transaction_count"] == "unknown"
    assert state["card_history"]["mean_amount_usd_prior"] == "unknown"
    assert state["card_history"]["card_is_new"] == "unknown"
    assert state["card_history"]["recent_attempts"] == "unknown"
    assert state["card_history"]["confirmed_outcomes"] == "unknown"
    assert state["device"]["device_seen_before_on_this_card"] == "unknown"
    assert "card_is_new" not in build_state(Transaction(amount=12.5)).get("card_history", {})
