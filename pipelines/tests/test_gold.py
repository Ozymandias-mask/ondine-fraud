import duckdb
import pandas as pd
import pytest
from ondine_pipelines.gold.features import build_features_month, build_first_seen
from ondine_pipelines.ingest.landing import DATASETS, ingest_dataset
from ondine_pipelines.registry import Registry
from ondine_pipelines.silver.activity import build_month
from ondine_pipelines.silver.labels import build_labels
from ondine_pipelines.silver.referentials import build_referentials


@pytest.fixture(scope="module")
def gold(base_settings, landing):
    registry = Registry(base_settings)
    for dataset in DATASETS:
        ingest_dataset(base_settings, dataset)
    build_referentials(base_settings)
    build_labels(base_settings)
    periods = registry.ingested_periods("transactions")
    for period in periods:
        y, m = int(period[:4]), int(period[5:])
        for dataset in ("transactions", "app_events"):
            if not registry.last_success("silver", f"{dataset}_{period}"):
                build_month(base_settings, dataset, y, m)
    build_first_seen(base_settings)
    for period in periods:
        build_features_month(base_settings, int(period[:4]), int(period[5:]))
    features = duckdb.sql(
        f"SELECT * FROM read_parquet('{base_settings.uri('gold', 'transaction_features')}/**/*.parquet')"
    ).df()
    truth = pd.concat(pd.read_csv(f) for f in (landing.parent / "ground_truth").glob("*.csv.gz"))
    return features, truth


def test_one_row_per_transaction(gold, base_settings):
    features, _ = gold
    silver = duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{base_settings.uri('silver', 'transactions')}/**/*.parquet')"
    ).fetchone()[0]
    assert len(features) == silver
    assert features["transaction_id"].is_unique


def test_card_velocity_is_point_in_time(gold):
    features, _ = gold
    tx = features[features["card_id"].notna()].sort_values("event_time")
    sample = tx.sample(300, random_state=1)
    by_card = {card: grp for card, grp in tx.groupby("card_id")}
    for row in sample.itertuples():
        history = by_card[row.card_id]
        expected = (
            (history["event_time"] < row.event_time)
            & (history["event_time"] >= row.event_time - pd.Timedelta(hours=1))
        ).sum()
        assert row.card_tx_count_1h == expected


def test_labels_match_confirmed_reports_and_are_dated(gold):
    features, truth = gold
    labelled = features[features["is_fraud"]]
    assert len(labelled) > 0
    assert set(labelled["transaction_id"]) <= set(truth["transaction_id"])
    assert (labelled["label_reported_at"] > labelled["event_time"]).all()


def test_features_separate_fraud_scenarios(gold):
    features, truth = gold
    df = features.merge(truth[["transaction_id", "fraud_type"]], on="transaction_id", how="left")
    legit = df[df["fraud_type"].isna()]

    def mean(frame, column):
        return frame[column].fillna(0).astype(float).mean()

    testing = df[df["fraud_type"] == "card_testing"]
    assert mean(testing, "card_small_online_tx_1h") > 5 * mean(legit, "card_small_online_tx_1h")
    ato = df[df["fraud_type"] == "account_takeover"]
    if len(ato):
        assert mean(ato, "failed_logins_24h") > 0.5 > mean(legit, "failed_logins_24h")
        assert ato["is_new_beneficiary"].mean() > 0.3
    wallet = df[df["fraud_type"] == "wallet_enrolment"]
    if len(wallet):
        assert mean(wallet, "wallet_provisionings_7d") >= 1 > mean(legit, "wallet_provisionings_7d")


def test_fairness_attributes_present_and_pseudonyms_absent(gold):
    features, _ = gold
    assert {"age_band", "region", "segment"} <= set(features.columns)
    assert not [c for c in features.columns if "hash" in c]


def test_history_flag_marks_warm_up_period(gold):
    features, _ = gold
    assert not features.loc[
        features["event_time"] < features["event_time"].min() + pd.Timedelta(days=5),
        "history_complete",
    ].any()


def test_lake_integrity_check_names_a_truncated_file(gold, base_settings, tmp_path):
    """Un Parquet coupé en deux est détecté et désigné, avec le motif de l'erreur.

    C'est le mode de panne d'un arrêt brutal : le fichier reste présent et de taille
    plausible, mais n'échoue qu'à la relecture — parfois des heures plus tard, dans un
    traitement sans rapport. Le contrôle le nomme avant que cela n'arrive.
    """
    import shutil
    from dataclasses import replace

    from ondine_pipelines.integrity import check_layer, summary

    copie = tmp_path / "lake"
    shutil.copytree(base_settings.lake_root, copie)
    settings = replace(base_settings, lake_root=str(copie))

    sain = summary(check_layer(settings, "gold"))
    assert sain["fichiers"] > 0 and sain["fichiers_illisibles"] == 0 and sain["lignes"] > 0

    # Coupure en plein milieu : les métadonnées de fin disparaissent, comme après un arrêt
    # brutal. Un contrôle qui se contenterait de compter les lignes n'y verrait rien.
    cible = sorted(copie.glob("gold/transaction_features/**/*.parquet"))[0]
    contenu = cible.read_bytes()
    cible.write_bytes(contenu[: len(contenu) // 2])

    abime = summary(check_layer(settings, "gold"))
    assert abime["fichiers_illisibles"] == 1
    assert abime["a_recalculer"][0]["fichier"].endswith(cible.name)
    assert abime["a_recalculer"][0]["motif"]
