"""Variables (features) de détection de fraude, une ligne par transaction.

Familles de variables et fraudes qu'elles permettent de repérer :

- transaction : montant, canal, pays, heure, 3-D Secure, wallet (contexte) ;
- vélocité de la carte : opérations sur 1 h et 24 h, petits achats en ligne (test de carte) ;
- historique du client : volume et montant moyen sur 30 jours (montant inhabituel) ;
- déplacement : distance et vitesse depuis le dernier paiement physique (carte clonée) ;
- appareil : ancienneté de l'appareil pour ce client (prise de contrôle) ;
- bénéficiaire : premier virement, nouveaux payeurs sur 30 jours (comptes mules) ;
- session applicative : échecs de connexion, mot de passe réinitialisé, méthode
  d'authentification changée, enrôlement de wallet (prise de contrôle, SIM swap, wallet).

Les attributs `age_band`, `region` et `segment` sont conservés pour mesurer l'équité du
modèle (tests de biais) ; ils ne doivent pas être utilisés comme variables d'entrée.
"""

from __future__ import annotations

import logging
from datetime import date

import duckdb

from .. import quality
from ..lake import connect, copy_to_parquet, exists, month_glob
from ..registry import Registry, Run
from ..settings import Settings
from ..silver.common import count, require_input, validate_and_publish

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = 31  # historique nécessaire aux fenêtres de 30 jours

TX_COLUMNS = (
    "transaction_id, event_time, event_date, event_hour, is_weekend, client_id, card_id,"
    " merchant_id, mcc, merchant_country, channel, entry_mode, amount, three_ds_authenticated,"
    " device_hash, merchant_latitude, merchant_longitude, beneficiary_hash, beneficiary_country,"
    " beneficiary_bank_code, auth_status, decline_reason, wallet_type, _source_file"
)
EVENT_COLUMNS = "event_type, event_time, client_id, device_hash, ip_country, sca_method"
EVENT_TYPES_USED = (
    "('login', 'login_failed', 'password_reset', 'beneficiary_added', 'sca_method_changed',"
    " 'wallet_provisioning', 'sca_challenge')"
)


def _month_start(year: int, month: int) -> date:
    return date(year, month, 1)


def _previous(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


def build_first_seen(settings: Settings, orchestrator_id: str | None = None) -> dict[str, int]:
    """Tables « première apparition » calculées sur tout l'historique silver.

    - (client, appareil) : quand un client a utilisé un appareil pour la première fois ;
    - (client, bénéficiaire) : date du premier virement vers ce bénéficiaire, et nombre de
      clients différents ayant commencé à payer ce bénéficiaire dans les 30 jours précédents ;
    - (client, pays IP) : première connexion depuis un pays.
    Ce sont des minimums de dates : leur usage reste correct dans le temps (point-in-time).
    """
    registry = Registry(settings)
    con = connect(settings)
    tx = settings.uri("silver", "transactions", "*", "*", "*.parquet")
    ev = settings.uri("silver", "app_events", "*", "*", "*.parquet")
    try:
        with registry.run("gold", "first_seen", orchestrator_id) as run:
            require_input(con, tx)
            require_input(con, ev)
            outputs = {
                "client_device": f"""
                    SELECT client_id, device_hash, min(event_time) AS first_seen
                    FROM (SELECT client_id, device_hash, event_time FROM read_parquet('{tx}')
                          UNION ALL
                          SELECT client_id, device_hash, event_time FROM read_parquet('{ev}'))
                    WHERE device_hash IS NOT NULL
                    GROUP BY ALL""",
                "client_beneficiary": f"""
                    WITH pairs AS (
                        SELECT client_id, beneficiary_hash, min(event_time) AS first_seen
                        FROM read_parquet('{tx}')
                        WHERE beneficiary_hash IS NOT NULL GROUP BY ALL)
                    SELECT *, count(*) OVER (
                        PARTITION BY beneficiary_hash ORDER BY first_seen
                        RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW
                        EXCLUDE CURRENT ROW) AS beneficiary_new_senders_30d
                    FROM pairs""",
                "client_ip_country": f"""
                    SELECT client_id, ip_country, min(event_time) AS first_seen
                    FROM read_parquet('{ev}') WHERE ip_country IS NOT NULL
                    GROUP BY ALL""",
            }
            for name, query in outputs.items():
                target = settings.uri("gold", "first_seen", f"{name}.parquet")
                written = copy_to_parquet(con, settings, query, target)
                run.outputs.append({"uri": target, "rows": written})
                run.rows_out += written
    finally:
        con.close()
    return {"rows": run.rows_out}


def _load_inputs(
    con: duckdb.DuckDBPyConnection, settings: Settings, year: int, month: int, run: Run
) -> None:
    """Charge le mois et son historique (31 jours avant) depuis la couche silver."""
    start = _month_start(year, month)
    months = [(year, month)]
    prev = _previous(year, month)
    if exists(con, month_glob(settings, "silver", "transactions", *prev)):
        months.insert(0, prev)
    # Seules les colonnes et les événements utiles sont chargés, avec le recul strictement
    # nécessaire (30 jours pour les transactions, 7 jours pour les événements) : la mémoire
    # utilisée reste proportionnelle au volume d'un mois.
    sources = {
        "tx_all": ("transactions", LOOKBACK_DAYS, TX_COLUMNS, "TRUE"),
        "ev_all": ("app_events", 8, EVENT_COLUMNS, f"event_type IN {EVENT_TYPES_USED}"),
    }
    for table, (dataset, lookback, columns, condition) in sources.items():
        files = [month_glob(settings, "silver", dataset, y, m) for y, m in months]
        require_input(con, files[-1])
        file_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {table} AS
            SELECT {columns} FROM read_parquet({file_list}, union_by_name = true)
            WHERE event_time >= TIMESTAMPTZ '{start}' - INTERVAL {lookback} DAYS
              AND event_time < TIMESTAMPTZ '{start}' + INTERVAL 1 MONTH
              AND {condition}
        """)
        run.inputs.append({"uris": files, "rows": count(con, table)})
    all_transactions = settings.uri("silver", "transactions", "*", "*", "*.parquet")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE period AS
        SELECT TIMESTAMPTZ '{start}' AS period_start,
               TIMESTAMPTZ '{start}' + INTERVAL 1 MONTH AS period_end,
               (SELECT min(event_time) FROM read_parquet('{all_transactions}')) AS history_start
    """)
    views = {
        "fs_client_device": settings.uri("gold", "first_seen", "client_device.parquet"),
        "fs_client_beneficiary": settings.uri("gold", "first_seen", "client_beneficiary.parquet"),
        "fs_client_ip_country": settings.uri("gold", "first_seen", "client_ip_country.parquet"),
        "clients": settings.uri("silver", "clients", "clients.parquet"),
        "merchants": settings.uri("silver", "merchants", "merchants.parquet"),
        "labels": settings.uri("silver", "fraud_labels", "fraud_labels.parquet"),
    }
    for view, uri in views.items():
        require_input(con, uri)
        con.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM read_parquet('{uri}')")


FEATURES_SQL = """
-- 1. Vélocité de la carte (fenêtres qui excluent la transaction courante).
CREATE OR REPLACE TEMP TABLE f_card AS
SELECT transaction_id,
    count(*) OVER card_w1h AS card_tx_count_1h,
    count(*) OVER card_w24h AS card_tx_count_24h,
    coalesce(sum(amount) OVER card_w24h, 0) AS card_amount_sum_24h,
    count(DISTINCT merchant_id) OVER card_w1h AS card_distinct_merchants_1h,
    count(*) FILTER (WHERE channel = 'ecommerce' AND amount < 5) OVER card_w1h
        AS card_small_online_tx_1h,
    count(*) FILTER (WHERE auth_status = 'declined') OVER card_w1h AS card_declined_1h,
    date_diff('second', lag(event_time) OVER (PARTITION BY card_id ORDER BY event_time), event_time)
        AS seconds_since_prev_card_tx
FROM tx_all WHERE card_id IS NOT NULL
WINDOW
    card_w1h AS (PARTITION BY card_id ORDER BY event_time
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
    card_w24h AS (PARTITION BY card_id ORDER BY event_time
             RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW);

-- 2. Habitudes du client sur 30 jours et virements récents.
CREATE OR REPLACE TEMP TABLE f_client AS
SELECT transaction_id,
    count(*) OVER cli_w30d AS client_tx_count_30d,
    avg(amount) OVER cli_w30d AS client_avg_amount_30d,
    max(amount) OVER cli_w30d AS client_max_amount_30d,
    count(DISTINCT merchant_country) OVER cli_w7d AS client_distinct_countries_7d,
    count(*) FILTER (WHERE channel = 'sepa_instant') OVER cli_w24h AS client_transfers_24h,
    coalesce(sum(amount) FILTER (WHERE channel = 'sepa_instant') OVER cli_w24h, 0)
        AS client_transfer_amount_24h
FROM tx_all
WINDOW
    cli_w30d AS (PARTITION BY client_id ORDER BY event_time
             RANGE BETWEEN INTERVAL 30 DAYS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
    cli_w7d AS (PARTITION BY client_id ORDER BY event_time
            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
    cli_w24h AS (PARTITION BY client_id ORDER BY event_time
             RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW);

-- 3. Déplacement physique : distance et vitesse depuis le paiement physique précédent.
CREATE OR REPLACE TEMP TABLE f_travel AS
WITH physical AS (
    SELECT transaction_id, event_time, merchant_latitude AS lat, merchant_longitude AS lon,
        lag(event_time) OVER w AS prev_time,
        lag(merchant_latitude) OVER w AS prev_lat,
        lag(merchant_longitude) OVER w AS prev_lon
    FROM tx_all
    WHERE merchant_latitude IS NOT NULL
    WINDOW w AS (PARTITION BY client_id ORDER BY event_time))
SELECT transaction_id,
    -- Formule de haversine (rayon terrestre 6 371 km).
    2 * 6371 * asin(sqrt(
        pow(sin(radians(lat - prev_lat) / 2), 2)
        + cos(radians(prev_lat)) * cos(radians(lat)) * pow(sin(radians(lon - prev_lon) / 2), 2)
    )) AS km_from_prev_physical,
    date_diff('second', prev_time, event_time) / 3600.0 AS hours_from_prev_physical
FROM physical WHERE prev_time IS NOT NULL;

-- 4. Signaux de session applicative avant la transaction (chronologie commune par client).
CREATE OR REPLACE TEMP TABLE timeline AS
SELECT e.client_id, e.event_time, NULL::VARCHAR AS transaction_id,
    (e.event_type = 'login_failed')::INT AS is_failed_login,
    (e.event_type = 'password_reset')::INT AS is_password_reset,
    (e.event_type = 'beneficiary_added')::INT AS is_beneficiary_added,
    (e.event_type = 'sca_method_changed')::INT AS is_sca_change,
    (e.event_type = 'wallet_provisioning')::INT AS is_wallet_provisioning,
    (e.event_type IN ('login', 'login_failed') AND e.ip_country <> 'FR')::INT AS is_foreign_login,
    (e.event_type IN ('login', 'login_failed')
        AND fd.first_seen >= e.event_time - INTERVAL 1 SECOND)::INT AS is_new_device_login,
    (e.event_type = 'sca_challenge' AND e.sca_method = 'sms_otp')::INT AS is_sms_otp
FROM ev_all e
LEFT JOIN fs_client_device fd USING (client_id, device_hash)
UNION ALL
SELECT client_id, event_time, transaction_id, 0, 0, 0, 0, 0, 0, 0, 0 FROM tx_all;

CREATE OR REPLACE TEMP TABLE f_session AS
SELECT transaction_id,
    sum(is_failed_login) OVER ses_w24h AS failed_logins_24h,
    sum(is_password_reset) OVER ses_w24h AS password_resets_24h,
    sum(is_beneficiary_added) OVER ses_w24h AS beneficiaries_added_24h,
    sum(is_foreign_login) OVER ses_w24h AS foreign_ip_logins_24h,
    sum(is_new_device_login) OVER ses_w24h AS new_device_logins_24h,
    sum(is_sms_otp) OVER ses_w1h AS sms_otp_challenges_1h,
    sum(is_sca_change) OVER ses_w7d AS sca_method_changes_7d,
    sum(is_wallet_provisioning) OVER ses_w7d AS wallet_provisionings_7d
FROM timeline
WINDOW
    ses_w1h AS (PARTITION BY client_id ORDER BY event_time
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
    ses_w24h AS (PARTITION BY client_id ORDER BY event_time
             RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW),
    ses_w7d AS (PARTITION BY client_id ORDER BY event_time
            RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW EXCLUDE CURRENT ROW)
QUALIFY transaction_id IS NOT NULL;

-- 5. Labels : fraude confirmée et date à laquelle la banque l'a apprise.
CREATE OR REPLACE TEMP TABLE f_labels AS
SELECT transaction_id,
    bool_or(is_confirmed_fraud) AS is_fraud,
    min(reported_at) FILTER (WHERE is_confirmed_fraud) AS label_reported_at,
    any_value(claim_category) FILTER (WHERE is_confirmed_fraud) AS fraud_claim_category,
    bool_or(NOT is_confirmed_fraud) AS has_rejected_claim
FROM labels GROUP BY transaction_id;

-- 6. Assemblage des transactions du mois (l'historique ne sert qu'aux fenêtres).
CREATE OR REPLACE TEMP TABLE gold_rows AS
SELECT
    t.transaction_id, t.event_time, t.event_date, t.client_id, t.card_id, t.merchant_id,
    -- Transaction
    t.amount, ln(1 + t.amount) AS log_amount, t.channel, t.entry_mode, t.mcc,
    m.category AS merchant_category,
    t.merchant_country, coalesce(t.merchant_country <> 'FR', false) AS is_foreign_merchant,
    coalesce(m.is_online, false) AS is_online_merchant,
    t.event_hour, t.is_weekend, (t.event_hour BETWEEN 0 AND 5) AS is_night,
    t.three_ds_authenticated, coalesce(t.wallet_type, 'none') AS wallet_type,
    t.beneficiary_country, t.beneficiary_bank_code,
    -- Vélocité carte
    coalesce(c.card_tx_count_1h, 0) AS card_tx_count_1h,
    coalesce(c.card_tx_count_24h, 0) AS card_tx_count_24h,
    coalesce(c.card_amount_sum_24h, 0) AS card_amount_sum_24h,
    coalesce(c.card_distinct_merchants_1h, 0) AS card_distinct_merchants_1h,
    coalesce(c.card_small_online_tx_1h, 0) AS card_small_online_tx_1h,
    coalesce(c.card_declined_1h, 0) AS card_declined_1h,
    c.seconds_since_prev_card_tx,
    -- Historique client
    k.client_tx_count_30d, k.client_avg_amount_30d, k.client_max_amount_30d,
    t.amount / nullif(k.client_avg_amount_30d, 0) AS amount_to_client_avg,
    k.client_distinct_countries_7d, k.client_transfers_24h, k.client_transfer_amount_24h,
    -- Déplacement
    tr.km_from_prev_physical,
    tr.km_from_prev_physical / greatest(tr.hours_from_prev_physical, 1.0 / 60) AS travel_speed_kmh,
    -- Appareil
    date_diff('second', fd.first_seen, t.event_time) / 3600.0 AS device_age_hours_for_client,
    -- Bénéficiaire
    CASE WHEN t.beneficiary_hash IS NOT NULL THEN fb.first_seen >= t.event_time END
        AS is_new_beneficiary,
    CASE WHEN t.beneficiary_hash IS NOT NULL THEN coalesce(fb.beneficiary_new_senders_30d, 0) END
        AS beneficiary_new_senders_30d,
    -- Session applicative
    s.failed_logins_24h, s.password_resets_24h, s.beneficiaries_added_24h,
    s.foreign_ip_logins_24h, s.new_device_logins_24h, s.sms_otp_challenges_1h,
    s.sca_method_changes_7d, s.wallet_provisionings_7d,
    -- Attributs d'équité (analyse de biais uniquement, pas des variables d'entrée)
    cl.age_band, cl.region, cl.segment, cl.tenure_days,
    -- Décision des règles actuelles (comparaison avec le modèle, pas une variable d'entrée)
    t.auth_status AS legacy_auth_status, t.decline_reason AS legacy_decline_reason,
    -- Labels
    coalesce(l.is_fraud, false) AS is_fraud, l.label_reported_at, l.fraud_claim_category,
    coalesce(l.has_rejected_claim, false) AS has_rejected_claim,
    -- Qualité de l'historique : moins de 30 jours de recul en début de données
    date_diff('day', p.history_start, t.event_time) >= 30 AS history_complete,
    t._source_file
FROM tx_all t
CROSS JOIN period p
LEFT JOIN f_card c ON c.transaction_id = t.transaction_id
LEFT JOIN f_client k ON k.transaction_id = t.transaction_id
LEFT JOIN f_travel tr ON tr.transaction_id = t.transaction_id
LEFT JOIN f_session s ON s.transaction_id = t.transaction_id
LEFT JOIN f_labels l ON l.transaction_id = t.transaction_id
LEFT JOIN fs_client_device fd ON fd.client_id = t.client_id AND fd.device_hash = t.device_hash
LEFT JOIN fs_client_beneficiary fb
    ON fb.client_id = t.client_id AND fb.beneficiary_hash = t.beneficiary_hash
LEFT JOIN merchants m ON m.merchant_id = t.merchant_id
LEFT JOIN clients cl ON cl.client_id = t.client_id
WHERE t.event_time >= p.period_start AND t.event_time < p.period_end;
"""


def _checks(year: int, month: int) -> list[quality.Check]:
    start = _month_start(year, month)
    return [
        quality.Check(
            "completeness",
            "SELECT (SELECT count(*) FROM gold_rows) - (SELECT count(*) FROM tx_all, period"
            " WHERE event_time >= period_start AND event_time < period_end)",
            "==",
            0,
            description="Une ligne de features par transaction silver du mois",
        ),
        quality.Check(
            "transaction_id_unique",
            "SELECT count(*) - count(DISTINCT transaction_id) FROM gold_rows",
            "==",
            0,
        ),
        quality.Check(
            "no_future_window",
            "SELECT count(*) FROM gold_rows WHERE seconds_since_prev_card_tx < 0"
            " OR card_tx_count_1h > card_tx_count_24h",
            "==",
            0,
            description="Cohérence temporelle des fenêtres (pas de fuite du futur)",
        ),
        quality.Check(
            "label_known_after_transaction",
            "SELECT count(*) FROM gold_rows WHERE label_reported_at < event_time",
            "==",
            0,
            description="Une fraude ne peut pas être déclarée avant d'avoir eu lieu",
        ),
        quality.Check(
            "fraud_rate_plausible",
            "SELECT avg(is_fraud::INT) FROM gold_rows",
            "<=",
            0.02,
            severity="warning",
        ),
        quality.Check(
            "fraud_rate_not_null",
            "SELECT avg(is_fraud::INT) FROM gold_rows",
            ">=",
            0.0002,
            severity="warning",
            description="Labels absents : déclarations non ingérées ?",
        ),
        quality.Check(
            "client_attributes_joined",
            "SELECT avg((age_band IS NOT NULL)::INT) FROM gold_rows",
            ">=",
            0.999,
        ),
        quality.Check(
            "no_pseudonym_in_gold",
            "SELECT count(*) FROM (DESCRIBE gold_rows) WHERE column_name LIKE '%hash%'",
            "==",
            0,
            description="Minimisation : les pseudonymes ne sortent pas de silver",
        ),
        quality.Check(
            "month_bounds",
            f"SELECT count(*) FROM gold_rows WHERE event_date < DATE '{start}'"
            f" OR event_date >= DATE '{start}' + INTERVAL 1 MONTH",
            "==",
            0,
        ),
    ]


def build_features_month(
    settings: Settings, year: int, month: int, orchestrator_id: str | None = None
) -> dict[str, int]:
    """Calcule les features et labels d'un mois de transactions."""
    registry = Registry(settings)
    con = connect(settings)
    try:
        with registry.run(
            "gold", f"transaction_features_{year}-{month:02d}", orchestrator_id
        ) as run:
            _load_inputs(con, settings, year, month, run)
            run.rows_in = count(con, "tx_all")
            con.execute(FEATURES_SQL)
            validate_and_publish(
                con,
                settings,
                registry,
                run,
                "transaction_features",
                "gold_rows",
                settings.uri(
                    "gold",
                    "transaction_features",
                    f"year={year}",
                    f"month={month:02d}",
                    "transaction_features.parquet",
                ),
                _checks(year, month),
                order_by="event_time",
                layer="gold",
            )
            fraud = con.execute("SELECT sum(is_fraud::INT) FROM gold_rows").fetchone()[0]
            logger.info(
                "%d-%02d : %s transactions, %s fraudes labellisées",
                year,
                month,
                f"{run.rows_out:,}",
                f"{fraud:,}",
            )
    finally:
        con.close()
    return {"rows": run.rows_out}
