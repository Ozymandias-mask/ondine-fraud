"""Comportement de l'API de scoring : décision, latence, robustesse, sécurité des entrées.

Le modèle est un vrai modèle entraîné sur le petit lac, chargé depuis un registre MLflow
local : ce qui est vérifié ici est le chemin complet d'une décision, pas une simulation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import fakeredis
import pytest
import redis
from fastapi.testclient import TestClient
from ondine_api.app import Service, create_app
from ondine_api.settings import ApiSettings
from ondine_api.store import OnlineStore
from ondine_ml.settings import MlSettings
from ondine_ml.train import train

BASE_TIME = datetime(2025, 5, 15, 14, 30, tzinfo=UTC)


def transaction(**overrides) -> dict:
    """Demande d'autorisation type (paiement en ligne)."""
    payload = {
        "transaction_id": "TX-TEST-1",
        "event_time": BASE_TIME.isoformat(),
        "client_id": "C00000001",
        "card_id": "K000000001",
        "merchant_id": "M00000001",
        "mcc": 5411,
        "merchant_country": "FR",
        "channel": "ecommerce",
        "entry_mode": "ecommerce",
        "amount": 89.9,
        "currency": "EUR",
        "three_ds_authenticated": True,
        "device_id": "device-1",
        "wallet_type": None,
    }
    return payload | overrides


@pytest.fixture(scope="module")
def trained_model(lake, tmp_path_factory):
    """Un modèle entraîné et promu « champion », comme en production."""
    import mlflow

    store = tmp_path_factory.mktemp("mlflow-api")
    ml = MlSettings(
        tracking_uri=f"sqlite:///{store}/mlflow.db",
        experiment="tests-api",
        model_name="ondine-fraud-detector-api",
        negative_sampling_rate=0.3,
        num_boost_round=120,
        early_stopping_rounds=30,
        shap_sample=1000,
        robustness_sample=2000,
        # Le premier mois est écarté par `history_complete` (fenêtres de 30 jours incomplètes).
        train_bounds="2025-02..2025-02",
        valid_bounds="2025-03..2025-03",
        test_bounds="2025-04..2025-04",
    )
    result = train(lake, ml)
    client = mlflow.MlflowClient(ml.tracking_uri)
    client.set_registered_model_alias(ml.model_name, "champion", result.model_version)
    return ml


@pytest.fixture
def client(trained_model):
    """API prête à répondre, avec un magasin en ligne isolé en mémoire."""
    settings = ApiSettings(
        redis_url="redis://inutilisé", pseudonymisation_key="clé-de-test", model_alias="champion"
    )
    application = create_app(settings, trained_model)
    service: Service = application.state.service
    fake = fakeredis.FakeStrictRedis(decode_responses=True)
    original = OnlineStore.from_url
    OnlineStore.from_url = classmethod(lambda cls, *_: cls(fake, "clé-de-test"))
    try:
        with TestClient(application) as test_client:
            yield test_client, service
    finally:
        OnlineStore.from_url = original


def test_scores_a_transaction_with_an_explanation(client):
    """Une demande d'autorisation reçoit une décision motivée, en moins du budget de latence."""
    http, service = client
    response = http.post("/scorer", json=transaction())
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] in ("bloquer", "autoriser")
    assert 0.0 <= body["probabilite_fraude"] <= 1.0
    assert body["seuil"] == pytest.approx(service.model.threshold)
    assert body["mode_degrade"] is False
    assert body["latence_ms"] < service.settings.latency_budget_ms
    assert len(body["facteurs"]) == service.settings.explain_top_factors
    assert body["explication"]
    # Aucune donnée identifiante ne ressort de l'API.
    assert "device_id" not in response.text and "client_id" not in response.text


def test_latency_stays_within_budget_over_many_calls(client):
    """La latence de décision reste sous le budget annoncé (100 ms), médiane et 95e centile.

    La durée mesurée est celle que l'API rapporte et exporte à Prometheus : lecture du
    magasin, calcul des variables, prédiction et explication. Le transport HTTP et le client
    de test n'en font pas partie — ils ne relèvent pas du budget de la solution.
    """
    http, service = client
    durations = []
    for index in range(60):
        response = http.post(
            "/scorer",
            json=transaction(
                transaction_id=f"TX-LAT-{index}",
                event_time=(BASE_TIME + timedelta(seconds=index)).isoformat(),
            ),
        )
        assert response.status_code == 200
        durations.append(response.json()["latence_ms"])
    durations.sort()
    budget = service.settings.latency_budget_ms
    assert durations[len(durations) // 2] < budget, (
        f"médiane {durations[len(durations) // 2]:.1f} ms"
    )
    assert durations[int(len(durations) * 0.95)] < budget, (
        f"p95 {durations[int(len(durations) * 0.95)]:.1f} ms"
    )


def test_history_written_by_the_api_is_visible_to_the_next_transaction(client):
    """Une rafale sur la même carte est vue comme telle, sans attendre le flux Kafka."""
    http, _ = client
    for index in range(4):
        http.post(
            "/scorer",
            json=transaction(
                transaction_id=f"TX-BURST-{index}",
                event_time=(BASE_TIME + timedelta(seconds=30 * index)).isoformat(),
            ),
        )
    response = http.post(
        "/scorer",
        json=transaction(
            transaction_id="TX-BURST-last",
            event_time=(BASE_TIME + timedelta(seconds=200)).isoformat(),
        ),
    )
    assert response.status_code == 200
    noms = {facteur["variable"] for facteur in response.json()["facteurs"]}
    assert noms  # l'explication porte bien sur des variables du modèle
    # La vélocité de la carte a été alimentée par les appels précédents.
    from ondine_api.features import compute

    service = http.app.state.service
    state = service.store.state(
        transaction(transaction_id="TX-CHECK", event_time=BASE_TIME + timedelta(seconds=300))
    )
    assert compute(transaction(), state)["card_tx_count_1h"] == 5


def test_degraded_mode_when_the_online_store_is_unavailable(client):
    """Magasin en panne : la banque continue d'autoriser, la réponse le signale."""
    http, service = client

    def boom(*_args, **_kwargs):
        raise redis.ConnectionError("magasin injoignable")

    service.store.state = boom
    response = http.post("/scorer", json=transaction(transaction_id="TX-DEGRADE"))
    assert response.status_code == 200
    body = response.json()
    assert body["mode_degrade"] is True
    assert 0.0 <= body["probabilite_fraude"] <= 1.0


def test_fail_closed_configuration_refuses_rather_than_guessing(client):
    """Configuré en « fail closed », l'API refuse de décider plutôt que de deviner."""
    http, service = client
    service.settings = ApiSettings(
        redis_url="redis://inutilisé", pseudonymisation_key="clé-de-test", fail_open=False
    )

    def boom(*_args, **_kwargs):
        raise redis.ConnectionError("magasin injoignable")

    service.store.state = boom
    assert http.post("/scorer", json=transaction()).status_code == 503


@pytest.mark.parametrize(
    "invalide",
    [
        {"amount": -10},
        {"amount": "beaucoup"},
        {"transaction_id": ""},
        {"merchant_country": "FRANCE"},
        {"event_time": "hier"},
        {"colonne_inconnue": 1},
    ],
)
def test_invalid_requests_are_refused_before_scoring(client, invalide):
    """Une entrée malformée est refusée (422), jamais scorée « au mieux »."""
    http, _ = client
    assert http.post("/scorer", json=transaction(**invalide)).status_code == 422


def test_health_and_readiness_probes(client):
    """Les sondes distinguent « en vie » et « prêt à servir »."""
    http, service = client
    assert http.get("/sante").status_code == 200
    ready = http.get("/pret")
    assert ready.status_code == 200
    assert ready.json()["magasin_en_ligne"] is True

    service.model = None
    assert http.get("/pret").status_code == 503


def test_metrics_are_exposed_for_supervision(client):
    """Les métriques nécessaires à la supervision sont publiées au format Prometheus."""
    http, _ = client
    http.post("/scorer", json=transaction(transaction_id="TX-METRIQUE"))
    body = http.get("/metrics").text
    for metric in (
        "scoring_requests_total",
        "scoring_latency_seconds",
        "scoring_model_loaded",
        "scoring_model_threshold",
    ):
        assert metric in body


def test_fast_path_gives_exactly_the_same_score_as_the_training_path(client):
    """Le prétraitement rapide de l'API donne la *même* prédiction que celui de l'entraînement.

    L'API convertit la transaction directement en tableau de réels (25 ms de moins par appel)
    au lieu de passer par pandas comme à l'entraînement. Les deux chemins doivent rester
    strictement équivalents, sinon l'optimisation introduirait un écart silencieux.
    """
    import pandas as pd
    from ondine_api.features import compute

    _, service = client
    model = service.model
    for index in range(20):
        payload = transaction(
            transaction_id=f"TX-PATH-{index}",
            amount=10.0 * (index + 1),
            channel=("ecommerce", "pos_chip", "sepa_instant")[index % 3],
            merchant_country=("FR", "US", None)[index % 3],
        )
        features = compute(payload, service.store.state(payload))
        rapide = model.probabilities(model.row_matrix(features))[0]
        reference = model.probabilities(pd.DataFrame([features]))[0]
        assert rapide == pytest.approx(reference, rel=1e-12)


def test_threshold_is_recalibrated_to_hold_the_alert_budget(client):
    """Le seuil se recale sur les scores récents : c'est le budget d'alertes qui commande.

    Un seuil figé en probabilité ne tient pas la promesse faite à l'équipe fraude — sur les
    données réelles du projet, 4 % d'écart de seuil change le volume d'alertes d'un facteur
    cent. Le budget est donc appliqué comme ce qu'il est : un quantile de la population
    décidée, recalculé à intervalle régulier sur une fenêtre partagée par toutes les
    instances de l'API.
    """
    import numpy as np
    from ondine_api.settings import ApiSettings

    _, service = client
    service.settings = ApiSettings(
        redis_url="redis://inutilisé",
        pseudonymisation_key="clé-de-test",
        recalibration_interval_s=0.0,
        recalibration_min_samples=1000,
        score_window=50_000,
    )
    service.model.alert_budget = 0.01
    depart = service.threshold

    # Une population de scores connue : le quantile à 99 % vaut 0,99.
    scores = np.linspace(0.0, 1.0, 5000)
    for score in scores:
        service.store.record_score(float(score), service.settings.score_window)

    service.recalibrate()
    assert service.threshold != depart
    assert service.threshold == pytest.approx(0.99, abs=0.01)

    # Trop peu d'observations : on ne bouge pas un seuil sur un quantile qui ne veut rien dire.
    service.store.redis.delete("of:scores")
    service.recalibrated_at = float("-inf")
    stable = service.threshold
    for score in np.linspace(0.0, 1.0, 100):
        service.store.record_score(float(score), service.settings.score_window)
    service.recalibrate()
    assert service.threshold == stable


@pytest.fixture
def analyste(client, lake):
    """API dont l'interface analyste est branchée sur la base de test."""
    from ondine_api.alerts import AlertQueue

    http, service = client
    service.queue = AlertQueue(lake.pg_dsn)
    return http, service


def test_an_alert_reaches_the_analyst_queue_and_can_be_decided(analyste):
    """Chaîne complète : blocage → file d'alertes → décision tracée."""
    from ondine_api.app import _record_alert
    from ondine_api.schemas import Decision, Factor

    http, service = analyste
    decision = Decision(
        transaction_id="TX-UI-1",
        probabilite_fraude=0.91,
        decision="bloquer",
        seuil=0.24,
        modele_version="abc12345",
        latence_ms=3.1,
        mode_degrade=False,
        explication="Signalée principalement sur : ancienneté de l'appareil (0.5).",
        facteurs=[
            Factor(
                variable="device_age_hours_for_client",
                libellé="ancienneté de l'appareil pour ce client",
                valeur=0.5,
                effet="augmente le risque",
                poids=1.2,
            )
        ],
    )
    _record_alert(service, transaction(transaction_id="TX-UI-1"), decision)

    file_attente = http.get("/alertes")
    assert file_attente.status_code == 200
    assert "TX-UI-1"[:12] in file_attente.text

    detail = http.get("/alertes/TX-UI-1")
    assert detail.status_code == 200
    # Le rendu échappe les apostrophes : c'est précisément ce qu'on attend d'une page qui
    # affiche des données, et la vérification en tient compte.
    assert "appareil" in detail.text and "0.5" in detail.text

    envoi = http.post(
        "/alertes/TX-UI-1",
        data={"analyste": "a.martin", "etat": "fraude_confirmee", "note": "Client confirme"},
        follow_redirects=True,
    )
    assert envoi.status_code == 200
    assert "Fraude confirmée" in envoi.text and "a.martin" in envoi.text
    assert http.get("/alertes/inconnue").status_code == 404


def test_the_analyst_interface_meets_the_accessibility_rules(analyste):
    """Vérifie les règles RGAA tenables automatiquement, sur les deux pages de l'interface.

    L'accessibilité ne se contrôle pas entièrement par un test — mais ces points-là, si, et
    ils sont ceux qui se dégradent le plus vite au fil des modifications.
    """
    from ondine_api.app import _record_alert
    from ondine_api.schemas import Decision, Factor

    http, service = analyste
    _record_alert(
        service,
        transaction(transaction_id="TX-RGAA-1"),
        Decision(
            transaction_id="TX-RGAA-1",
            probabilite_fraude=0.8,
            decision="bloquer",
            seuil=0.2,
            modele_version="v1",
            latence_ms=2.0,
            mode_degrade=False,
            explication="Signalée sur un montant inhabituel.",
            facteurs=[
                Factor(
                    variable="amount",
                    libellé="montant",
                    valeur=900.0,
                    effet="augmente le risque",
                    poids=0.4,
                )
            ],
        ),
    )
    for page in (http.get("/alertes").text, http.get("/alertes/TX-RGAA-1").text):
        assert '<html lang="fr">' in page, "langue de la page déclarée"
        assert page.count("<title>") == 1 and "Ondine Bank" in page, "titre unique et explicite"
        assert 'class="evitement" href="#contenu"' in page, "lien d'évitement vers le contenu"
        assert '<main id="contenu">' in page and "<header>" in page, "régions sémantiques"
        assert page.count("<h1>") == 1, "un seul titre de niveau 1"
        assert "aria-label" in page, "régions de navigation nommées"
        assert ":focus-visible" in page, "focus clavier visible"
        assert "prefers-reduced-motion" in page, "respect du mouvement réduit"

    liste = http.get("/alertes").text
    assert "<caption>" in liste and 'scope="col"' in liste, "tableau de données légendé et porté"

    detail = http.get("/alertes/TX-RGAA-1").text
    for champ in ("analyste", "etat", "note"):
        assert f'for="{champ}"' in detail and f'id="{champ}"' in detail, f"étiquette de {champ}"
    assert "▲" in detail or "▼" in detail, "le sens d'un facteur ne repose pas sur la couleur"
    assert 'role="status"' not in detail, "aucun message parasite sur une page sans action"
    apres = http.get("/alertes/TX-RGAA-1?message=Test").text
    assert 'aria-live="polite"' in apres, "les confirmations sont annoncées aux lecteurs d'écran"


def test_the_interface_neutralises_injected_markup(analyste):
    """Une donnée hostile s'affiche, elle ne s'exécute pas.

    Les champs affichés viennent du flux de paiement et des analystes : ils ne sont pas
    dignes de confiance. Une interface bancaire qui interpréterait du balisage venu de là
    offrirait une prise directe à un attaquant (injection de script).
    """
    from ondine_api.app import _record_alert
    from ondine_api.schemas import Decision, Factor

    http, service = analyste
    piege = "<script>alert('xss')</script>"
    _record_alert(
        service,
        transaction(transaction_id="TX-XSS-1"),
        Decision(
            transaction_id="TX-XSS-1",
            probabilite_fraude=0.7,
            decision="bloquer",
            seuil=0.2,
            modele_version=piege,
            latence_ms=2.0,
            mode_degrade=False,
            explication=piege,
            facteurs=[
                Factor(
                    variable="amount",
                    libellé=piege,
                    valeur=1.0,
                    effet="augmente le risque",
                    poids=0.1,
                )
            ],
        ),
    )
    http.post(
        "/alertes/TX-XSS-1",
        data={"analyste": piege, "etat": "ecartee", "note": piege},
        follow_redirects=True,
    )
    rendu = http.get("/alertes/TX-XSS-1").text
    assert "<script>" not in rendu, "aucun balisage injecté ne doit être interprété"
    assert "&lt;script&gt;" in rendu, "la donnée reste visible, sous forme de texte"


def test_warming_the_store_gives_the_model_a_real_history(lake, store, monkeypatch):
    """Sans préchauffage, un rejeu part d'un magasin vide et les scores ne valent rien.

    Tous les appareils y paraissent neufs et toutes les vélocités valent zéro : le modèle
    voit partout le même profil. Le préchauffage reconstitue l'historique des clients
    concernés, comme le ferait le flux Kafka en production.
    """
    from ondine_api import replay as rejeu
    from ondine_api.features import compute
    from ondine_api.replay import read_transactions, warm_store

    # Le préchauffage lit les clients par paquets, faute de quoi le tri chronologique de
    # trente jours dépasse la mémoire allouée à DuckDB dans l'image de service. Un paquet
    # d'un seul client force ce découpage ici, sur un lac de test forcément petit.
    monkeypatch.setattr(rejeu, "CLIENT_BATCH", 1)

    transactions = read_transactions(lake, "2025-04", 200)
    assert transactions, "le lac de test doit contenir des transactions à rejouer"

    froid = [compute(t, store.state(t))["card_tx_count_24h"] for t in transactions]
    assert set(froid) == {0}, "magasin vide : aucune vélocité, donc aucun signal"

    lus = warm_store(lake, store, transactions)
    assert lus["transactions"] > 0 and lus["clients"] > 0

    chaud = [compute(t, store.state(t))["card_tx_count_24h"] for t in transactions]
    ages = [compute(t, store.state(t))["device_age_hours_for_client"] for t in transactions]
    assert max(chaud) > 0, "après préchauffage, l'historique de carte existe"
    assert max(a for a in ages if a is not None) > 0, "les appareils ne sont plus tous neufs"


def test_a_too_small_window_does_not_postpone_the_next_recalibration(client):
    """Faute d'observations, le seuil ne bouge pas — mais l'occasion suivante n'est pas perdue.

    Au démarrage, la fenêtre de scores se remplit progressivement. Si un essai infructueux
    consommait l'intervalle, le seuil resterait figé cinq minutes de plus à chaque fois, et la
    mise en service commencerait par une longue période où le budget d'alertes n'est pas tenu.
    """
    import numpy as np
    from ondine_api.settings import ApiSettings

    _, service = client
    service.settings = ApiSettings(
        redis_url="redis://inutilisé",
        pseudonymisation_key="clé-de-test",
        recalibration_interval_s=3600.0,
        recalibration_min_samples=500,
    )
    service.model.alert_budget = 0.01
    service.recalibrated_at = float("-inf")
    depart = service.threshold

    for score in np.linspace(0.0, 1.0, 100):  # trop peu : aucun recalage
        service.store.record_score(float(score), 50_000)
    service.recalibrate()
    assert service.threshold == depart

    for score in np.linspace(0.0, 1.0, 900):  # la fenêtre est maintenant suffisante
        service.store.record_score(float(score), 50_000)
    service.recalibrate()
    assert service.threshold != depart, "l'intervalle ne doit pas avoir été consommé à vide"
