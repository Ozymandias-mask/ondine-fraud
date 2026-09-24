"""API de scoring temps réel : décide en moins de 100 ms si un paiement doit être bloqué.

Chaîne complète d'une décision :

1. le système d'autorisation envoie la demande (même format que le message Kafka) ;
2. le magasin en ligne est lu en un aller-retour : historique glissant de la carte, du
   client, de l'appareil et du bénéficiaire ;
3. les variables sont calculées avec les **mêmes définitions** que la couche gold ;
4. le modèle promu « champion » dans le registre MLflow rend une probabilité calibrée, que
   son seuil transforme en décision ;
5. les facteurs explicatifs (SHAP) accompagnent la réponse : l'analyste — et le client —
   savent pourquoi ;
6. la transaction est écrite dans le magasin *après* la réponse, pour que la suivante voie
   déjà celle-ci (une attaque par test de carte enchaîne les paiements en quelques secondes).

Choix de disponibilité : une banque ne peut pas refuser tous ses paiements parce qu'un cache
est tombé. En cas d'incident sur le magasin en ligne, l'API répond en **mode dégradé** (score
calculé sur les seules données de la demande, indicateur `mode_degrade` à vrai, métrique et
journal dédiés) au lieu d'échouer. Ce comportement est configurable (`ONDINE_API_FAIL_OPEN`).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import redis
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from ondine_ml.explain import explain_row, sentence
from ondine_ml.model import FraudModel
from ondine_ml.serving import load_model
from ondine_ml.settings import MlSettings
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.background import BackgroundTask

from .features import compute
from .schemas import Decision, Health, Transaction
from .settings import ApiSettings
from .store import OnlineStore, State

logger = logging.getLogger(__name__)

SCORED = Counter("scoring_requests_total", "Demandes d'autorisation scorées", ["decision"])
DEGRADED = Counter("scoring_degraded_total", "Décisions rendues sans magasin en ligne", ["cause"])
ERRORS = Counter("scoring_errors_total", "Décisions en échec", ["cause"])
LATENCY = Histogram(
    "scoring_latency_seconds",
    "Durée d'une décision, de la réception à la réponse",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5, 1),
)
FEATURE_LATENCY = Histogram(
    "scoring_feature_latency_seconds",
    "Durée de la lecture du magasin en ligne et du calcul des variables",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
OVER_BUDGET = Counter("scoring_over_budget_total", "Décisions au-delà du budget de latence")
MODEL_READY = Gauge("scoring_model_loaded", "1 si un modèle est chargé et servi")
MODEL_THRESHOLD = Gauge("scoring_model_threshold", "Seuil de décision du modèle servi")


@dataclass
class Service:
    """Ce que l'API garde en mémoire entre deux requêtes : le modèle et le magasin."""

    settings: ApiSettings
    ml: MlSettings
    model: FraudModel | None = None
    version: str = "inconnue"
    store: OnlineStore | None = None

    def load(self) -> None:
        """Charge le modèle promu et ouvre le magasin. Échouer ici évite de servir à vide."""
        self.store = OnlineStore.from_url(
            self.settings.redis_url, self.settings.pseudonymisation_key
        )
        self.model, self.version = load_model(self.ml, self.settings.model_alias)
        self._warm_up()
        MODEL_READY.set(1)
        MODEL_THRESHOLD.set(self.model.threshold)

    def _warm_up(self) -> None:
        """Première prédiction à vide, au démarrage.

        LightGBM et SHAP allouent leurs tampons au premier appel : sans cette mise en chauffe,
        la toute première transaction réelle paierait ce coût (plus de 100 ms) — le pire moment.
        """
        empty = self.model.row_matrix({})
        self.model.probabilities(empty)
        explain_row(self.model, empty, {}, top=self.settings.explain_top_factors)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois, au démarrage : aucune latence de chargement en vol."""
    service: Service = app.state.service
    try:
        service.load()
    except Exception as error:  # noqa: BLE001 — l'API doit démarrer pour dire pourquoi elle échoue
        MODEL_READY.set(0)
        logger.error("Modèle non chargé au démarrage : %s", error)
    yield


def create_app(settings: ApiSettings | None = None, ml: MlSettings | None = None) -> FastAPI:
    """Construit l'application (utilisée telle quelle par les tests et par la production)."""
    settings = settings or ApiSettings.from_env()
    application = FastAPI(
        title="Ondine Bank — scoring de fraude",
        version="1.0.0",
        summary="Décision de blocage d'un paiement, explicable, en moins de 100 ms.",
        lifespan=lifespan,
    )
    application.state.service = Service(settings=settings, ml=ml or MlSettings.from_env())
    _register_routes(application)
    return application


def _register_routes(application: FastAPI) -> None:
    @application.get("/sante", response_model=Health, tags=["exploitation"])
    def sante(request: Request) -> Health:
        """Sonde de vivacité : l'API répond-elle ? (sans dépendance externe)."""
        service: Service = request.app.state.service
        # Le magasin n'est volontairement pas sollicité ici : une sonde de vivacité qui
        # dépend d'un service tiers fait redémarrer l'API pour une panne qui n'est pas la
        # sienne. C'est `/pret` qui vérifie les dépendances.
        return Health(
            statut="en vie",
            modele=service.ml.model_name if service.model else None,
            modele_version=service.version if service.model else None,
            seuil=service.model.threshold if service.model else None,
        )

    @application.get("/pret", response_model=Health, tags=["exploitation"])
    def pret(request: Request, response: Response) -> Health:
        """Sonde de mise en service : modèle chargé **et** magasin joignable."""
        service: Service = request.app.state.service
        online = bool(service.store and service.store.ping())
        ready = bool(service.model) and online
        response.status_code = 200 if ready else 503
        return Health(
            statut="prêt" if ready else "indisponible",
            modele=service.ml.model_name if service.model else None,
            modele_version=service.version if service.model else None,
            seuil=service.model.threshold if service.model else None,
            magasin_en_ligne=online,
        )

    @application.get("/metrics", tags=["exploitation"])
    def metrics() -> Response:
        """Métriques Prometheus : volume, latence, décisions, mode dégradé."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.post("/scorer", response_model=Decision, tags=["décision"])
    def scorer(transaction: Transaction, request: Request) -> Response:
        """Rend la décision pour une demande d'autorisation."""
        service: Service = request.app.state.service
        if service.model is None:
            ERRORS.labels("modele_absent").inc()
            raise HTTPException(503, "Aucun modèle chargé : vérifier le registre MLflow.")
        return _decide(service, transaction)


def _decide(service: Service, transaction: Transaction) -> Response:
    """Lit le magasin, calcule les variables, score, explique et répond."""
    started = time.perf_counter()
    payload = transaction.model_dump()
    features, state, degraded = _features(service, payload)
    # Un seul prétraitement pour la probabilité *et* l'explication.
    matrix = service.model.row_matrix(features)
    probability = float(service.model.probabilities(matrix)[0])
    alert = probability >= service.model.threshold
    factors = explain_row(service.model, matrix, features, top=service.settings.explain_top_factors)
    elapsed = time.perf_counter() - started

    LATENCY.observe(elapsed)
    SCORED.labels("bloquer" if alert else "autoriser").inc()
    if elapsed * 1000 > service.settings.latency_budget_ms:
        OVER_BUDGET.inc()
    decision = Decision(
        transaction_id=transaction.transaction_id,
        probabilite_fraude=probability,
        decision="bloquer" if alert else "autoriser",
        seuil=service.model.threshold,
        modele_version=service.version,
        latence_ms=round(elapsed * 1000, 3),
        mode_degrade=degraded,
        explication=sentence(factors),
        facteurs=factors,
    )
    # Écriture après la réponse : la transaction suivante de la même carte voit déjà celle-ci
    # sans que l'appelant en paie la latence. Le flux Kafka réécrira le même membre.
    task = None if degraded or state is None else BackgroundTask(_remember, service, payload)
    return JSONResponse(decision.model_dump(), background=task)


def _features(service: Service, payload: dict) -> tuple[dict, State | None, bool]:
    """Variables du modèle, avec repli si le magasin en ligne est indisponible.

    Le repli n'invente rien : les variables d'historique restent manquantes, ce que le modèle
    sait traiter (LightGBM gère nativement les valeurs absentes) et ce que les tests de
    robustesse mesurent. La réponse le signale explicitement.
    """
    started = time.perf_counter()
    if service.store is None:
        DEGRADED.labels("magasin_absent").inc()
        return compute(payload, State()), None, True
    try:
        state = service.store.state(payload)
        features = compute(payload, state)
        FEATURE_LATENCY.observe(time.perf_counter() - started)
        return features, state, False
    except redis.RedisError as error:
        if not service.settings.fail_open:
            ERRORS.labels("magasin_indisponible").inc()
            raise HTTPException(503, "Magasin de variables en ligne indisponible.") from error
        DEGRADED.labels("magasin_indisponible").inc()
        logger.warning("Magasin en ligne indisponible, décision dégradée : %s", error)
        return compute(payload, State()), None, True


def _remember(service: Service, payload: dict) -> None:
    """Écrit la transaction dans le magasin (après la réponse), sans faire échouer l'appel."""
    try:
        service.store.record_transaction(payload)
    except redis.RedisError as error:  # l'écriture est rattrapée par le flux Kafka
        DEGRADED.labels("ecriture_differee").inc()
        logger.warning("Transaction non écrite dans le magasin en ligne : %s", error)


def build() -> FastAPI:
    """Fabrique appelée par uvicorn (`ondine_api.app:build --factory`) et par les tests."""
    return create_app()
