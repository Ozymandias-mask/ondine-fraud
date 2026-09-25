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
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import redis
from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from ondine_ml.explain import explain_row, sentence
from ondine_ml.model import FraudModel
from ondine_ml.serving import load_model
from ondine_ml.settings import MlSettings
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.background import BackgroundTasks

from .alerts import STATUSES, AlertQueue
from .features import compute
from .schemas import Decision, Health, Transaction
from .settings import ApiSettings
from .store import OnlineStore, State
from .ui import alert_page, queue_page

logger = logging.getLogger(__name__)

# Empreinte Git inscrite dans l'image à la construction (chaîne de livraison continue).
BUILD = os.environ.get("ONDINE_BUILD", "inconnu")

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
ALERTS_FAILED = Counter("scoring_alerts_failed_total", "Alertes non déposées dans la file")
MODEL_READY = Gauge("scoring_model_loaded", "1 si un modèle est chargé et servi")
# Repli si le modèle servi est antérieur à l'introduction du budget dans ses métadonnées.
DEFAULT_ALERT_BUDGET = 0.001
MODEL_THRESHOLD = Gauge("scoring_model_threshold", "Seuil de décision effectivement appliqué")
RECALIBRATIONS = Counter(
    "scoring_threshold_recalibrations_total", "Recalages du seuil sur les scores récents"
)


@dataclass
class Service:
    """Ce que l'API garde en mémoire entre deux requêtes : le modèle et le magasin."""

    settings: ApiSettings
    ml: MlSettings
    model: FraudModel | None = None
    version: str = "inconnue"
    store: OnlineStore | None = None
    queue: AlertQueue | None = None
    # Seuil réellement appliqué : celui du modèle au démarrage, puis recalé sur les scores
    # récents pour tenir le budget d'alertes.
    threshold: float = 0.5
    # Moins l'infini : la toute première occasion de recaler ne doit pas dépendre de la
    # valeur qu'affiche l'horloge monotone de la machine au démarrage.
    recalibrated_at: float = float("-inf")

    def load(self) -> None:
        """Charge le modèle promu et ouvre le magasin. Échouer ici évite de servir à vide."""
        self.store = OnlineStore.from_url(
            self.settings.redis_url, self.settings.pseudonymisation_key
        )
        self.queue = AlertQueue(self.settings.pg_dsn) if self.settings.pg_dsn else None
        self.model, self.version = load_model(self.ml, self.settings.model_alias)
        self.threshold = self.model.threshold
        self._warm_up()
        MODEL_READY.set(1)
        MODEL_THRESHOLD.set(self.model.threshold)

    def recalibrate(self) -> None:
        """Recale le seuil sur la distribution récente des scores, pour tenir le budget.

        Le budget d'alertes est une contrainte d'organisation : l'équipe fraude examine une
        transaction sur mille. Un seuil figé en probabilité ne tient pas cette promesse — sur
        ces données, 4 % d'écart de seuil change le volume d'alertes d'un facteur cent. On
        applique donc le budget comme ce qu'il est : un quantile de la population décidée.

        Chaque instance de l'API recale depuis la **même** fenêtre partagée : elles
        convergent vers le même seuil sans avoir à se coordonner.
        """
        settings = self.settings
        now = time.monotonic()
        if self.store is None or now - self.recalibrated_at < settings.recalibration_interval_s:
            return
        try:
            scores = self.store.recent_scores(settings.score_window)
        except redis.RedisError as error:
            logger.warning("Seuil non recalé (magasin injoignable) : %s", error)
            return
        if len(scores) < settings.recalibration_min_samples:
            # Trop peu d'observations : un quantile extrême n'y voudrait rien dire. Le délai
            # n'est délibérément **pas** consommé : au démarrage, le seuil doit se caler dès
            # que la fenêtre est assez remplie, sans attendre l'intervalle suivant.
            return
        self.recalibrated_at = now
        budget = self.model.alert_budget if self.model else DEFAULT_ALERT_BUDGET
        recalibrated = float(np.quantile(scores, 1 - budget))
        if recalibrated != self.threshold:
            logger.info(
                "Seuil recalé sur %d transactions : %.6f (était %.6f)",
                len(scores),
                recalibrated,
                self.threshold,
            )
        self.threshold = recalibrated
        MODEL_THRESHOLD.set(recalibrated)
        RECALIBRATIONS.inc()

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
            build=BUILD,
            modele=service.ml.model_name if service.model else None,
            modele_version=service.version if service.model else None,
            seuil=service.threshold if service.model else None,
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
            seuil=service.threshold if service.model else None,
            magasin_en_ligne=online,
        )

    @application.get("/metrics", tags=["exploitation"])
    def metrics() -> Response:
        """Métriques Prometheus : volume, latence, décisions, mode dégradé."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @application.get("/alertes", response_class=HTMLResponse, tags=["analyste"])
    def file_alertes(request: Request, etat: str = "a_traiter") -> HTMLResponse:
        """File de travail des analystes : les alertes à examiner, les plus risquées d'abord."""
        queue = _queue(request)
        if etat not in STATUSES:
            raise HTTPException(404, "État inconnu.")
        alertes = queue.pending(request.app.state.service.settings.alerts_page_size, etat)
        return HTMLResponse(queue_page(alertes, queue.counts(), etat))

    @application.get("/alertes/{transaction_id}", response_class=HTMLResponse, tags=["analyste"])
    def detail_alerte(transaction_id: str, request: Request, message: str = "") -> HTMLResponse:
        """Détail d'une alerte : ce que le modèle a vu, et la décision de l'analyste."""
        alerte = _queue(request).get(transaction_id)
        if alerte is None:
            raise HTTPException(404, "Alerte inconnue.")
        return HTMLResponse(alert_page(alerte, message))

    @application.post("/alertes/{transaction_id}", tags=["analyste"])
    def decider_alerte(
        transaction_id: str,
        request: Request,
        analyste: str = Form(..., min_length=2, max_length=64),
        etat: str = Form(...),
        note: str = Form("", max_length=1000),
    ) -> RedirectResponse:
        """Enregistre la décision d'un analyste, tracée au journal d'audit.

        Redirection après enregistrement : recharger la page ne rejoue pas la décision, et le
        bouton « précédent » du navigateur reste sans surprise.
        """
        if etat not in ("fraude_confirmee", "ecartee"):
            raise HTTPException(422, "Suite donnée invalide.")
        if not _queue(request).review(transaction_id, etat, analyste, note):
            raise HTTPException(404, "Alerte inconnue.")
        return RedirectResponse(
            f"/alertes/{transaction_id}?message=Décision+enregistrée.", status_code=303
        )

    @application.post("/scorer", response_model=Decision, tags=["décision"])
    def scorer(transaction: Transaction, request: Request) -> Response:
        """Rend la décision pour une demande d'autorisation."""
        service: Service = request.app.state.service
        if service.model is None:
            ERRORS.labels("modele_absent").inc()
            raise HTTPException(503, "Aucun modèle chargé : vérifier le registre MLflow.")
        return _decide(service, transaction)


def _queue(request: Request) -> AlertQueue:
    """File d'alertes, ou une erreur explicite si l'interface analyste n'est pas configurée."""
    queue = request.app.state.service.queue
    if queue is None:
        raise HTTPException(503, "Interface analyste non configurée (ONDINE_API_PG_DSN).")
    return queue


def _decide(service: Service, transaction: Transaction) -> Response:
    """Lit le magasin, calcule les variables, score, explique et répond."""
    started = time.perf_counter()
    payload = transaction.model_dump()
    features, state, degraded = _features(service, payload)
    # Un seul prétraitement pour la probabilité *et* l'explication.
    matrix = service.model.row_matrix(features)
    probability = float(service.model.probabilities(matrix)[0])
    alert = probability >= service.threshold
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
        seuil=service.threshold,
        modele_version=service.version,
        latence_ms=round(elapsed * 1000, 3),
        mode_degrade=degraded,
        explication=sentence(factors),
        facteurs=factors,
    )
    # Écriture après la réponse : la transaction suivante de la même carte voit déjà celle-ci
    # sans que l'appelant en paie la latence. Le flux Kafka réécrira le même membre.
    tasks = BackgroundTasks()
    if not degraded and state is not None:
        tasks.add_task(_remember, service, payload, probability)
    tasks.add_task(_record_alert, service, payload, decision)
    return JSONResponse(decision.model_dump(), background=tasks)


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


def _record_alert(service: Service, payload: dict, decision: Decision) -> None:
    """Dépose l'alerte dans la file des analystes (après la réponse, jamais bloquant)."""
    if service.queue is None or decision.decision != "bloquer":
        return
    try:
        service.queue.record(
            {
                "transaction_id": decision.transaction_id,
                "event_time": payload["event_time"],
                "client_id": payload["client_id"],
                "amount": payload["amount"],
                "channel": payload.get("channel"),
                "merchant_country": payload.get("merchant_country"),
                "probability": decision.probabilite_fraude,
                "threshold": decision.seuil,
                "model_version": decision.modele_version,
                "explanation": decision.explication,
                "factors": [f.model_dump() for f in decision.facteurs],
            }
        )
    except Exception as error:  # noqa: BLE001 — une file indisponible ne bloque pas un paiement
        ALERTS_FAILED.inc()
        logger.error("Alerte non déposée pour %s : %s", decision.transaction_id, error)


def _remember(service: Service, payload: dict, probability: float) -> None:
    """Après la réponse : mémorise la transaction, le score, et recale le seuil si besoin."""
    try:
        service.store.record_transaction(payload)
        service.store.record_score(probability, service.settings.score_window)
        service.recalibrate()
    except redis.RedisError as error:  # l'écriture est rattrapée par le flux Kafka
        DEGRADED.labels("ecriture_differee").inc()
        logger.warning("Transaction non écrite dans le magasin en ligne : %s", error)


def build() -> FastAPI:
    """Fabrique appelée par uvicorn (`ondine_api.app:build --factory`) et par les tests."""
    return create_app()
