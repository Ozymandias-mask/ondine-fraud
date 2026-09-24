"""Magasin de variables en ligne (Redis), alimenté par le flux Kafka.

Pourquoi un magasin en ligne : la couche gold calcule les mêmes variables en SQL sur des
mois entiers, ce qui prend des minutes. Une autorisation de paiement doit être décidée en
moins de 100 ms, avant que le client ne retire sa carte. On garde donc en mémoire, par carte,
par client, par appareil et par bénéficiaire, le strict historique nécessaire aux fenêtres
glissantes (1 h, 24 h, 7 jours, 30 jours), et rien de plus.

Principes :

- **mêmes clés que le lac** : les identifiants indirects (appareil, bénéficiaire) sont
  pseudonymisés avec la *même* fonction et la *même* clé que la couche silver ; un même
  appareil a donc le même pseudonyme en ligne et hors ligne ;
- **minimisation RGPD** : aucune donnée directement identifiante n'entre ici (pas d'IBAN,
  pas d'adresse IP, pas de nom), et chaque clé porte une durée de vie égale à la fenêtre
  qu'elle sert — l'historique s'efface tout seul ;
- **idempotence** : un message rejoué réécrit le même membre (identifiant métier) dans le
  même ensemble trié ; le relire deux fois ne compte pas deux transactions ;
- **coût borné** : à chaque écriture, les entrées sorties de la fenêtre sont supprimées.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

import redis

logger = logging.getLogger(__name__)

SEPARATOR = "\t"  # absent des identifiants métier

# Types d'événements retenus dans la chronologie du client : exactement ceux qu'utilise la
# couche gold (`EVENT_TYPES_USED`). Les autres n'entrent pas dans le magasin — ils ne servent
# à aucune variable et compteraient à tort comme « activité » dans les fenêtres.
EVENT_TYPES_USED = frozenset(
    {
        "login",
        "login_failed",
        "password_reset",
        "beneficiary_added",
        "sca_method_changed",
        "wallet_provisioning",
        "sca_challenge",
    }
)

HOUR = 3600
DAY = 24 * HOUR
# Durées de vie : la fenêtre la plus longue servie par la clé, avec une marge d'un jour.
# La carte est conservée 31 jours et non 24 h : le « délai depuis le paiement précédent »
# porte sur 30 jours, comme dans la couche gold.
CARD_TTL = 31 * DAY
CLIENT_TTL = 31 * DAY
EVENT_TTL = 8 * DAY
PHYSICAL_TTL = 31 * DAY
# Les « première apparition » servent à l'ancienneté d'un appareil : conservées un an, durée
# alignée sur la politique de rétention du lac.
FIRST_SEEN_TTL = 366 * DAY


def epoch(value: str | datetime | float) -> float:
    """Horodatage en secondes depuis 1970 (UTC), quel que soit le format reçu."""
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        moment = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


@dataclass
class State:
    """Photographie de l'historique utile à une transaction, à l'instant de la décision."""

    card: list[tuple[float, list[str]]] = field(default_factory=list)
    card_declines: list[float] = field(default_factory=list)
    client: list[tuple[float, list[str]]] = field(default_factory=list)
    events: list[tuple[float, list[str]]] = field(default_factory=list)
    previous_physical: tuple[float, float, float] | None = None
    device_first_seen: float | None = None
    beneficiary_first_seen: float | None = None
    beneficiary_new_senders_30d: int = 0
    merchant: dict[str, str] = field(default_factory=dict)


def _nested(message: dict, name: str) -> dict:
    """Sous-objet d'un message (`session`, `payload`), qu'il soit déjà décodé ou en texte."""
    value = message.get(name)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _pack(*parts: object) -> str:
    return SEPARATOR.join("" if p is None else str(p) for p in parts)


def _unpack(rows: list[tuple[str, float]]) -> list[tuple[float, list[str]]]:
    return [(score, member.split(SEPARATOR)) for member, score in rows]


class OnlineStore:
    """Lecture et écriture de l'historique glissant utilisé par l'API de scoring."""

    def __init__(self, client: redis.Redis, pseudonymisation_key: str) -> None:
        """Magasin adossé à un client Redis déjà ouvert (le même en test et en production)."""
        self.redis = client
        self._key = pseudonymisation_key

    @classmethod
    def from_url(cls, url: str, pseudonymisation_key: str) -> OnlineStore:
        """Client Redis prêt à l'emploi (réponses décodées, délais courts)."""
        client = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
            health_check_interval=30,
        )
        return cls(client, pseudonymisation_key)

    # ------------------------------------------------------------------ pseudonymisation
    def pseudo(self, value: str | None) -> str | None:
        """Pseudonyme identique à celui de la couche silver (SHA-256 avec clé, 32 caractères)."""
        if value in (None, ""):
            return None
        return hashlib.sha256(f"{self._key}|{value}".encode()).hexdigest()[:32]

    # -------------------------------------------------------------------------- écriture
    def record_transaction(self, tx: dict) -> None:
        """Enregistre une transaction autorisée dans les historiques carte, client et trajet."""
        moment = epoch(tx["event_time"])
        identifier = str(tx["transaction_id"])
        pipe = self.redis.pipeline(transaction=False)
        if tx.get("card_id"):
            key = f"of:card:{tx['card_id']}"
            # Le membre ne contient que des attributs connus dès la demande d'autorisation :
            # l'API (qui écrit avant de connaître le sort de la transaction) et le flux
            # écrivent donc exactement le même membre, et rejouer n'ajoute rien.
            member = _pack(
                identifier, tx.get("amount", 0), tx.get("merchant_id"), tx.get("channel")
            )
            pipe.zadd(key, {member: moment})
            pipe.zremrangebyscore(key, 0, moment - CARD_TTL)
            pipe.expire(key, CARD_TTL)
            if tx.get("auth_status") == "declined":
                # Les refus ne sont connus qu'après coup : ensemble séparé, alimenté par le flux.
                # Une seule fenêtre les utilise (1 h), d'où une durée de vie bien plus courte.
                key = f"of:carddecl:{tx['card_id']}"
                pipe.zadd(key, {identifier: moment})
                pipe.zremrangebyscore(key, 0, moment - 25 * HOUR)
                pipe.expire(key, 25 * HOUR)
        if tx.get("client_id"):
            key = f"of:cli:{tx['client_id']}"
            member = _pack(
                identifier, tx.get("amount", 0), tx.get("channel"), tx.get("merchant_country")
            )
            pipe.zadd(key, {member: moment})
            pipe.zremrangebyscore(key, 0, moment - CLIENT_TTL)
            pipe.expire(key, CLIENT_TTL)
            if tx.get("latitude") is not None and tx.get("longitude") is not None:
                key = f"of:phys:{tx['client_id']}"
                pipe.zadd(key, {_pack(identifier, tx["latitude"], tx["longitude"]): moment})
                pipe.zremrangebyscore(key, 0, moment - PHYSICAL_TTL)
                pipe.expire(key, PHYSICAL_TTL)
            device = self.pseudo(tx.get("device_id"))
            if device:
                self._first_seen(pipe, f"of:dev:{tx['client_id']}", device, moment)
            beneficiary = self.pseudo(tx.get("beneficiary_iban"))
            if beneficiary:
                self._first_seen(pipe, f"of:ben:{tx['client_id']}", beneficiary, moment)
                # Compte des payeurs distincts d'un bénéficiaire : un compte « mule » voit
                # arriver plusieurs nouveaux payeurs en quelques jours.
                senders = f"of:bensend:{beneficiary}"
                pipe.zadd(senders, {str(tx["client_id"]): moment}, nx=True)
                pipe.expire(senders, FIRST_SEEN_TTL)
        pipe.execute()

    def record_event(self, event: dict) -> None:
        """Enregistre un événement applicatif (connexion, réinitialisation, enrôlement…).

        Le message reçu est celui du topic `app-events.raw` : les blocs `session` et
        `payload` y sont imbriqués (et parfois transmis sous forme de texte JSON).
        """
        client = event.get("client_id")
        if not client:
            return
        moment = epoch(event["event_time"])
        session, payload = _nested(event, "session"), _nested(event, "payload")
        device = self.pseudo(session.get("device_id"))
        # « Nouvel appareil » est figé à l'écriture : au moment du scoring, on ne saurait plus
        # dire si l'appareil était déjà connu lors de cette connexion.
        new_device = 0
        if device:
            key = f"of:dev:{client}"
            first = self.redis.hget(key, device)
            new_device = int(first is None or float(first) >= moment)
            self._first_seen(self.redis, key, device, moment)
        if event.get("event_type") not in EVENT_TYPES_USED:
            # La première apparition de l'appareil est tout de même mémorisée ci-dessus :
            # la couche gold la calcule elle aussi sur *tous* les événements.
            return
        key = f"of:ev:{client}"
        member = _pack(
            event.get("event_id"),
            event.get("event_type"),
            session.get("ip_country"),
            payload.get("method"),
            new_device,
        )
        pipe = self.redis.pipeline(transaction=False)
        pipe.zadd(key, {member: moment})
        pipe.zremrangebyscore(key, 0, moment - EVENT_TTL)
        pipe.expire(key, EVENT_TTL)
        pipe.execute()

    def load_merchants(self, rows: list[dict]) -> int:
        """Charge le référentiel marchands (catégorie, en ligne) depuis la couche silver.

        Ce référentiel change lentement : il est rechargé par une tâche planifiée, pas à
        chaque transaction. Sans lui, deux variables du modèle seraient manquantes.
        """
        pipe = self.redis.pipeline(transaction=False)
        for row in rows:
            pipe.hset(
                f"of:merchant:{row['merchant_id']}",
                mapping={
                    "category": str(row.get("category") or ""),
                    "is_online": str(int(bool(row.get("is_online")))),
                },
            )
        pipe.execute()
        return len(rows)

    @staticmethod
    def _first_seen(pipe, key: str, member: str, moment: float) -> None:
        """Mémorise la première apparition (jamais écrasée par une occurrence postérieure)."""
        pipe.hsetnx(key, member, moment)
        pipe.expire(key, FIRST_SEEN_TTL)

    # -------------------------------------------------------------------------- lecture
    def state(self, tx: dict) -> State:
        """Lit en un seul aller-retour tout l'historique nécessaire au scoring.

        Les fenêtres sont bornées à droite par l'horodatage de la transaction et **excluent**
        la transaction en cours, qui n'est pas encore écrite : c'est exactement la définition
        des fenêtres de la couche gold (`EXCLUDE CURRENT ROW`).
        """
        moment = epoch(tx["event_time"])
        client, card = tx.get("client_id"), tx.get("card_id")
        device = self.pseudo(tx.get("device_id"))
        beneficiary = self.pseudo(tx.get("beneficiary_iban"))
        pipe = self.redis.pipeline(transaction=False)
        pipe.zrangebyscore(f"of:card:{card}", moment - 30 * DAY, moment, withscores=True)
        pipe.zrangebyscore(f"of:carddecl:{card}", moment - HOUR, moment, withscores=True)
        pipe.zrangebyscore(f"of:cli:{client}", moment - 30 * DAY, moment, withscores=True)
        pipe.zrangebyscore(f"of:ev:{client}", moment - 7 * DAY, moment, withscores=True)
        pipe.zrevrangebyscore(
            f"of:phys:{client}", f"({moment}", moment - 30 * DAY, 0, 1, withscores=True
        )
        # `echo` tient la place quand la clé n'a pas de sens : le nombre et l'ordre des
        # réponses restent constants, la lecture reste un seul aller-retour.
        pipe.hget(f"of:dev:{client}", device) if device else pipe.echo("")  # noqa: B018
        pipe.hget(f"of:ben:{client}", beneficiary) if beneficiary else pipe.echo("")  # noqa: B018
        pipe.hgetall(f"of:merchant:{tx.get('merchant_id')}")
        (
            card_rows,
            decline_rows,
            client_rows,
            event_rows,
            physical,
            device_seen,
            ben_seen,
            merchant,
        ) = pipe.execute()

        state = State(
            card=_unpack(card_rows),
            card_declines=[score for _, score in decline_rows],
            client=_unpack(client_rows),
            events=_unpack(event_rows),
            device_first_seen=float(device_seen) if device_seen else None,
            beneficiary_first_seen=float(ben_seen) if ben_seen else None,
            merchant=merchant or {},
        )
        if physical:
            member, when = physical[0]
            parts = member.split(SEPARATOR)
            state.previous_physical = (when, float(parts[1]), float(parts[2]))
        if beneficiary:
            # Première apparition de ce couple (client, bénéficiaire) : maintenant si le
            # virement est le premier. Les autres payeurs des 30 jours précédents comptent.
            first = state.beneficiary_first_seen or moment
            state.beneficiary_new_senders_30d = max(
                self.redis.zcount(f"of:bensend:{beneficiary}", first - 30 * DAY, first)
                - (1 if state.beneficiary_first_seen is not None else 0),
                0,
            )
        return state

    def ping(self) -> bool:
        """Disponibilité du magasin (sonde de mise en service)."""
        try:
            return bool(self.redis.ping())
        except redis.RedisError as error:
            logger.warning("Magasin en ligne injoignable : %s", error)
            return False
