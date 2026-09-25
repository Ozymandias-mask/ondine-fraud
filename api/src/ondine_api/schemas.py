"""Contrats d'entrée et de sortie de l'API de scoring.

Le contrat d'entrée reprend **exactement** le message d'autorisation déjà publié par le
système de paiement dans Kafka (`transactions.raw`) : aucun format nouveau à produire côté
appelant, et la même transaction peut être rejouée depuis le flux pour reproduire une
décision. Les champs inconnus sont refusés : une faute de frappe sur un nom de variable
donnerait sinon une transaction scorée avec une donnée manquante, sans que personne ne le voie.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """Demande d'autorisation soumise au modèle, avant la décision de la banque."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1, max_length=64)
    event_time: datetime
    client_id: str = Field(min_length=1, max_length=64)
    card_id: str | None = Field(default=None, max_length=64)
    merchant_id: str | None = Field(default=None, max_length=64)
    mcc: int | None = Field(default=None, ge=0, le=9999)
    merchant_country: str | None = Field(default=None, max_length=2)
    channel: str = Field(max_length=32)
    entry_mode: str | None = Field(default=None, max_length=32)
    amount: float = Field(ge=0, le=1_000_000)
    currency: str = Field(default="EUR", max_length=3)
    three_ds_authenticated: bool | None = None
    # Données indirectement identifiantes : pseudonymisées immédiatement, jamais stockées
    # en clair ni renvoyées dans la réponse (minimisation RGPD).
    device_id: str | None = Field(default=None, max_length=128)
    beneficiary_iban: str | None = Field(default=None, max_length=34)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    wallet_type: str | None = Field(default=None, max_length=32)


class Factor(BaseModel):
    """Un facteur ayant pesé sur la décision, tel que produit par l'explicabilité du modèle.

    Les noms de champs reprennent ceux de `ondine_ml.explain` : une seule définition des
    explications, partagée par l'API, l'interface analyste et les rapports d'entraînement.
    """

    variable: str
    libellé: str
    valeur: float | str | None
    effet: str
    poids: float


class Decision(BaseModel):
    """Réponse rendue au système d'autorisation."""

    transaction_id: str
    probabilite_fraude: float
    decision: str
    seuil: float
    modele_version: str
    latence_ms: float
    mode_degrade: bool
    explication: str
    facteurs: list[Factor]


class Health(BaseModel):
    """État de l'API : modèle chargé, magasin en ligne joignable."""

    statut: str
    # Version du code déployé (empreinte Git inscrite à la construction de l'image) : savoir
    # *quel* code répond est la première question d'une astreinte, et la seule que les
    # journaux ne répondent pas toujours.
    build: str = "inconnu"
    modele: str | None = None
    modele_version: str | None = None
    seuil: float | None = None
    # Renseigné par `/pret` seulement : `/sante` ne sollicite aucune dépendance.
    magasin_en_ligne: bool | None = None
