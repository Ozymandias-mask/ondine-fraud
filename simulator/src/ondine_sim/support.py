"""Déclarations de fraude (labels) et e-mails du service client (données non structurées)."""

from __future__ import annotations

from email.message import EmailMessage
from email.utils import format_datetime

import numpy as np
import pandas as pd

from .config import SimConfig
from .entities import client_contacts
from .identifiers import random_hex_ids

SUPPORT_ADDRESS = "support@ondine-bank.example"

# Délai médian de déclaration (jours) et canal le plus probable, par type de fraude.
_REPORT_PROFILE = {
    "card_testing": (9.0, "chargeback"),
    "cloned_card": (4.0, "app_claim"),
    "account_takeover": (1.5, "phone"),
    "sim_swap": (1.0, "phone"),
    "app_scam": (2.0, "email"),
    "wallet_enrolment": (6.0, "app_claim"),
}
CLAIM_LABELS = {
    "card_testing": "paiement_a_distance_non_reconnu",
    "cloned_card": "carte_contrefaite",
    "account_takeover": "acces_compte_non_autorise",
    "sim_swap": "acces_compte_non_autorise",
    "app_scam": "manipulation_virement",
    "wallet_enrolment": "paiement_mobile_non_reconnu",
}


def _naive_utc(values: pd.Series) -> pd.Series:
    """Convertit des horodatages ISO « ...Z » en datetime UTC sans fuseau (calculs)."""
    return pd.to_datetime(values, utc=True).dt.tz_convert(None)


def fraud_reports(
    rng: np.random.Generator, cfg: SimConfig, truth: pd.DataFrame, legit_candidates: pd.DataFrame
) -> pd.DataFrame:
    """Déclarations de fraude reçues par la banque, avec délai et bruit réalistes.

    - 10 % des incidents ne sont jamais déclarés (labels manquants) ;
    - le délai dépend du mode opératoire (virement : jours ; carte : relevé mensuel) ;
    - quelques contestations portent sur des opérations légitimes (rejetées après enquête),
      tirées parmi `legit_candidates` (échantillon d'opérations approuvées).
    """
    columns = [
        "report_id",
        "case_id",
        "transaction_id",
        "client_id",
        "reported_at",
        "report_channel",
        "claim_category",
        "investigation_status",
    ]
    parts = []
    if len(truth):
        incidents = (
            truth.groupby("incident_id")
            .agg(
                client_id=("client_id", "first"),
                fraud_type=("fraud_type", "first"),
                last_time=("event_time", "max"),
            )
            .reset_index()
        )
        reported = incidents[rng.random(len(incidents)) < cfg.fraud_report_rate].copy()
        median = reported["fraud_type"].map(lambda t: _REPORT_PROFILE[t][0]).to_numpy()
        delay_h = np.clip(rng.lognormal(np.log(median * 24), 0.6), 1, 90 * 24)
        reported["reported_at"] = _naive_utc(reported["last_time"]) + pd.to_timedelta(
            delay_h, unit="h"
        )
        usual = reported["fraud_type"].map(lambda t: _REPORT_PROFILE[t][1]).to_numpy()
        other = np.array(["app_claim", "email", "phone"])[rng.integers(0, 3, len(reported))]
        reported["report_channel"] = np.where(rng.random(len(reported)) < 0.7, usual, other)
        rows = truth.merge(
            reported[["incident_id", "reported_at", "report_channel"]], on="incident_id"
        )
        rows["claim_category"] = rows["fraud_type"].map(CLAIM_LABELS)
        rows["investigation_status"] = "confirmed"
        rows["case_id"] = rows["incident_id"].str.replace("INC", "CASE", regex=False)
        parts.append(rows)

    # Contestations infondées (« fraude amicale ») sur des opérations légitimes approuvées.
    legit = legit_candidates[~legit_candidates["transaction_id"].isin(truth["transaction_id"])]
    n_false = min(len(legit), int(len(truth) * 0.04) + 1) if len(legit) else 0
    if n_false:
        sample = legit.sample(n=n_false, random_state=int(rng.integers(0, 2**31)))
        false = pd.DataFrame(
            {
                "transaction_id": sample["transaction_id"].to_numpy(),
                "client_id": sample["client_id"].to_numpy(),
                "reported_at": _naive_utc(sample["event_time"]).to_numpy()
                + pd.to_timedelta(rng.uniform(24, 30 * 24, n_false), unit="h").to_numpy(),
                "report_channel": np.array(["app_claim", "email"])[rng.integers(0, 2, n_false)],
                "claim_category": "paiement_a_distance_non_reconnu",
                "investigation_status": "rejected",
                "case_id": np.char.add("CASE-R", random_hex_ids(rng, n_false, 5)),
            }
        )
        parts.append(false)

    if not parts:
        return pd.DataFrame(columns=columns)
    reports = pd.concat(parts, ignore_index=True)
    reports["reported_at"] = pd.to_datetime(reports["reported_at"]).dt.strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    reports["report_id"] = np.char.add("R", random_hex_ids(rng, len(reports), 8))
    return reports[columns]


# --------------------------------------------------------------------------- e-mails

_GREETINGS = [
    "Bonjour,",
    "Bonjour Madame, Monsieur,",
    "Bonsoir,",
    "Madame, Monsieur,",
    "Bonjour à l'équipe Ondine,",
    "Salut,",
]
_CLOSINGS = [
    "Cordialement,",
    "Merci d'avance,",
    "Bien à vous,",
    "Merci de me répondre vite.",
    "Dans l'attente de votre retour,",
    "Merci.",
]

_FRAUD_TEMPLATES = {
    "card_testing": [
        "Je viens de voir sur mon relevé plusieurs petits paiements de moins de 3 euros sur des "
        "sites que je ne connais pas, puis un achat de {amount} euros. Ce n'est pas moi ! "
        "Ma carte finissant par {last4} est toujours en ma possession.",
        "Des paiements en ligne que je n'ai jamais faits apparaissent sur mon compte depuis le "
        "{date}. Le plus gros fait {amount} EUR. Merci de bloquer ma carte {last4}.",
    ],
    "cloned_card": [
        "On a retiré de l'argent avec ma carte à l'étranger alors que je suis à {city} ! "
        "J'ai {amount} euros de retraits que je ne reconnais pas depuis le {date}.",
        "Ma carte a été utilisée dans un pays où je n'ai jamais mis les pieds. Montant : "
        "{amount} €. Je pense qu'elle a été copiée dans un distributeur.",
    ],
    "account_takeover": [
        "Je ne peux plus me connecter à mon appli et j'ai reçu un mail de changement de mot de "
        "passe que je n'ai pas demandé. Un virement de {amount} euros est parti de mon compte "
        "{iban} !! C'est une urgence.",
        "Quelqu'un a pris le contrôle de mon espace client le {date}, un bénéficiaire a été "
        "ajouté et {amount} EUR ont été virés. Rappelez-moi au {phone}.",
    ],
    "sim_swap": [
        "Mon téléphone n'avait plus de réseau hier soir et ce matin je découvre un virement de "
        "{amount} euros que je n'ai pas fait. Mon opérateur parle d'une nouvelle carte SIM.",
        "Plus de réseau mobile depuis le {date} et un virement de {amount} € vers un compte "
        "inconnu. Je pense que ma ligne a été piratée. Mon numéro : {phone}.",
    ],
    "app_scam": [
        "Un conseiller de chez vous m'a appelé en disant que mon compte était piraté et qu'il "
        "fallait mettre mon argent en sécurité. J'ai fait un virement de {amount} euros sur le "
        "compte qu'il m'a donné. Je comprends maintenant que c'était une arnaque...",
        "J'ai été victime d'un faux conseiller le {date}. Il connaissait mon nom et mon adresse. "
        "J'ai validé moi-même un virement de {amount} €. Est-ce que vous pouvez le récupérer ?",
    ],
    "wallet_enrolment": [
        "Des paiements sans contact ont été faits avec ma carte {last4} dans des magasins où je "
        "ne suis jamais allé, pour {amount} euros au total. J'ai reçu un code par SMS pour "
        "« ajouter la carte à un wallet » que j'ai donné par erreur.",
        "Je vois des achats chez des commerçants d'une autre ville le {date}, payés par "
        "téléphone, alors que je n'utilise pas Apple Pay ni Google Pay. Montant : {amount} €.",
    ],
}

_OTHER_TEMPLATES = {
    "carte_bloquee": (
        0.14,
        [
            "Ma carte a été refusée plusieurs fois aujourd'hui alors que j'ai de l'argent sur mon "
            "compte. C'est très gênant, j'étais à la caisse. Pouvez-vous la débloquer ?",
            "Paiement refusé pour « suspicion de fraude » alors que c'était bien moi. Merci de ne "
            "plus bloquer ma carte {last4} sans me prévenir.",
        ],
    ),
    "frais_contestation": (
        0.15,
        [
            "Je conteste les frais de {amount} euros prélevés ce mois-ci, je n'ai rien fait "
            "d'inhabituel. Merci de me rembourser.",
            "Pourquoi ai-je été facturé de frais d'incident ? Mon compte {iban} n'a jamais été à "
            "découvert.",
        ],
    ),
    "virement_retard": (
        0.12,
        [
            "J'ai fait un virement instantané de {amount} euros il y a deux jours et mon "
            "destinataire ne l'a toujours pas reçu. Que se passe-t-il ?",
            "Un virement attendu n'apparaît pas sur mon compte {iban}. Pouvez-vous vérifier ?",
        ],
    ),
    "probleme_application": (
        0.20,
        [
            "L'application plante à chaque ouverture depuis la dernière mise à jour. J'ai un "
            "téléphone Android.",
            "Impossible de valider mes paiements, la notification de validation n'arrive jamais "
            "sur mon iPhone.",
            "L'appli ne fonctionne pas avec le lecteur d'écran de mon téléphone, je suis malvoyant "
            "et je ne peux pas consulter mon solde.",
        ],
    ),
    "plafond_carte": (
        0.12,
        [
            "Je voudrais augmenter temporairement mon plafond de paiement à {amount} euros pour un "
            "achat important.",
            "Mon plafond de retrait est trop bas, comment l'augmenter ?",
        ],
    ),
    "demande_rgpd_acces": (
        0.04,
        [
            "Conformément au RGPD, je souhaite obtenir une copie de toutes les données "
            "personnelles que vous détenez sur moi. Mon adresse e-mail client est {email}.",
            "Je demande l'accès à mes données personnelles (article 15 du RGPD) et à la logique "
            "des décisions automatiques qui ont bloqué ma carte.",
        ],
    ),
    "demande_rgpd_suppression": (
        0.03,
        [
            "Je souhaite que vous supprimiez toutes mes données personnelles (article 17 du RGPD) "
            "et que vous clôturiez mon compte {iban}.",
            "Merci d'effacer mes données et de ne plus m'envoyer de communication commerciale.",
        ],
    ),
    "cloture_compte": (
        0.05,
        [
            "Je souhaite clôturer mon compte. Merci de virer le solde sur mon nouveau compte.",
            "Comment fermer mon compte Ondine ? Je n'arrive pas à trouver l'option dans l'appli.",
        ],
    ),
    "changement_coordonnees": (
        0.15,
        [
            "J'ai déménagé, ma nouvelle adresse est le {street_number} rue des Lilas à {city}. "
            "Mon numéro reste le {phone}.",
            "Je change de numéro de téléphone, merci de mettre à jour mon dossier. Ancien numéro : "
            "{phone}.",
        ],
    ),
}
_OTHER_CATEGORIES = list(_OTHER_TEMPLATES)
_OTHER_PROBS = np.array([v[0] for v in _OTHER_TEMPLATES.values()])
_OTHER_PROBS = _OTHER_PROBS / _OTHER_PROBS.sum()


def _add_typos(text: str, rng: np.random.Generator) -> str:
    """Introduit quelques fautes de frappe (inversion de deux lettres)."""
    chars = list(text)
    for _ in range(int(rng.integers(1, 4))):
        i = int(rng.integers(1, len(chars) - 2))
        if chars[i].isalpha() and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def support_emails(
    rng: np.random.Generator,
    cfg: SimConfig,
    reports: pd.DataFrame,
    truth: pd.DataFrame,
    client_pool: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[list[bytes], pd.DataFrame]:
    """E-mails bruts (format mbox) et tickets associés classés par les conseillers.

    Les e-mails contiennent volontairement des données personnelles (nom, IBAN,
    téléphone, adresse) : le pipeline devra les détecter et les masquer.
    """
    requests: list[tuple[int, str, pd.Timestamp, str | None]] = []

    # 1. Déclarations de fraude faites par e-mail (une par dossier).
    by_email = reports[
        (reports["report_channel"] == "email") & (reports["investigation_status"] == "confirmed")
    ]
    if len(by_email):
        cases = by_email.merge(
            truth[["transaction_id", "fraud_type", "amount"]], on="transaction_id"
        )
        cases = (
            cases.groupby("case_id")
            .agg(
                client_id=("client_id", "first"),
                fraud_type=("fraud_type", "first"),
                reported_at=("reported_at", "first"),
                amount=("amount", "sum"),
            )
            .reset_index()
        )
        for row in cases.itertuples():
            received = pd.Timestamp(row.reported_at).tz_localize(None)
            requests.append(
                (
                    int(row.client_id[1:]),
                    "signalement_fraude",
                    received,
                    f"{row.fraud_type}|{row.amount:.2f}",
                )
            )

    # 2. Autres demandes, réparties sur la période.
    n_other = rng.poisson(
        len(client_pool) * cfg.support_emails_per_client_month * max((end - start).days, 1) / 30.4
    )
    senders = rng.choice(client_pool, size=n_other) if len(client_pool) else np.array([], int)
    seconds = rng.integers(0, int((end - start).total_seconds()), n_other)
    cats = rng.choice(len(_OTHER_CATEGORIES), size=n_other, p=_OTHER_PROBS)
    for sender, sec, cat in zip(senders, seconds, cats, strict=True):
        requests.append(
            (int(sender), _OTHER_CATEGORIES[cat], start + pd.Timedelta(seconds=int(sec)), None)
        )

    if not requests:
        return [], pd.DataFrame(
            columns=[
                "ticket_id",
                "message_id",
                "client_id",
                "received_at",
                "channel",
                "category",
                "priority",
            ]
        )

    client_idx = np.array([r[0] for r in requests])
    contacts = client_contacts(cfg, client_idx)
    messages, tickets = [], []
    for i, (_, category, received, extra) in enumerate(requests):
        msg, message_id = _build_email(rng, contacts, i, category, received, extra)
        messages.append(msg)
        # Classement par le conseiller : 4 % d'erreurs de catégorie (bruit réaliste).
        assigned = category
        if rng.random() < 0.04:
            assigned = _OTHER_CATEGORIES[int(rng.integers(0, len(_OTHER_CATEGORIES)))]
        tickets.append(
            {
                "ticket_id": f"TCK{int(rng.integers(0, 16**10)):010x}",
                "message_id": message_id,
                "client_id": contacts["client_id"][i],
                "received_at": received.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "channel": "email",
                "category": assigned,
                "priority": "high"
                if category
                in ("signalement_fraude", "demande_rgpd_acces", "demande_rgpd_suppression")
                else "normal",
            }
        )
    return messages, pd.DataFrame(tickets)


def _build_email(
    rng: np.random.Generator,
    contacts: dict[str, np.ndarray],
    i: int,
    category: str,
    received: pd.Timestamp,
    extra: str | None,
) -> tuple[bytes, str]:
    amount = f"{rng.lognormal(4.5, 0.9):.2f}".replace(".", ",")
    if category == "signalement_fraude" and extra:
        fraud_type, total = extra.split("|")
        template = _FRAUD_TEMPLATES[fraud_type][int(rng.integers(0, 2))]
        amount = total.replace(".", ",")
        subject = rng.choice(
            [
                "URGENT fraude sur mon compte",
                "Opérations non reconnues",
                "Arnaque",
                "Fraude !!!",
                "Signalement d'opérations frauduleuses",
            ]
        )
    else:
        options = _OTHER_TEMPLATES[category][1]
        template = options[int(rng.integers(0, len(options)))]
        subject = category.replace("_", " ").capitalize()

    body = template.format(
        amount=amount,
        last4=f"{int(rng.integers(0, 10_000)):04d}",
        date=(received - pd.Timedelta(days=int(rng.integers(0, 4)))).strftime("%d/%m/%Y"),
        city=rng.choice(["Paris", "Lyon", "Nantes", "Lille", "Toulouse"]),
        iban=contacts["iban"][i],
        phone=contacts["phone"][i],
        email=contacts["email"][i],
        street_number=int(rng.integers(1, 120)),
    )
    if rng.random() < 0.15:
        body = _add_typos(body, rng)
    signature = f"{contacts['first_name'][i]} {contacts['last_name'][i]}"
    if rng.random() < 0.3:
        signature += f"\nTél : {contacts['phone'][i]}"
    text = f"{rng.choice(_GREETINGS)}\n\n{body}\n\n{rng.choice(_CLOSINGS)}\n{signature}\n"

    msg = EmailMessage()
    # Identifiant tiré du générateur (et non de l'horloge) : génération reproductible.
    message_id = f"<{int(rng.integers(0, 2**62)):016x}.{i}@mail.example>"
    msg["From"] = f"{contacts['first_name'][i]} {contacts['last_name'][i]} <{contacts['email'][i]}>"
    msg["To"] = SUPPORT_ADDRESS
    msg["Subject"] = str(subject)
    msg["Date"] = format_datetime(received.tz_localize("UTC").to_pydatetime())
    msg["Message-ID"] = message_id
    msg.set_content(text, cte="8bit")  # texte lisible tel quel dans le fichier
    envelope = f"From {contacts['email'][i]} {received.strftime('%a %b %d %H:%M:%S %Y')}\n"
    return envelope.encode() + msg.as_bytes() + b"\n", message_id
