"""Interface des analystes fraude, rendue côté serveur et accessible.

Pourquoi du HTML produit par le serveur, sans cadriciel ni JavaScript obligatoire : une
interface de travail bancaire doit rester utilisable au clavier, avec un lecteur d'écran, sur
un poste verrouillé, et continuer de fonctionner si un script ne se charge pas. Le RGAA
(déclinaison française des WCAG, obligatoire pour l'accessibilité) est plus facile à tenir —
et à démontrer — sur des pages simples et sémantiques.

Ce qui est mis en œuvre, et vérifiable dans le code comme par les tests :

- langue déclarée, titre unique et explicite par page, hiérarchie de titres respectée ;
- lien d'évitement vers le contenu, régions sémantiques (`header`, `nav`, `main`, `footer`) ;
- tableau de données avec légende et en-têtes de colonnes portées (`scope`) ;
- champs de formulaire tous associés à une étiquette visible ;
- aucune information portée par la seule couleur : chaque état porte aussi son libellé ;
- contrastes supérieurs à 4,5:1, focus clavier visible, respect de « mouvement réduit » ;
- messages de confirmation annoncés aux lecteurs d'écran (`aria-live`) ;
- mise en page fluide, lisible jusqu'à 200 % de zoom.
"""

from __future__ import annotations

from datetime import datetime
from html import escape

from .alerts import STATUSES, Alert

STYLE = """
:root {
  --fond: #ffffff; --texte: #1a1d21; --discret: #4a5057; --bordure: #c9ced4;
  --accent: #0b4f9e; --alerte: #8a1c1c; --ok: #14562c; --surface: #f4f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--fond); color: var(--texte);
  font: 1rem/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.evitement { position: absolute; left: -9999px; top: 0; background: var(--accent);
  color: #fff; padding: .75rem 1rem; z-index: 10; }
.evitement:focus { left: 0; }
header, main, footer { max-width: 62rem; margin: 0 auto; padding: 1rem; }
header { border-bottom: 2px solid var(--bordure); }
h1 { font-size: 1.5rem; margin: .25rem 0; }
h2 { font-size: 1.2rem; margin-top: 2rem; }
a { color: var(--accent); }
a:focus-visible, button:focus-visible, input:focus-visible, select:focus-visible,
textarea:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
table { border-collapse: collapse; width: 100%; margin-top: .5rem; }
caption { text-align: left; font-weight: 600; padding: .5rem 0; }
th, td { border: 1px solid var(--bordure); padding: .5rem .6rem; text-align: left;
  vertical-align: top; }
th { background: var(--surface); }
td.montant, th.montant { text-align: right; white-space: nowrap; }
.etat { display: inline-block; padding: .1rem .5rem; border: 1px solid currentColor;
  border-radius: .25rem; font-size: .875rem; }
.etat-a_traiter { color: var(--alerte); }
.etat-fraude_confirmee { color: var(--alerte); background: #fbeaea; }
.etat-ecartee { color: var(--ok); background: #eaf5ee; }
.facteurs li { margin-bottom: .35rem; }
.hausse::before { content: "▲ "; }
.baisse::before { content: "▼ "; }
form.decision { margin-top: 1.5rem; border: 1px solid var(--bordure); padding: 1rem;
  background: var(--surface); }
label { display: block; font-weight: 600; margin-top: .75rem; }
input[type=text], textarea, select { width: 100%; max-width: 32rem; padding: .5rem;
  border: 1px solid var(--bordure); font: inherit; }
button { margin-top: 1rem; padding: .6rem 1.2rem; font: inherit; font-weight: 600;
  color: #fff; background: var(--accent); border: 0; cursor: pointer; }
button:hover { background: #083b78; }
.message { border-left: 4px solid var(--ok); background: #eaf5ee; padding: .75rem 1rem;
  margin-top: 1rem; }
.discret { color: var(--discret); font-size: .9rem; }
footer { border-top: 1px solid var(--bordure); color: var(--discret); font-size: .9rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""


def page(title: str, body: str, message: str = "") -> str:
    """Enveloppe commune : langue, titre unique, lien d'évitement, régions sémantiques."""
    annonce = (
        f'<p class="message" role="status" aria-live="polite">{escape(message)}</p>'
        if message
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} — Ondine Bank</title>
<style>{STYLE}</style>
</head>
<body>
<a class="evitement" href="#contenu">Aller au contenu</a>
<header>
  <p class="discret">Ondine Bank — détection de fraude</p>
  <h1>{escape(title)}</h1>
  <nav aria-label="Navigation principale">
    <a href="/alertes">File d'alertes</a>
  </nav>
</header>
<main id="contenu">
{annonce}
{body}
</main>
<footer>
  <p>Les décisions du modèle sont consultatives : un analyste tranche et sa décision est
  tracée. Toute personne concernée peut demander le réexamen d'un blocage.</p>
</footer>
</body>
</html>"""


def _moment(value: datetime | None) -> str:
    return value.strftime("%d/%m/%Y à %H:%M") if value else "—"


def _euros(value: float) -> str:
    """Montant en euros, séparateur de milliers insécable (usage typographique français)."""
    return f"{value:,.2f}".replace(",", "\u00a0").replace(".", ",") + "\u00a0€"


def queue_page(alerts: list[Alert], counts: dict[str, int], status: str) -> str:
    """File de travail : un tableau de données, trié du plus risqué au moins risqué."""
    onglets = " ".join(
        f'<li><a href="/alertes?etat={code}"'
        + (' aria-current="page"' if code == status else "")
        + f">{escape(libelle)} ({counts.get(code, 0)})</a></li>"
        for code, libelle in STATUSES.items()
    )
    if not alerts:
        corps = f"<p>Aucune alerte dans l'état « {escape(STATUSES.get(status, status))} ».</p>"
    else:
        lignes = "\n".join(
            f"""<tr>
  <td><a href="/alertes/{escape(a.transaction_id)}">{escape(a.transaction_id[:12])}…</a></td>
  <td>{_moment(a.event_time)}</td>
  <td class="montant">{_euros(a.amount)}</td>
  <td>{escape(a.channel or "—")}</td>
  <td class="montant">{100 * a.probability:.1f}&nbsp;%</td>
  <td>{escape(a.explanation)}</td>
</tr>"""
            for a in alerts
        )
        corps = f"""<table>
  <caption>Alertes « {escape(STATUSES.get(status, status))} », de la plus risquée à la moins
  risquée ({len(alerts)} affichées)</caption>
  <thead><tr>
    <th scope="col">Transaction</th><th scope="col">Date de l'opération</th>
    <th scope="col" class="montant">Montant</th><th scope="col">Canal</th>
    <th scope="col" class="montant">Probabilité de fraude</th>
    <th scope="col">Principaux facteurs</th>
  </tr></thead>
  <tbody>{lignes}</tbody>
</table>"""
    return page(
        "File d'alertes",
        f"""<nav aria-label="Filtrer par état"><ul>{onglets}</ul></nav>
{corps}""",
    )


def alert_page(alert: Alert, message: str = "") -> str:
    """Détail d'une alerte : ce que le modèle a vu, et la décision de l'analyste."""
    facteurs = "\n".join(
        f'<li class="{"hausse" if f.get("effet", "").startswith("augmente") else "baisse"}">'
        f"<strong>{escape(str(f.get('libellé', f.get('variable', ''))))}</strong> : "
        f"{escape(str(f.get('valeur', '—')))} — {escape(str(f.get('effet', '')))}</li>"
        for f in alert.factors
    )
    decision = (
        f"""<h2>Décision</h2>
<p>Traitée par {escape(alert.reviewed_by or "—")} le {_moment(alert.reviewed_at)} :
<span class="etat etat-{escape(alert.status)}">{escape(alert.status_label)}</span></p>
<p>{escape(alert.review_note or "Sans commentaire.")}</p>"""
        if alert.status != "a_traiter"
        else f"""<form class="decision" method="post"
      action="/alertes/{escape(alert.transaction_id)}">
  <h2>Votre décision</h2>
  <label for="analyste">Votre identifiant d'analyste</label>
  <input type="text" id="analyste" name="analyste" required autocomplete="username">
  <label for="etat">Suite donnée</label>
  <select id="etat" name="etat">
    <option value="fraude_confirmee">Fraude confirmée</option>
    <option value="ecartee">Écartée — opération légitime</option>
  </select>
  <label for="note">Commentaire (facultatif, visible du client en cas de réclamation)</label>
  <textarea id="note" name="note" rows="3"></textarea>
  <button type="submit">Enregistrer la décision</button>
</form>"""
    )
    return page(
        f"Alerte {alert.transaction_id[:12]}…",
        f"""<h2>Opération</h2>
<dl>
  <dt>Identifiant</dt><dd>{escape(alert.transaction_id)}</dd>
  <dt>Date</dt><dd>{_moment(alert.event_time)}</dd>
  <dt>Montant</dt><dd>{_euros(alert.amount)}</dd>
  <dt>Canal</dt><dd>{escape(alert.channel or "—")}</dd>
  <dt>Pays du marchand</dt><dd>{escape(alert.merchant_country or "—")}</dd>
  <dt>État</dt>
  <dd><span class="etat etat-{escape(alert.status)}">{escape(alert.status_label)}</span></dd>
</dl>

<h2>Pourquoi cette opération a été signalée</h2>
<p>{escape(alert.explanation)}</p>
<ul class="facteurs">{facteurs}</ul>
<p class="discret">Probabilité estimée {100 * alert.probability:.2f} %, pour un seuil
d'alerte à {100 * alert.threshold:.2f} % — modèle version {escape(alert.model_version)}.
Les flèches ▲ et ▼ indiquent le sens de chaque facteur, en plus de leur libellé.</p>

{decision}""",
        message,
    )
