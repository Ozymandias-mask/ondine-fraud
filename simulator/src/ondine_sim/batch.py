"""Génération de l'historique (mode batch) : 12 mois de données brutes.

Découpage : chaque mois est découpé en « parts » (une tranche de clients chacune).
Une part est générée entièrement par un seul processus et possède sa propre graine
dérivée de (graine, mois, part) : le résultat est identique quel que soit le nombre
de processus, et une part en échec peut être relancée seule (`--resume`).
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from . import columns as col
from .behaviour import background_events, legit_transactions, linked_events, month_timestamps
from .config import SimConfig
from .entities import cached_entities, cards_frame, clients_frame, merchants_frame, month_bounds
from .formatting import events_jsonl, transactions_frame
from .fraud import inject_fraud
from .reference import SEASONALITY
from .support import fraud_reports, support_emails
from .writers import FileInfo, GzipWriter, write_csv_gz, write_manifest, write_mbox_gz

logger = logging.getLogger(__name__)

FORMAT_CHUNK = 200_000  # lignes converties en texte à la fois (borne la mémoire)


def _to_ms(ts: object) -> int:
    return int(ts.value // 1_000_000)  # type: ignore[attr-defined]


def parts_per_month(cfg: SimConfig) -> int:
    """Nombre de parts par mois pour viser ~`rows_per_part` transactions par fichier."""
    expected = cfg.n_clients * cfg.avg_tx_per_client_month * SEASONALITY.max()
    return max(1, math.ceil(expected / cfg.rows_per_part))


def app_v5_share(cfg: SimConfig, month: int) -> float:
    """Part des événements émis par l'application v5 (déploiement progressif)."""
    if month < cfg.schema_change_month:
        return 0.0
    return min(0.95, 0.5 + 0.1 * (month - cfg.schema_change_month))


def generate_part(cfg: SimConfig, month: int, part: int, n_parts: int) -> dict[str, object]:
    """Génère et écrit toutes les données d'une part de mois. Retourne des statistiques."""
    started = time.perf_counter()
    ent = cached_entities(cfg)
    rng = np.random.default_rng([cfg.seed, month, part])
    start, end = month_bounds(cfg, month)
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    lo = part * ent.n_clients // n_parts
    hi = (part + 1) * ent.n_clients // n_parts
    schema_v2 = month >= cfg.schema_change_month
    drift = month >= cfg.drift_month

    # 1. Transactions légitimes : nombre par client ~ Poisson(activité × saisonnalité).
    clients = np.arange(lo, hi)
    season = SEASONALITY[start.month - 1]
    counts = rng.poisson(ent.client_activity[lo:hi] * cfg.avg_tx_per_client_month * season)
    tx_client = np.repeat(clients, counts)
    legit = legit_transactions(
        ent, rng, tx_client, month_timestamps(rng, len(tx_client), start_ms, end_ms), schema_v2
    )

    # 2. Fraudes : une partie des clients de la tranche est victime ce mois-ci.
    n_victims = rng.binomial(hi - lo, cfg.fraud_victim_rate)
    victims = lo + rng.choice(hi - lo, size=n_victims, replace=False)
    fraud = inject_fraud(
        ent, rng, victims, start_ms, end_ms, drift, incident_base=(month * 1_000 + part) * 100_000
    )
    tx = col.sort_by_ts(col.concat(col.TX_COLUMNS, [legit, fraud.tx]))

    # 3. Événements applicatifs : liés aux opérations, liés aux fraudes, et usage courant.
    linked = linked_events(ent, rng, legit)
    n_background = rng.poisson(
        max(0.0, len(tx["ts"]) * cfg.events_per_tx - len(linked["ts"]) - len(fraud.events["ts"]))
    )
    weights = ent.client_activity[lo:hi] / ent.client_activity[lo:hi].sum()
    bg_clients = rng.choice(clients, size=n_background, p=weights)
    background = background_events(
        ent, rng, bg_clients, month_timestamps(rng, n_background, start_ms, end_ms)
    )
    events = col.sort_by_ts(col.concat(col.EVENT_COLUMNS, [linked, fraud.events, background]))

    # 4. Mise en forme et écriture par blocs (mémoire bornée quel que soit le volume).
    root = cfg.output_dir
    period = start.strftime("%Y-%m")
    partition = f"year={start.year}/month={start.month:02d}"
    suffix = f"{period}_part-{part:03d}"
    truth_parts, candidate_parts = [], []
    with GzipWriter(
        root / "transactions" / partition / f"transactions_{suffix}.csv.gz", root
    ) as tx_out:
        for chunk_start in range(0, len(tx["ts"]), FORMAT_CHUNK):
            chunk = col.take(tx, slice(chunk_start, chunk_start + FORMAT_CHUNK))
            frame, truth_chunk = transactions_frame(ent, rng, chunk, schema_v2)
            tx_out.write_frame(frame)
            truth_parts.append(truth_chunk)
            approved = frame.loc[
                frame["auth_status"] == "approved", ["transaction_id", "client_id", "event_time"]
            ]
            seed = int(rng.integers(2**31))
            candidate_parts.append(approved.sample(frac=0.01, random_state=seed))
    truth = pd.concat(truth_parts, ignore_index=True)

    with GzipWriter(
        root / "app_events" / partition / f"app_events_{suffix}.jsonl.gz", root
    ) as ev_out:
        for chunk_start in range(0, len(events["ts"]), FORMAT_CHUNK):
            chunk = col.take(events, slice(chunk_start, chunk_start + FORMAT_CHUNK))
            lines, _ = events_jsonl(ent, rng, chunk, app_v5_share(cfg, month))
            ev_out.write_lines(lines)

    reports = fraud_reports(rng, cfg, truth, pd.concat(candidate_parts, ignore_index=True))
    emails, tickets = support_emails(rng, cfg, reports, truth, clients, start, end)
    files: list[FileInfo] = [
        tx_out.info,
        ev_out.info,  # type: ignore[list-item]
        write_csv_gz(
            reports, root / "fraud_reports" / partition / f"fraud_reports_{suffix}.csv.gz", root
        ),
        write_mbox_gz(
            emails, root / "support_emails" / partition / f"support_emails_{suffix}.mbox.gz", root
        ),
        write_csv_gz(
            tickets, root / "support_tickets" / partition / f"support_tickets_{suffix}.csv.gz", root
        ),
    ]
    # Vérité terrain : hors zone d'arrivée, réservée à l'évaluation du simulateur.
    truth_root = root.parent / "ground_truth"
    write_csv_gz(truth, truth_root / f"fraud_truth_{suffix}.csv.gz", truth_root)

    duration = time.perf_counter() - started
    write_manifest(
        _manifest_path(cfg, month, part),
        files,
        {
            "dataset_period": period,
            "part": part,
            "parts": n_parts,
            "generator": "ondine_sim",
            "generator_version": __version__,
            "seed": cfg.seed,
            "transaction_schema": "v2" if schema_v2 else "v1",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(duration, 2),
        },
    )
    return {
        "month": period,
        "part": part,
        "transactions": tx_out.rows,
        "fraud_transactions": len(truth),
        "events": ev_out.rows,
        "reports": len(reports),
        "emails": len(emails),
        "bytes": sum(f.bytes for f in files),
        "duration_s": duration,
    }


def _manifest_path(cfg: SimConfig, month: int, part: int) -> Path:
    start, _ = month_bounds(cfg, month)
    return cfg.output_dir / "_manifests" / f"period={start:%Y-%m}" / f"part-{part:03d}.json"


def write_referentials(cfg: SimConfig) -> list[FileInfo]:
    """Exporte les référentiels (photo à la veille de la période simulée)."""
    ent = cached_entities(cfg)
    root = cfg.output_dir
    snapshot = (month_bounds(cfg, 0)[0] - np.timedelta64(1, "D")).strftime("%Y-%m-%d")
    files = [
        write_csv_gz(
            clients_frame(ent),
            root / "referentials" / "clients" / f"clients_{snapshot}.csv.gz",
            root,
        ),
        write_csv_gz(
            cards_frame(ent), root / "referentials" / "cards" / f"cards_{snapshot}.csv.gz", root
        ),
        write_csv_gz(
            merchants_frame(ent),
            root / "referentials" / "merchants" / f"merchants_{snapshot}.csv.gz",
            root,
        ),
    ]
    write_manifest(
        root / "_manifests" / "referentials" / f"snapshot={snapshot}.json",
        files,
        {
            "generator": "ondine_sim",
            "generator_version": __version__,
            "seed": cfg.seed,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return files


def run_batch(cfg: SimConfig, resume: bool = False) -> dict[str, object]:
    """Génère l'historique complet. Lève une erreur si au moins une part a échoué."""
    started = time.perf_counter()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    n_parts = parts_per_month(cfg)
    tasks = [(m, p) for m in range(cfg.months) for p in range(n_parts)]
    if resume:
        tasks = [(m, p) for m, p in tasks if not _manifest_path(cfg, m, p).exists()]
    logger.info(
        "Génération : %d clients, %d mois, %d parts/mois, %d parts à produire, %d processus",
        cfg.n_clients,
        cfg.months,
        n_parts,
        len(tasks),
        cfg.workers,
    )

    referential_manifest = cfg.output_dir / "_manifests" / "referentials"
    if not (resume and referential_manifest.exists()):
        write_referentials(cfg)
        logger.info("Référentiels écrits (clients, cartes, marchands)")

    totals = {
        "transactions": 0,
        "fraud_transactions": 0,
        "events": 0,
        "reports": 0,
        "emails": 0,
        "bytes": 0,
    }
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(generate_part, cfg, m, p, n_parts): (m, p) for m, p in tasks}
        for future in as_completed(futures):
            m, p = futures[future]
            try:
                stats = future.result()
            except Exception:  # une part en échec n'arrête pas les autres
                logger.exception("Échec de la part mois=%d part=%d", m, p)
                failures.append(f"{m}:{p}")
                continue
            for key in totals:
                totals[key] += int(stats[key])  # type: ignore[call-overload]
            logger.info(
                "Part %s/%03d : %s transactions, %s événements en %.1f s",
                stats["month"],
                p,
                f"{stats['transactions']:,}",
                f"{stats['events']:,}",
                stats["duration_s"],
            )

    duration = time.perf_counter() - started
    summary = {
        **totals,
        "duration_s": round(duration, 1),
        "transactions_per_s": round(totals["transactions"] / max(duration, 1e-9)),
        "failed_parts": failures,
    }
    write_manifest(
        cfg.output_dir / "_manifests" / "runs" / f"run_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
        [],
        summary,
    )
    logger.info("Terminé : %s", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} part(s) en échec : relancer avec --resume")
    return summary
