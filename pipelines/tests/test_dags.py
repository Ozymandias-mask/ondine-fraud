"""Chargement des DAG par Airflow (exécuté si Airflow est installé, sinon ignoré)."""

from pathlib import Path

import pytest

DAGS = Path(__file__).resolve().parents[1] / "dags"


def test_dags_load_without_errors():
    pytest.importorskip("airflow")
    from airflow.dag_processing.dagbag import DagBag

    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    assert set(bag.dags) == {
        "ondine_ingestion",
        "ondine_lake_refresh",
        "ondine_governance",
        "ondine_retention",
        "ondine_model_refresh",
    }
    for dag in bag.dags.values():
        assert dag.max_active_runs == 1 and not dag.catchup
        assert all(t.retries >= 3 and t.on_failure_callback for t in dag.tasks)
