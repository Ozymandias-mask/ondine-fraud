"""Le decoupage glissant se cale sur un mois complet, jamais sur la queue du flux."""

import pytest

from ondine_ml.drift import (
    _dernier_mois_complet,
    _mois_complets,
    _mois_suivant,
    _series_continues,
)


def test_ignore_le_mois_a_peine_entame():
    """Le flux temps reel laisse quelques milliers de lignes dans le mois courant."""
    volumes = [("2025-10", 2_400_000), ("2025-11", 2_500_000), ("2025-12", 2_450_000),
               ("2026-09", 1_200)]
    assert _dernier_mois_complet(volumes) == "2025-12"


def test_garde_un_mois_simplement_plus_calme():
    """Un mois a -30 % reste un mois complet : on n'ecarte que ce qui est hors d'echelle."""
    volumes = [("2025-10", 2_400_000), ("2025-11", 2_500_000), ("2025-12", 1_700_000)]
    assert _dernier_mois_complet(volumes) == "2025-12"


def test_refuse_un_lac_vide():
    with pytest.raises(FileNotFoundError):
        _dernier_mois_complet([])



def _fin_de_serie(volumes, minimum=5):
    """Ce que `dernier_mois_entrainable` calcule, sans toucher au lac."""
    series = _series_continues(_mois_complets(volumes))
    assez = [s for s in series if len(s) >= minimum]
    if not assez:
        raise FileNotFoundError("Aucune série de mois consécutifs assez longue.")
    return assez[-1][-1]


def test_le_mois_de_la_derive_n_est_pas_le_mois_d_entrainement():
    """Le flux temps réel écrit un mois en avance sur l'historique chargé par lots.

    Ce mois-là est le bon pour mesurer la dérive — c'est là que sont les données fraîches —
    mais pas pour finir un découpage : les mois de validation et de test qui le précèdent
    n'existent pas encore.
    """
    volumes = [(f"2025-{m:02d}", 4_000_000) for m in range(1, 13)]
    volumes.append(("2026-09", 21_000_000))
    assert _dernier_mois_complet(volumes) == "2026-09"
    assert _fin_de_serie(volumes) == "2025-12"


def test_serie_continue_unique():
    volumes = [(f"2025-{m:02d}", 4_000_000) for m in range(1, 13)]
    assert _fin_de_serie(volumes) == "2025-12"


def test_refuse_une_serie_trop_courte():
    volumes = [("2025-01", 4_000_000), ("2025-02", 4_000_000), ("2026-09", 4_000_000)]
    with pytest.raises(FileNotFoundError):
        _fin_de_serie(volumes)


def test_passage_d_annee():
    assert _mois_suivant("2025-12") == "2026-01"
    assert _mois_suivant("2025-09") == "2025-10"
