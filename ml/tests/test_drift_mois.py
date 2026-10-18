"""Le decoupage glissant se cale sur un mois complet, jamais sur la queue du flux."""

import pytest

from ondine_ml.drift import _dernier_mois_complet


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
