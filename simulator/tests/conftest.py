"""Fixtures partagées : une configuration minuscule pour des tests rapides."""

from __future__ import annotations

from dataclasses import replace

import pytest
from ondine_sim.config import get_config
from ondine_sim.entities import build_entities


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim") / "landing"
    return get_config("small", output_dir=out, workers=1)


@pytest.fixture(scope="session")
def entities(small_cfg):
    return build_entities(small_cfg)


@pytest.fixture()
def cfg_in(tmp_path, small_cfg):
    """Configuration « small » écrivant dans un dossier temporaire propre au test."""

    def _make(**overrides):
        return replace(small_cfg, output_dir=tmp_path / "landing", **overrides)

    return _make
