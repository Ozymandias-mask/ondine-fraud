import re

import numpy as np
from ondine_sim.identifiers import (
    external_ibans,
    french_ibans,
    hashed_hex_ids,
    iban_is_valid,
    ipv4_from_keys,
    random_hex_ids,
    sequential_ids,
    splitmix64,
)


def test_generated_ibans_have_valid_check_digits():
    keys = np.arange(5_000, dtype=np.int64)
    for iban in np.concatenate([french_ibans(keys, 99001), external_ibans(keys)]):
        assert len(iban) == 27 and iban.startswith("FR")
        assert iban_is_valid(iban), iban


def test_iban_validation_detects_typo():
    iban = french_ibans(np.array([42]), 99001)[0]
    corrupted = iban[:-1] + ("0" if iban[-1] != "0" else "1")
    assert not iban_is_valid(corrupted)


def test_hash_is_deterministic_and_spread():
    keys = np.arange(10_000)
    assert np.array_equal(splitmix64(keys), splitmix64(keys))
    assert len(np.unique(splitmix64(keys))) == len(keys)


def test_random_ids_are_unique_hex():
    ids = random_hex_ids(np.random.default_rng(0), 50_000)
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[0-9a-f]{32}", x) for x in ids[:100])


def test_readable_ids_and_empty_input():
    assert list(sequential_ids(np.array([0, 123]), "C", 8)) == ["C00000000", "C00000123"]
    assert len(sequential_ids(np.array([], dtype=np.int64), "C", 8)) == 0
    assert hashed_hex_ids(np.array([7]), "dev_")[0].startswith("dev_")


def test_ipv4_format():
    ips = ipv4_from_keys(np.arange(1_000), np.full(1_000, 82))
    for ip in ips:
        parts = [int(p) for p in ip.split(".")]
        assert parts[0] == 82 and all(0 <= p <= 255 for p in parts)
