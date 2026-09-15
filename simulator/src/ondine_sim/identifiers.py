"""Génération vectorisée d'identifiants : IDs, IBAN, jetons de carte, adresses IP.

Toutes les fonctions travaillent sur des tableaux numpy pour traiter des millions
de lignes sans boucle Python.
"""

from __future__ import annotations

import numpy as np

_HEX = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)
_U64 = np.uint64


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Hache des entiers en uint64 pseudo-aléatoires, de manière déterministe.

    Permet de dériver un attribut stable d'une entité (ex. l'IBAN du 2e bénéficiaire
    habituel d'un client) sans le stocker en mémoire.
    """
    z = values.astype(_U64) + _U64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB)
    return z ^ (z >> _U64(31))


def random_hex_ids(
    rng: np.random.Generator, n: int, n_bytes: int = 16, prefix: str = ""
) -> np.ndarray:
    """Retourne `n` identifiants hexadécimaux aléatoires (32 caractères par défaut)."""
    raw = rng.integers(0, 256, size=(n, n_bytes), dtype=np.uint8)
    chars = np.empty((n, n_bytes * 2), dtype=np.uint8)
    chars[:, 0::2] = _HEX[raw >> 4]
    chars[:, 1::2] = _HEX[raw & 0x0F]
    ids = chars.view(f"S{n_bytes * 2}").ravel().astype(str)
    return np.char.add(prefix, ids) if prefix else ids


def hashed_hex_ids(keys: np.ndarray, prefix: str) -> np.ndarray:
    """Identifiant hexadécimal stable (16 caractères) dérivé d'une clé entière."""
    h = splitmix64(keys)
    as_bytes = h.astype(">u8").view(np.uint8).reshape(-1, 8)
    chars = np.empty((len(keys), 16), dtype=np.uint8)
    chars[:, 0::2] = _HEX[as_bytes >> 4]
    chars[:, 1::2] = _HEX[as_bytes & 0x0F]
    return np.char.add(prefix, chars.view("S16").ravel().astype(str))


def sequential_ids(indices: np.ndarray, prefix: str, width: int) -> np.ndarray:
    """Identifiants lisibles de type `C00001234`."""
    if len(indices) == 0:
        return np.array([], dtype=str)
    return np.char.add(prefix, np.char.zfill(indices.astype(str), width))


def french_ibans(keys: np.ndarray, bank_code: int | np.ndarray) -> np.ndarray:
    """Génère des IBAN français valides (clé RIB et clé IBAN correctes).

    `bank_code` fictif (un code ou un tableau de codes) : 99001 pour Ondine Bank,
    99002 à 99021 pour les banques externes. Le calcul modulo 97 est fait par
    morceaux pour rester en int64.
    """
    h = splitmix64(keys)
    bank = np.broadcast_to(np.asarray(bank_code, dtype=np.int64), (len(keys),)).copy()
    branch = (h % _U64(100_000)).astype(np.int64)
    account = ((h >> _U64(17)) % _U64(100_000_000_000)).astype(np.int64)

    # Clé RIB : 97 - ((89 × banque + 15 × guichet + 3 × compte) mod 97).
    rib_key = 97 - ((89 * bank + 15 * branch + 3 * (account % 97)) % 97)

    # Clé IBAN : BBAN + "FR00" converti en chiffres (F=15, R=27) => "152700".
    m = bank % 97
    m = (m * 100_000 + branch) % 97
    m = (m * (10**11 % 97) + account) % 97
    m = (m * 100 + rib_key) % 97
    m = (m * 1_000_000 + 152_700) % 97
    check = 98 - m

    def z(arr: np.ndarray, width: int) -> np.ndarray:
        return np.char.zfill(arr.astype(str), width)

    parts = [z(check, 2), z(bank, 5), z(branch, 5), z(account, 11), z(rib_key, 2)]
    out = np.full(len(keys), "FR", dtype="<U27")
    for part in parts:
        out = np.char.add(out, part)
    return out


def iban_is_valid(iban: str) -> bool:
    """Vérifie une clé IBAN (utilisé par les tests et le contrôle qualité)."""
    rearranged = iban[4:] + iban[:4]
    digits = "".join(str(int(c, 36)) for c in rearranged)
    return int(digits) % 97 == 1


def ipv4_from_keys(keys: np.ndarray, first_octets: np.ndarray) -> np.ndarray:
    """Adresse IPv4 dont le premier octet est imposé et le reste dérivé de la clé."""
    h = splitmix64(keys)
    b = ((h >> _U64(8)) % _U64(256)).tolist()
    c = ((h >> _U64(16)) % _U64(256)).tolist()
    d = ((h >> _U64(24)) % _U64(254) + _U64(1)).tolist()
    a = np.asarray(first_octets).tolist()
    # Formatage en Python pur : bien plus rapide que np.char pour ce motif.
    return np.array(
        [f"{w}.{x}.{y}.{z}" for w, x, y, z in zip(a, b, c, d, strict=True)], dtype=object
    )


def external_ibans(keys: np.ndarray) -> np.ndarray:
    """IBAN de comptes externes (bénéficiaires, comptes mules), banque tirée de la clé.

    Les comptes mules utilisent les mêmes banques fictives que les bénéficiaires
    légitimes : le code banque ne doit pas trahir la fraude.
    """
    bank_hash = splitmix64(keys.astype(np.uint64) + np.uint64(11))
    bank = 99_002 + (bank_hash % np.uint64(20)).astype(np.int64)
    return french_ibans(keys, bank)
