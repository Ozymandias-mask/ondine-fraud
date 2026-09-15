"""Écriture des fichiers bruts et des manifestes de livraison.

Chaque fichier est écrit sous un nom temporaire puis renommé : un pipeline qui
surveille la zone d'arrivée ne voit donc jamais de fichier à moitié écrit.
Chaque livraison est accompagnée d'un manifeste (lignes, taille, empreinte SHA-256)
qui sert au contrôle d'intégrité et à la traçabilité (lignage) des données.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

GZIP_LEVEL = 5  # bon compromis vitesse / taille pour des volumes de plusieurs Go


@dataclass
class FileInfo:
    """Description d'un fichier livré, reprise dans le manifeste."""

    path: str
    rows: int
    bytes: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _finalize(tmp: Path, final: Path, rows: int, root: Path) -> FileInfo:
    os.replace(tmp, final)  # renommage atomique sur un même système de fichiers
    return FileInfo(
        path=final.relative_to(root).as_posix(),
        rows=rows,
        bytes=final.stat().st_size,
        sha256=_sha256(final),
    )


def write_csv_gz(frame: pd.DataFrame, path: Path, root: Path) -> FileInfo:
    """Écrit un DataFrame en CSV compressé (séparateur virgule, UTF-8, en-tête)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression={"method": "gzip", "compresslevel": GZIP_LEVEL})
    return _finalize(tmp, path, len(frame), root)


class GzipWriter:
    """Écriture incrémentale dans un fichier compressé, par blocs.

    Permet de produire des fichiers de plusieurs centaines de Mo sans jamais
    charger toutes les lignes en mémoire. Utilisation :

        with GzipWriter(path, root) as out:
            out.write_frame(df)        # CSV (en-tête écrit une seule fois)
            out.write_lines(lines)     # JSON lines
        info = out.info
    """

    def __init__(self, path: Path, root: Path) -> None:
        """Prépare l'écriture de `path` (chemin relatif à `root` dans le manifeste)."""
        self.path, self.root = path, root
        self.tmp = path.with_name(path.name + ".tmp")
        self.rows = 0
        self._header_written = False
        self.info: FileInfo | None = None

    def __enter__(self) -> GzipWriter:
        """Ouvre le fichier temporaire."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = gzip.open(self.tmp, "wb", compresslevel=GZIP_LEVEL)
        return self

    def write_frame(self, frame: pd.DataFrame) -> None:
        """Ajoute un bloc de lignes CSV."""
        text = frame.to_csv(index=False, header=not self._header_written)
        self._handle.write(text.encode("utf-8"))
        self._header_written = True
        self.rows += len(frame)

    def write_lines(self, lines: list[bytes]) -> None:
        """Ajoute un bloc de lignes déjà sérialisées (JSON)."""
        if lines:
            self._handle.write(b"\n".join(lines) + b"\n")
            self.rows += len(lines)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Ferme le fichier ; ne le publie (renommage) qu'en l'absence d'erreur."""
        self._handle.close()
        if exc_type is None:
            self.info = _finalize(self.tmp, self.path, self.rows, self.root)
        else:
            self.tmp.unlink(missing_ok=True)


def write_mbox_gz(messages: list[bytes], path: Path, root: Path) -> FileInfo:
    """Écrit des e-mails bruts au format mbox compressé (standard des boîtes mail)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wb", compresslevel=GZIP_LEVEL) as handle:
        for message in messages:
            handle.write(message)
    return _finalize(tmp, path, len(messages), root)


def write_manifest(path: Path, files: list[FileInfo], metadata: dict[str, object]) -> None:
    """Écrit le manifeste JSON d'une livraison (en dernier : il signale qu'elle est complète)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    content = {**metadata, "files": [asdict(f) for f in files]}
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
