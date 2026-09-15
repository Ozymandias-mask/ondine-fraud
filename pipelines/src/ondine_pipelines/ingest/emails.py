"""Lecture des e-mails bruts du support (format mbox compressé)."""

from __future__ import annotations

import gzip
import mailbox
import shutil
import tempfile
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from pathlib import Path

import pyarrow as pa

EMAIL_SCHEMA = pa.schema(
    [
        ("message_id", pa.string()),
        ("received_at", pa.timestamp("ms", tz="UTC")),
        ("sender_name", pa.string()),
        ("sender_address", pa.string()),
        ("subject", pa.string()),
        ("body", pa.string()),
        ("parse_error", pa.string()),
    ]
)


def _decode(value: str | None) -> str:
    if not value:
        return ""
    return str(make_header(decode_header(value)))


def read_mbox_gz(path: Path) -> pa.Table:
    """Extrait les champs utiles de chaque e-mail. Un e-mail illisible n'arrête pas la lecture.

    Les messages mal formés sont conservés avec leur erreur (`parse_error`) pour être
    envoyés en quarantaine au lieu d'être perdus silencieusement.
    """
    rows: dict[str, list] = {name: [] for name in EMAIL_SCHEMA.names}
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "support.mbox"
        with gzip.open(path, "rb") as src, raw.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        box = mailbox.mbox(raw, create=False)
        try:
            for message in box:
                error = None
                try:
                    name, address = parseaddr(_decode(message["From"]))
                    received = parsedate_to_datetime(message["Date"])
                    payload = message.get_payload(decode=True) or b""
                    charset = message.get_content_charset() or "utf-8"
                    body = payload.decode(charset, errors="replace")
                except (TypeError, ValueError, LookupError) as exc:
                    name, address, received, body = "", "", None, ""
                    error = f"{type(exc).__name__}: {exc}"
                rows["message_id"].append((message["Message-ID"] or "").strip())
                rows["received_at"].append(received)
                rows["sender_name"].append(name)
                rows["sender_address"].append(address.lower())
                rows["subject"].append(_decode(message["Subject"]))
                rows["body"].append(body)
                rows["parse_error"].append(error)
        finally:
            box.close()
    return pa.table(rows, schema=EMAIL_SCHEMA)
