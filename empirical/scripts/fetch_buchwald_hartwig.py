#!/usr/bin/env python3
"""Restore the measured-reaction CSV from its checksummed upstream workbook.

Only the requested output file is written. The upstream dataset is not bundled
with this archive. Requires openpyxl; no model training or yield prediction runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
URL = ("https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/"
       "data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx")
WORKBOOK_SHA256 = "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e"
CSV_SHA256 = "a7c415d9ff58d940cc8214c46fa4af12fd7a588f0b4314d0798e70f7a7eb2083"
SHEET = "Plates1-3"
COLUMNS = ("Ligand", "Additive", "Base", "Aryl halide", "Output")
REACTIONS = 3955


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def export_csv(workbook_bytes: bytes) -> bytes:
    if sha256(workbook_bytes) != WORKBOOK_SHA256:
        raise ValueError("Upstream workbook SHA-256 differs from the registered source.")
    try:
        import openpyxl
    except ImportError as exc:
        raise SystemExit("Install input dependencies: pip install -r empirical/requirements-inputs.txt") from exc

    book = openpyxl.load_workbook(io.BytesIO(workbook_bytes), read_only=True, data_only=True)
    try:
        rows = list(book[SHEET].values)
    finally:
        book.close()
    if tuple(rows[0]) != COLUMNS or len(rows) != REACTIONS + 1:
        raise ValueError("Unexpected worksheet columns or row count.")
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    csv_bytes = output.getvalue().encode("utf-8")
    if sha256(csv_bytes) != CSV_SHA256:
        raise ValueError("Exported CSV SHA-256 differs from the registered measured table.")
    return csv_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "buchwald_hartwig.csv")
    parser.add_argument("--workbook", type=Path,
                        help="Use an already downloaded, checksummed workbook instead of networking")
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    if destination.exists():
        if sha256(destination.read_bytes()) != CSV_SHA256:
            raise FileExistsError(f"Refusing to overwrite a different existing file: {destination}")
        status = "verified-existing"
    else:
        if args.workbook:
            workbook_bytes = args.workbook.expanduser().read_bytes()
        else:
            request = urllib.request.Request(URL, headers={"User-Agent": "ReactorGen-input-restoration/3"})
            with urllib.request.urlopen(request, timeout=60) as response:
                workbook_bytes = response.read()
        csv_bytes = export_csv(workbook_bytes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".buchwald_hartwig-",
                                             suffix=".csv.tmp", delete=False) as handle:
                temporary_path = Path(handle.name)
                handle.write(csv_bytes)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        status = "restored"
    print(json.dumps({"status": status, "output": str(destination), "rows": REACTIONS,
                      "csv_sha256": CSV_SHA256, "upstream_workbook_sha256": WORKBOOK_SHA256}, indent=2))


if __name__ == "__main__":
    main()
