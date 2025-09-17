#!/usr/bin/env python3
"""Convert JaskFish search logs into NumPy-friendly datasets.

The script is intentionally lightweight so that it can run inside
self-play pipelines or on archived logs without additional tooling.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets from engine search logs")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/search_logs.jsonl"),
        help="Path to the search log file (JSON lines or CSV)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Structured format used in the log file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/search_dataset.npz"),
        help="Destination for the generated NumPy archive",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of records to export",
    )
    return parser.parse_args()


def _load_json_records(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_csv_records(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _maybe_parse_pv(value) -> Sequence[str]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(move) for move in parsed]
    except json.JSONDecodeError:
        pass
    return [move for move in text.split() if move]


def _normalise_records(records: Iterable[dict]) -> List[dict]:
    normalised: List[dict] = []
    for raw in records:
        pv = _maybe_parse_pv(raw.get("principal_variation") or raw.get("pv"))
        score_value = raw.get("score", 0.0)
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            score = 0.0

        normalised.append(
            {
                "board_fen": raw.get("board_fen") or raw.get("fen") or "",
                "best_move": raw.get("best_move") or raw.get("move") or "",
                "principal_variation": pv,
                "score": score,
            }
        )
    return normalised


def _limit_records(records: List[dict], limit: int | None) -> List[dict]:
    if limit is None or limit >= len(records):
        return records
    return records[:limit]


def _to_numpy_columns(records: Sequence[dict]) -> dict:
    fens = np.array([entry["board_fen"] for entry in records], dtype=object)
    moves = np.array([entry["best_move"] for entry in records], dtype=object)
    pvs = np.array([entry["principal_variation"] for entry in records], dtype=object)
    scores = np.array([entry["score"] for entry in records], dtype=float)
    return {"fen": fens, "best_move": moves, "principal_variation": pvs, "score": scores}


def main() -> None:
    args = parse_arguments()
    if not args.log.exists():
        raise FileNotFoundError(f"Log file {args.log} does not exist")

    if args.format == "json":
        raw_records = _load_json_records(args.log)
    else:
        raw_records = _load_csv_records(args.log)

    normalised_records = _normalise_records(raw_records)
    limited_records = _limit_records(normalised_records, args.limit)

    if not limited_records:
        print("No records found to export.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = _to_numpy_columns(limited_records)
    np.savez(args.output, **dataset)
    print(f"Wrote {len(limited_records)} samples to {args.output}")


if __name__ == "__main__":
    main()
