#!/usr/bin/env python3
"""Run PVSearch profiling across predefined scenarios and capture the results."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PVSENGINE = ROOT / "pvsengine.py"
OUTPUT_ROOT = ROOT / "profiles"

POSITIONS = [
    ("Opening - Start Position", ""),
    (
        "Midgame - Complex Center",
        "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10",
    ),
    (
        "Midgame - Pressure on King",
        "2r2rk1/1b2bppp/p2p1n2/1p1Pp3/1P2P3/P1N1BN2/1B3PPP/2RR2K1 w - - 0 21",
    ),
]


def run_profile(fen: str) -> str:
    cmd = [sys.executable, str(PVSENGINE), "--profile"]
    if fen:
        cmd.extend(["--fen", fen])
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def main() -> None:
    output_dir = OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pvs_profile.txt"

    sections = []
    for label, fen in POSITIONS:
        output = run_profile(fen)
        sections.append(
            "\n".join(
                [
                    "=" * 80,
                    f"Scenario: {label}",
                    f"FEN: {fen or '(start position)'}",
                    "=" * 80,
                    output.rstrip(),
                    "",
                ]
            )
        )

    combined = "\n".join(sections).rstrip() + "\n"
    output_file.write_text(combined)
    print(f"Profile output written to {output_file}")


if __name__ == "__main__":
    main()
