# Feasibility of a native PVS engine

This note captures the constraints that block a drop-in C/C++ rewrite of
`pvsengine.py` within the current repository. It explains why a compatible
replacement is not available in this change set and outlines the engineering
work that would be required if a native port were attempted in the future.

## Why a straight port is not drop-in

* **Process contract** – The GUI and the headless self-play harness both spawn
the engine with `[sys.executable, "pvsengine.py"]`. A native binary would need
different launch arguments, so the orchestration layer would have to be taught
to detect and execute non-Python engines. That is outside the scope of a
self-contained “drop-in” swap.
* **Runtime dependencies** – The existing engine leans on `python-chess` for
bitboards, move generation, legality checks, repetition detection, and PGN
formatting. A native version would either need to embed a Python interpreter or
re-implement these systems. Neither option is feasible in a short iteration.
* **Protocol surface area** – Beyond the basic UCI commands, the Python engine
exposes rich telemetry (search traces, aspiration logs, budgeting decisions).
Replicating those side channels requires a wholesale port of several thousand
lines of tightly coupled Python.

## Work required for a full port

A credible C/C++ rewrite would need to supply:

1. A complete chess rules engine (bitboard representation, incremental make-
and-unmake, move ordering, quiescence search, SEE, repetition history).
2. Feature parity for configuration (`MetaRegistry`, `SearchLimits`), including
JSON profile ingestion and tunable aspiration windows.
3. Matching UCI façade behaviour (multi-threaded `go` handling, deterministic
`stop` semantics, verbose info strings) so that the GUI and tests remain valid.
4. A build system, packaging, and cross-platform binaries that integrate with
the Python entry points used throughout the repo.

Until those prerequisites are met, the Python tooling cannot interact with a
native engine without additional adapter code.

## Benchmarking note

Because no native engine has been produced, there is nothing to benchmark. The
existing `benchmark_pvs.py` script continues to exercise the Python engine; it
cannot be pointed at an unimplemented binary.

---

In short, the project does not yet provide a C/C++ replacement for
`pvsengine.py`. Enabling such a swap would demand significant architecture work
well beyond the scope of this task.
