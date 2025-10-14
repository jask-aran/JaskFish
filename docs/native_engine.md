# Native CPVS Engine

This document explains how to build and exercise the experimental C port of the
principal variation search engine. The native binary focuses on drop-in UCI
compatibility with `main.py` and the existing benchmarking harness so that
performance comparisons against the Python implementation can be made without
changing the surrounding tooling.

## Building on a native machine

The repository ships with a minimal `Makefile` that targets `gcc`, but any C11
compiler should work. When you have access to a machine with a toolchain
available, run:

```bash
cd native
make
```

This produces an executable named `cpvsengine` next to the source. On Windows
the output will be `cpvsengine.exe`. Both files are ignored by git so you can
rebuild locally without polluting the working tree.

If you prefer an out-of-tree build, point `make` (or your own CMake/Ninja setup)
at `native/cpvsengine.c` and emit the executable into `native/build/`. The shim
described below looks for `native/cpvsengine`, `native/cpvsengine.exe`, and
`native/build/cpvsengine` in that order by default.

## Running via UCI

The binary speaks the same UCI dialect as `pvsengine.py`. You can test it
interactively:

```bash
./native/cpvsengine
uci
isready
position startpos
go movetime 200
```

The engine prints iterative deepening summaries followed by a `bestmove` line
and honours `stop`, `ucinewgame`, and `quit`. Time controls are interpreted in
milliseconds (`movetime`, `wtime`, `btime`, `winc`, `binc`), matching the
Python engine.

## Using the Python shim

`main.py` always launches engines as Python scripts. To slot the native binary
into either the GUI or the headless harness, point the engine entry to
`native/cpvsengine_shim.py`. The shim spawns the compiled executable and
forwards UCI traffic between the Python process and the native binary.

By default the shim expects to find `native/cpvsengine` (or `cpvsengine.exe` on
Windows). You can override the location by setting
`CPVSENGINE_NATIVE_PATH=/absolute/path/to/cpvsengine` or by invoking the shim
with `--engine-path /custom/build/cpvsengine`. This allows you to keep multiple
builds around (for instance, profile vs. release) without editing the launcher
code.

To verify the setup end-to-end:

```bash
python native/cpvsengine_shim.py
uci
isready
quit
```

When a build is present the shim will echo the binary's responses to stdout, so
`uci` should produce the engine's identification banner and supported option
list.

## Benchmarking against the Python engine

`benchmark_pvs.py` now includes a `compare` sub-command that drives both engines
through the same set of positions and search budgets. For example, to compare a
200 ms search window on the default curated scenarios:

```bash
python benchmark_pvs.py compare --movetime 200 --max-wait 5 \
    --native-cmd ./native/cpvsengine
```

The command prints per-position metrics (depth, elapsed time, nodes searched,
and principal variation) for each engine, along with aggregate averages and
Python/native ratios.

## Known gaps

The C engine mirrors the Python heuristics at a high level (piece-square tables,
simple quiescence search, history ordering, and transposition table with
principal variation search), but it remains a single-threaded baseline without
aspiration windows or sophisticated pruning. A few midgame positions currently
resolve immediately because the legality checks are stricter than the Python
engine’s move generator, so the reported node counts can drop to zero when the
search recognises a forced mate at the root. Despite those limitations, the
binary provides a useful reference point for evaluating the raw overhead of the
Python implementation versus a native build.
