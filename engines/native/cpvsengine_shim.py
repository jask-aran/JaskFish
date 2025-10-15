"""UCI shim for the native cpvsengine binary.

This script allows the GUI and headless harnesses that expect Python entry
points to talk to the native C engine. It forwards stdin to the binary and
writes the binary's stdout back to the Python process's stdout.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_BINARY_NAMES = ("cpvsengine", "cpvsengine.exe")
ENV_OVERRIDE = "CPVSENGINE_NATIVE_PATH"


def resolve_engine_path(explicit: Optional[str]) -> Path:
    """Resolve the native engine path.

    Preference order:
    1. Command-line override
    2. Environment variable `CPVSENGINE_NATIVE_PATH`
    3. Sibling executables next to this shim ("cpvsengine" / "cpvsengine.exe")
    4. `engines/native/build/cpvsengine` for out-of-tree builds
    """

    shim_path = Path(__file__).resolve()
    base_dir = shim_path.parent

    if explicit:
        candidates: Iterable[Path] = (Path(explicit),)
    else:
        env_value = os.environ.get(ENV_OVERRIDE)
        candidate_list = []
        if env_value:
            candidate_list.append(Path(env_value))
        for name in DEFAULT_BINARY_NAMES:
            candidate_list.append(base_dir / name)
        # CMake-style tree (if the user builds into engines/native/build/)
        for name in DEFAULT_BINARY_NAMES:
            candidate_list.append(base_dir / "build" / name)
        candidates = tuple(candidate_list)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # If nothing matches, return the first candidate even if missing so callers
    # can display an informative error that includes the expected path.
    return next(iter(candidates))


def pump_stdout(process: subprocess.Popen[str]) -> None:
    if not process.stdout:
        return

    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shim for the native cpvsengine binary")
    parser.add_argument(
        "--engine-path",
        dest="engine_path",
        help="Path to the compiled cpvsengine binary (defaults to sibling executable).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    engine_path = resolve_engine_path(args.engine_path)
    if not engine_path or not engine_path.exists():
        sys.stderr.write(
            f"error: cpvsengine binary not found. Expected at: {engine_path}\n"
        )
        sys.stderr.flush()
        return 2

    try:
        process = subprocess.Popen(
            [str(engine_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:  # pragma: no cover - depends on local toolchain
        sys.stderr.write(f"error: failed to start native engine: {exc}\n")
        sys.stderr.flush()
        return 1

    stdout_thread = threading.Thread(target=pump_stdout, args=(process,), daemon=True)
    stdout_thread.start()

    try:
        for line in sys.stdin:
            if process.poll() is not None:
                break
            if not process.stdin:
                break
            process.stdin.write(line)
            process.stdin.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if process.stdin:
            try:
                process.stdin.close()
            except Exception:
                pass

    try:
        return_code = process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return_code = process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return_code = process.returncode

    stdout_thread.join(timeout=1.0)

    if return_code is None:
        return_code = 0
    return int(return_code)


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
