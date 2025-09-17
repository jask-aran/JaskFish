import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

try:  # pragma: no cover - optional GUI dependency
    from PySide2.QtWidgets import QApplication  # type: ignore
except ImportError:  # pragma: no cover - GUI tooling may be unavailable
    QApplication = None  # type: ignore


def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def debug_text(text):
    return f"{color_text('DEBUG', '31')} {text}"


def info_text(text):
    return f"{color_text('INFO', '34')}  {text}"


def sending_text(text):
    return f"{color_text('SENDING  ', '32')} {text}"


def recieved_text(text):
    return f"{color_text('RECIEVED ', '35')} {text}"


def cleanup(process, thread, app, dev=False):
    if dev:
        print(debug_text("Cleaning up resources..."))

    # Safely terminate the process
    if process is not None:
        process.terminate()
        process.waitForFinished()

    if thread is not None:
        thread.join()

    app.quit()


def get_piece_unicode(piece):
    piece_unicode = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    }
    return piece_unicode[piece.symbol()]


def center_on_screen(window):
    if QApplication is None:
        return

    screen = QApplication.primaryScreen()
    if screen is None:
        return

    screen_geometry = screen.geometry()
    window_size = window.size()
    x = (screen_geometry.width() - window_size.width()) / 2 + screen_geometry.left()
    y = (screen_geometry.height() - window_size.height()) / 2 + screen_geometry.top()
    window.move(int(x), int(y))


def _normalise_search_payload(payload: Any) -> Dict[str, Any]:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, MutableMapping):
        return dict(payload)
    if isinstance(payload, Mapping):
        return dict(payload)

    raise TypeError(
        "Search payload must be a dataclass or mapping-like object to be serialised"
    )


def write_search_log_entry(
    search_info: Any,
    log_path: Path | str,
    fmt: str = "json",
    field_order: Iterable[str] | None = None,
) -> None:
    data = _normalise_search_payload(search_info)
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalised_fmt = fmt.lower()
    if normalised_fmt == "json":
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data) + "\n")
        return

    if normalised_fmt == "csv":
        fieldnames = list(field_order) if field_order else list(data.keys())
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: data.get(key, "") for key in fieldnames})
        return

    raise ValueError(f"Unsupported log format: {fmt}")
