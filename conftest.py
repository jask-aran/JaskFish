import os
import sys

import pytest

# Ensure repo-local imports (e.g., `import main`) resolve without extra setup.
src_dir = os.path.abspath(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "-G",
        "--gui",
        action="store_true",
        default=False,
        dest="run_gui",
        help="Run tests marked with @pytest.mark.gui (requires PySide6/display)",
    )
    parser.addoption(
        "-S",
        "--search",
        action="store_true",
        default=False,
        dest="run_search_slow",
        help="Run tests marked with @pytest.mark.search_slow (long-running search checks)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("run_gui"):
        skip_gui = pytest.mark.skip(reason="use -G/--gui to enable GUI tests")
        for item in items:
            if "gui" in item.keywords:
                item.add_marker(skip_gui)

    if not config.getoption("run_search_slow"):
        skip_slow = pytest.mark.skip(
            reason="use -S/--search to enable search smoke tests"
        )
        for item in items:
            if "search_slow" in item.keywords:
                item.add_marker(skip_slow)
