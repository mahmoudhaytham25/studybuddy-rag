"""Application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from .core.config import Settings, db_path
    from .core.store import Store
    from .ui.main_window import MainWindow, _STYLESHEET

    app = QApplication(sys.argv)
    app.setApplicationName("StudyBuddy")
    app.setOrganizationName("MahmoudHaytham")
    app.setStyleSheet(_STYLESHEET)

    settings = Settings.load()
    store = Store(db_path())
    window = MainWindow(settings, store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
