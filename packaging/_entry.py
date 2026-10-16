"""PyInstaller entry point.

The real entry (studybuddy/__main__.py) uses a relative import, which
PyInstaller cannot resolve in the top-level script — this launcher uses an
absolute import instead. Keep in sync with studybuddy/__main__.py.
"""

from studybuddy.app import main

raise SystemExit(main())
