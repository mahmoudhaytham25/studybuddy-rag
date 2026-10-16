# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for StudyBuddy (cross-platform).

- macOS:  onedir build wrapped in a `StudyBuddy.app` bundle (BUNDLE below).
- Windows: single-file `StudyBuddy.exe`.
- Other platforms fall back to a plain onefile windowed executable.

fastembed's ONNX models are deliberately NOT bundled — they download into
the OS app-data dir on first run (see studybuddy/core/embeddings.py).

Build:  pyinstaller packaging/studybuddy.spec --clean --noconfirm
"""

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    copy_metadata,
)

# SPECPATH is the spec's directory in PyInstaller >= 5; handle it being the
# spec *file* on older versions, then resolve the repo root one level up.
_spec = Path(SPECPATH).resolve()
ROOT = (_spec if _spec.is_dir() else _spec.parent).parent

APP_NAME = "StudyBuddy"
BUNDLE_ID = "com.mahmoudhaytham.studybuddy"
ICON_ICNS = ROOT / "packaging" / "icon.icns"
ICON_ICO = ROOT / "packaging" / "icon.ico"

datas = []
binaries = []
hiddenimports = []

# fastembed lazily imports onnxruntime/tokenizers/huggingface_hub from inside
# functions, and onnxruntime/pymupdf carry shared libraries
# (libonnxruntime.*.dylib, libmupdf.*) that dependency-walking can miss.
# collect_all pulls in submodules + binaries + data files for these.
for _pkg in ("fastembed", "onnxruntime", "pymupdf"):
    _d, _b, _h = collect_all(_pkg)
    datas += _d
    binaries += _b
    hiddenimports += _h

# fastembed/__init__.py resolves its version via importlib.metadata at
# import time (unguarded fallback to "fastembed-gpu") — the frozen app
# crashes without the bundled dist-info.
datas += copy_metadata("fastembed")

# sqlite-vec loads `vec0` (dylib/so/dll) through sqlite3's extension loader
# at runtime, resolving it next to its own __file__ — the binary must ship
# inside the bundle at `sqlite_vec/`.
binaries += collect_dynamic_libs("sqlite_vec")
hiddenimports += ["sqlite_vec"]

# store.py falls back to pysqlite3 when the bundled interpreter's sqlite3
# lacks enable_load_extension. python.org / Homebrew builds (and the macOS /
# Windows CI interpreters) all have it enabled, so this is normally absent —
# bundle it only if it happens to be installed on the build machine.
if importlib.util.find_spec("pysqlite3") is not None:
    binaries += collect_dynamic_libs("pysqlite3")
    hiddenimports += ["pysqlite3"]

# Imported lazily inside functions, or via plugins — cheap insurance against
# modulegraph misses:
hiddenimports += [
    "pymupdf4llm",  # imported inside documents._load_pdf
    "google.genai",  # imported inside GeminiProvider methods
    "hf_xet",  # huggingface_hub's Rust downloader, loaded dynamically
]

a = Analysis(
    # absolute-import launcher — studybuddy/__main__.py itself uses a
    # relative import that modulegraph cannot resolve as an entry script
    [str(ROOT / "packaging" / "_entry.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # keep the bundle lean — none of these are used by StudyBuddy
        "torch",
        "torchvision",
        "matplotlib",
        "tkinter",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

if sys.platform == "darwin":
    # onedir -> .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        strip=False,
        upx=False,
        console=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=str(ICON_ICNS) if ICON_ICNS.exists() else None,
        bundle_identifier=BUNDLE_ID,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
        },
    )
else:
    # Windows (and any other platform): single-file windowed executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        strip=False,
        upx=False,
        console=False,
        icon=str(ICON_ICO) if ICON_ICO.exists() else None,
    )
