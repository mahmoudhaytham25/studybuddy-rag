#!/usr/bin/env bash
# Local macOS build: PyInstaller -> dist/StudyBuddy.app -> dist/StudyBuddy-macOS.dmg
#
#   pip install -e ".[package]"
#   ./scripts/build_mac.sh
#
# Override the interpreter with PYTHON=/path/to/python if .venv isn't used.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m PyInstaller packaging/studybuddy.spec --clean --noconfirm

hdiutil create -volname StudyBuddy \
    -srcfolder dist/StudyBuddy.app \
    -ov -format UDZO \
    dist/StudyBuddy-macOS.dmg

echo "Built dist/StudyBuddy.app and dist/StudyBuddy-macOS.dmg"
