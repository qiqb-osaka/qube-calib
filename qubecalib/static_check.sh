#!/bin/bash

set -eu

srcs=(
    "qubecalib/__init__.py"
    "qubecalib/backendqube.py"
    "qubecalib/meas.py"
    "qubecalib/mock_qubelsi.py"
#    "qubecalib/neopulse.py"
    "qubecalib/pulse.py"
    "qubecalib/qube.py"
#    "qubecalib/setup.py"
    "qubecalib/setupqube.py"
#    "qubecalib/ui.py"
    "qubecalib/utils.py"
#    "qubecalib/visa.py"
)

echo "[isort]"
isort "${srcs[@]}"
echo "[black]"
black "${srcs[@]}"
echo "[pflake8]"
pflake8 "${srcs[@]}"
echo "[mypy]"
mypy --check-untyped-defs "${srcs[@]}"
