#!/bin/bash

set -eu

# clean all
rm -rf build dist qubecalib.egg-info

python -m build
