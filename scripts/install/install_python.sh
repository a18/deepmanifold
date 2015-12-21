#!/usr/bin/env sh

echo "Installing python requirements..."
pip install cython
pip install -r scripts/install/python-requirements.txt
echo "Done."

