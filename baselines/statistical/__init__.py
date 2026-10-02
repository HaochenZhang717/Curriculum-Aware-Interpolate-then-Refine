"""Baselines package. Adds itself to sys.path so the modules can import each other with bare names (matches the legacy import pattern)."""

import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
