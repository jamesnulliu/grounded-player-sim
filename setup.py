"""Shim so editable/legacy installs work.

All real configuration lives in ``pyproject.toml``; this file exists only
so that ``pip install -e .`` and tooling that still shells out to
``setup.py`` keep working.
"""

from setuptools import setup

setup()
