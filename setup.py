"""Build entry point for tools that invoke ``setup.py`` directly.

All package metadata (name, dependencies, classifiers, URLs) is declared in
``pyproject.toml``, and the version is read from ``src/featransform/_version.py``.
Release with ``python -m build`` (or ``python setup.py sdist bdist_wheel``)
followed by ``twine upload dist/*``.
"""

from setuptools import setup

setup()
