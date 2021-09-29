#!/usr/bin/env python
"""Setup script for Forager"""

import os
import sys

name = "forager-server"
version = "0.0.2-6"

if sys.version_info < (3, 8):
    error = (
        "Forager supports Python 3.8 and above. " f"Python {sys.version_info} detected."
    )
    print(error, file=sys.stderr)
    sys.exit(1)

# BEFORE importing distutils, remove MANIFEST. distutils doesn't properly
# update it when the contents of directories change.
if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")

from setuptools import setup

# Needed to support building with `setuptools.build_meta`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from forager.buildutils import (build_frontend, find_package_data,
                                find_package_deps, find_package_dirs,
                                find_packages)

setup_args = dict(
    name=name,
    description="A web-based data exploration system for rapid task definition.",
    long_description="""
Forager is a system for rapidly exploring a corpus of data, and
annotating this data with user-defined tags.
    """,
    version=version,
    packages=find_packages(),
    package_dir=find_package_dirs(),
    package_data={},
    data_files=[],
    author="Forager Research Team",
    author_email="faitpoms@gmail.com",
    url="",
    license="",
    platforms="Linux, Mac OS X, Windows",
    keywords=["Interactive", "Interpreter", "Shell", "Web"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
    install_requires=find_package_deps(),
    scripts=[
        "setup.py",
        "pyproject.toml",
    ],
    extras_require={},
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "forager-server = forager.main:main",
        ]
    },
)

# Custom distutils/setuptools commands ----------
from distutils.command.sdist import sdist

from setuptools.command.bdist_egg import bdist_egg


class no_bdist_egg(bdist_egg):
    """Never generate python eggs"""

    def run(self):
        sys.exit(
            "Disallowing creation of python eggs. "
            "Use `pip install .` to install from source."
        )


setup_args["cmdclass"] = {
    "sdist": build_frontend(sdist),
    "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else no_bdist_egg,
}

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    pass
else:
    setup_args["cmdclass"]["bdist_wheel"] = build_frontend(bdist_wheel)

# Run setup --------------------
def main():
    setup(**setup_args)


if __name__ == "__main__":
    main()
