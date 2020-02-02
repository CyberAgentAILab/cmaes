# type: ignore

import os
import types

from setuptools import setup, find_packages
from importlib.machinery import SourceFileLoader

BASE_PATH = os.path.abspath(os.path.dirname(__file__))


def get_version():
    version_filepath = os.path.join(BASE_PATH, "cmaes", "version.py")
    module_name = "version"
    target_module = types.ModuleType(module_name)
    loader = SourceFileLoader(module_name, version_filepath)
    loader.exec_module(target_module)

    return getattr(target_module, "__version__")


setup(
    name="cmaes",
    version=get_version(),
    author="Masashi Shibata",
    author_email="shibata_masashi@cyberagent.co.jp",
    url="https://github.com/CyberAgent/cmaes",
    description="Lightweight Covariance Matrix Adaptation Evolution Strategy (CMA-ES) "
    "implementation for Python 3.",
    long_description=open(os.path.join(BASE_PATH, "README.md")).read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
    ],
    packages=find_packages(exclude=["test*", "benchmark*", "examples"]),
    install_requires=["numpy"],
    extras_require={
        "benchmark": ["kurobako", "cma", "optuna"],
        "visualization": ["matplotlib", "scipy"],
        "lint": ["mypy", "flake8", "black", "optuna"],
        "test": ["optuna"],
        "release": ["wheel", "twine"],
    },
    tests_require=[],
    keywords="cma-es evolution-strategy optuna",
    license="MIT License",
    include_package_data=True,
    test_suite="tests",
)
