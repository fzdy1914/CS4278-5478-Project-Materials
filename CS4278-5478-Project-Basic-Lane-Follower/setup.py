#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from setuptools import Extension
from setuptools.command.build_py import build_py

TRACE_SKELETON_EXT = Extension(
    name="_trace_skeleton",
    sources=[
        "src/intelligent_robots_project/trace_skeleton/trace_skeleton.c",
        "src/intelligent_robots_project/trace_skeleton/trace_skeleton.i",
    ],
)


class BuildPy(build_py):
    def run(self):
        self.run_command("build_ext")
        super(build_py, self).run()


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="intelligent-robots-project",
    version="0.0.0",
    license="BSD-2-Clause",
    description="",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        "Topic :: Utilities",
        "Private :: Do Not Upload",
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    ext_modules=[TRACE_SKELETON_EXT],
    cmdclass={
        'build_py': BuildPy,
    },
    python_requires=">=3.6",
    install_requires=[
        "typer",
        "matplotlib",
        "scipy",
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        "dev": ["pytest"]
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        "console_scripts": [
            "intelligent-robots-project-example = intelligent_robots_project.example:main",
        ]
    },
)
