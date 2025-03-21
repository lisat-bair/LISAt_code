import numpy as np
from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        "pycocotools._mask",
        sources=["../common/maskApi.c", "pycocotools/_mask.pyx"],
        include_dirs=[np.get_include(), "../common"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    ),
    # Pybind11Extension(
    #     "pycocotools._eval",
    #     sources=["../fastcocoeval/cocoeval.cpp"],
    #     include_dirs=[np.get_include(), "../fastcocoeval"],
    #     # extra_compile_args=["-Wno-cpp", "-Wno-unused-function", "-std=c99"],
    # )
]

setup(
    name="pycocotools",
    packages=["pycocotools"],
    package_dir={"pycocotools": "pycocotools"},
    install_requires=["setuptools>=18.0", "cython>=0.27.3", "matplotlib>=2.1.0"],
    version="2.0.2",
    ext_modules=ext_modules,
)
