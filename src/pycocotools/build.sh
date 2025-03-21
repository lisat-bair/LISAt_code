#!/bin/bash
set -e


pushd ./PythonAPI
rm -rf build
rm -rf *.egg-info
rm -rf dist
python setup.py build_ext install
rm -rf build
popd