#!/bin/bash
set -e


# ./build.sh
pushd ./tests
python -m unittest discover -v -s .
popd