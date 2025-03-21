#!/bin/bash
set -e

# Get the directory of this script so that we can reference paths correctly no matter which folder
# the script was launched from:
SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(realpath "${SCRIPTS_DIR}")"
echo "SCRIPTS_DIR: ${SCRIPTS_DIR}"
echo "PROJ_ROOT: ${PROJ_ROOT}"
source "${PROJ_ROOT}/manifest"

# If you don't use anaconda  you can replace the relevant environment creation and activation lines
# with pyenv or whatever system you use to manage python environments.
# source ~/anaconda3/etc/profile.d/conda.sh
source ~/mambaforge/etc/profile.d/conda.sh
source ~/mambaforge/etc/profile.d/mamba.sh

ENV_NAME=$PYTHON_ENV_NAME
echo "ENV_NAME: ${ENV_NAME}"

## Remove env if exists:
mamba deactivate && mamba env remove --name "${ENV_NAME}"
rm -rf "${HOME}/mambaforge/envs/${ENV_NAME}"

# Create env:
mamba create --name "${ENV_NAME}" python=="${PYTHON_VERSION}" \
    cython setuptools==68.0.0 pip wheel ninja -y \
    -c pytorch -c nvidia -c conda-forge -c defaults

mamba activate "${ENV_NAME}"
echo "Current environment: "
mamba info --envs | grep "*"

##
## Base dependencies
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Make the python environment available for running jupyter kernels:
python -m ipykernel install --user --name="${ENV_NAME}"

# Install jupyter extensions
# GB 2023-12-22: This no longer works, gives error: ModuleNotFoundError: No module named 'notebook.base'
# jupyter contrib nbextension install --user

pushd ./PythonAPI || exit
pip install -e .
popd || exit

# We are done, show the python environment:
conda list
echo "Done!"
