#!/bin/bash

# https://github.com/scipy/scipy/issues/13409#issuecomment-960628840
brew install openblas
pipenv run pip install cython pybind11 pythran numpy
OPENBLAS=$(brew --prefix openblas) CFLAGS="-falign-functions=8 ${CFLAGS}" pipenv run pip install --no-use-pep517 scipy
pipenv run pip install -e ./wilds