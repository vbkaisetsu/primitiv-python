#!/bin/bash
set -xe

pip install twine
if [ "${BINARY_PACKAGE}" = "yes" ]; then
  twine upload --repository-url https://test.pypi.org/legacy/ -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" $TRAVIS_BUILD_DIR/wheelhouse/primitiv-*.whl;
else
  twine upload --repository-url https://test.pypi.org/legacy/ -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" $TRAVIS_BUILD_DIR/dist/primitiv-*.tar.gz;
fi
