#!/bin/bash
set -xe

if [ "${TRAVIS_BRANCH}" = "develop" ]; then
  PRIMITIV_PYTHON_BUILD_NUMBER="dev${TRAVIS_BUILD_NUMBER}"
else
  PRIMITIV_PYTHON_BUILD_NUMBER="${TRAVIS_BUILD_NUMBER}"
fi

case "${TRAVIS_PYTHON_VERSION}" in
  "3.5" ) DOCKER_IMAGE="zombiefeynman/manylinux2010_x86_64"
          PATHON_BIN_DIR="/opt/python/cp35-cp35m/bin"
          ;;
  "3.6" ) DOCKER_IMAGE="zombiefeynman/manylinux2010_x86_64"
          PATHON_BIN_DIR="/opt/python/cp36-cp36m/bin"
          ;;
esac

# before_install
docker pull ${DOCKER_IMAGE}
docker run --name travis-ci -v ${TRAVIS_BUILD_DIR}:/primitiv-python --env PRIMITIV_PYTHON_BUILD_NUMBER=${PRIMITIV_PYTHON_BUILD_NUMBER} -td ${DOCKER_IMAGE} /bin/bash

# install
docker exec travis-ci bash -c "${PATHON_BIN_DIR}/pip install numpy==1.16.1 cython scikit-build cmake auditwheel=2.0.0 wheel==0.31.1"

docker exec travis-ci bash -c "cd /primitiv-python && wget -q http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 -O eigen-downloaded.tar.gz"
docker exec travis-ci bash -c "cd /primitiv-python && mkdir ./eigen-downloaded"
docker exec travis-ci bash -c "cd /primitiv-python && tar xf ./eigen-downloaded.tar.gz --strip-components=1 -C ./eigen-downloaded"

# script
docker exec travis-ci bash -c "cd /primitiv-python && ${PATHON_BIN_DIR}/python ./setup.py build --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON"
docker exec travis-ci bash -c "cd /primitiv-python && ${PATHON_BIN_DIR}/python ./setup.py build_ext -i --enable-eigen -- -DCMAKE_VERBOSE_MAKEFILE=ON"
docker exec travis-ci bash -c "cd /primitiv-python && ${PATHON_BIN_DIR}/python ./setup.py test"

# binary package
docker exec travis-ci bash -c "cd /primitiv-python && ${PATHON_BIN_DIR}/python ./setup.py bdist_wheel"
docker exec travis-ci bash -c "cd /primitiv-python && ${PATHON_BIN_DIR}/auditwheel repair --plat manylinux2010_x86_64 dist/primitiv-*.whl"

# after_script
docker stop travis-ci

pip install -U pip
pip install ${TRAVIS_BUILD_DIR}/wheelhouse/primitiv-*.whl
mkdir ./work
pushd ./work
python -c "import primitiv; dev = primitiv.devices.Eigen()"
popd
