#!/bin/bash
set -ex

# Set up env vars
PYTHON_VER_YUM='36'
PYTHON_VER='3.6'
NUMPY_VER='1.19.4'
PANDAS_VER='1.1.5'
SCIPY_VER='1.5.4'
CATBOOST_VER='0.15'

LAMBDA_PACKAGE_DIR='outputs/lambda-package'
LIB_DIR="${LAMBDA_PACKAGE_DIR}/lib"
LAMBDA_PACKAGE_ZIP='lambda-package.zip'
LAMBDA_PACKAGE_ZIP_RELPATH="outputs/${LAMBDA_PACKAGE_ZIP}"

SITE_PACKAGES_DIR="venv/lib/python${PYTHON_VER}/site-packages"

yum install -y atlas-devel \
        atlas-sse3-devel \
        blas-devel \
        gcc \
        gcc-c++ \
        lapack-devel \
        python${PYTHON_VER_YUM}-devel \
        libgfortran

python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip

# the --no-binary flag hasn't been used yet
CFLAGS="-g0 -Wl,--strip-all -I/usr/include:/usr/local/include -L/usr/lib:/usr/local/lib" \
        pip install \
        --global-option=build_ext \
        --global-option="-j 4" \
        --no-cache-dir \
        --compile \
        --no-binary \
        numpy==${NUMPY_VER} scipy==${SCIPY_VER} pandas==${PANDAS_VER}

pip install \
--no-cache-dir \
--compile \
catboost==0.15

python -V
python -c "import numpy as np; print(np.version.version)"
python -c "import numpy as np; print(np.__config__.show())"
python -c "import pandas as pd; print(pd.__version__)"
python -c "import scipy as sp; print(sp.version.version)"
python -c "import catboost; print(catboost.__version__)"

echo "Preparing ${LIB_DIR}..." > /dev/null 2>&1
mkdir -p ${LIB_DIR}
echo "Preparing ${LIB_DIR}...done" > /dev/null 2>&1
ls $SITE_PACKAGES_DIR
echo "Copying ${SITE_PACKAGES_DIR} contents to ${LAMBDA_PACKAGE_DIR}..." > /dev/null 2>&1
cp -rf ${SITE_PACKAGES_DIR}/* ${LAMBDA_PACKAGE_DIR}
echo "Copying ${SITE_PACKAGES_DIR} contents to ${LAMBDA_PACKAGE_DIR}...done" > /dev/null 2>&1

echo "Copying compiled libraries to ${LIB_DIR}..." > /dev/null 2>&1
cp /usr/lib64/atlas/* ${LIB_DIR}
cp /usr/lib64/libquadmath.so.0 ${LIB_DIR}
cp /usr/lib64/libgfortran.so.3 ${LIB_DIR}

echo "Reducing package size..." > /dev/null 2>&1
echo "Original unzipped package size: $(du -sh ${LAMBDA_PACKAGE_DIR} | cut -f1)" > /dev/null 2>&1

# Remove README
rm ${LAMBDA_PACKAGE_DIR}/README
# Remove distribution info directories
rm -rf ${LAMBDA_PACKAGE_DIR}/*.egg-info
rm -rf ${LAMBDA_PACKAGE_DIR}/*.dist-info
# Remove all testing directories
find ${LAMBDA_PACKAGE_DIR} -name tests | xargs rm -rf
# strip excess from compiled .so files
find ${LAMBDA_PACKAGE_DIR} -name "*.so" | xargs strip

# Removing potentially non needed files
# keeping mainly the so.3 files
find ${LIB_DIR}/ -name "*.so" | xargs rm
find ${LIB_DIR}/ -name "*.a" | xargs rm
rm ${LIB_DIR}/liblapack.so.3.0
rm ${LIB_DIR}/libatlas.so.3.0

echo "Final unzipped package size: $(du -sh ${LAMBDA_PACKAGE_DIR} | cut -f1)" > /dev/null 2>&1
echo "Reducing package size...done" > /dev/null 2>&1

# echo "Show seperate package sizes: $(du -sh ${LAMBDA_PACKAGE_DIR}/* | sort -h)"
# cat .lambdaignore | xargs zip -9qr upload-to-s3.zip * -x
# echo "Zipped file size (excl. entries in .lambdaignore): $(du -mh upload-to-s3.zip | cut -f1)"

echo "Compressing packages into ${LAMBDA_PACKAGE_ZIP}..." > /dev/null 2>&1
# zip -r9q ${LAMBDA_PACKAGE_ZIP_RELPATH} ${LAMBDA_PACKAGE_DIR}
pushd ${LAMBDA_PACKAGE_DIR} > /dev/null 2>&1 && zip -r9q /${LAMBDA_PACKAGE_ZIP_RELPATH} * ; popd > /dev/null 2>&1
# zip -r9q ${LAMBDA_PACKAGE_ZIP_RELPATH} ${LAMBDA_PACKAGE_DIR}
echo "lambda-package.zip size: $(du -sh ${LAMBDA_PACKAGE_ZIP_RELPATH} | cut -f1)" > /dev/null 2>&1
echo "Compressing packages into lambda-package.zip...done" > /dev/null 2>&1
