#!/usr/bin/env bash

set -e
set -x

#Remove previous installation.
rm -rf openfhe-install
rm -rf openfhe-src

# Target installation directory.
mkdir -p $1
git submodule update --init --recursive --remote

#Source submodule.
cd openfhe-src
git checkout fideslib-ref-v1.5.1.1
#git checkout v1.5.1
#git config user.email "FIDESlib"
#git config user.name "FIDESlib"
git apply ../fideslib-ref-1.5.1.1.patch
#git apply ../openfhe-1.5.1.patch

# Compilation and installation.
mkdir build
cd build
echo "Installing into $1"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$1" ..
make -j12
make install -j12