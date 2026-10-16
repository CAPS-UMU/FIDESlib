#!/usr/bin/env bash
#
# Build and install the OpenFHE release required by FIDESlib.
#
# Usage: build.sh <openfhe-install-prefix>
#
# The OpenFHE tree can be obtained in one of three ways, selected by
# FIDESLIB_OPENFHE_SOURCE:
#
#   patch (default) - the official OpenFHE v1.6.1 release plus the FIDESlib
#                     reference patch (deps/fideslib-1.6.1-compile.patch), which
#                     exposes the accessors FIDESlib needs and carries the
#                     bootstrap/Chebyshev reference changes it must match.
#                     Self-contained: no FIDESlib-specific fork branch is needed.
#
#   patch-minimal   - the official OpenFHE v1.6.1 release plus the compile-only
#                     subset of the reference patch
#                     (deps/fideslib-1.6.1-compile-minimal.patch). It keeps the
#                     visibility/API additions FIDESlib compiles against but
#                     drops the bit-exact CPU reference changes (the FIDESlib
#                     fused Chebyshev/Patterson-Stockmeyer path, the bootstrap
#                     precompute tweaks, etc.). Enough to build FIDESlib, but
#                     the CPU reference arithmetic is the stock OpenFHE one.
#                     Set FIDESLIB_OPENFHE_PATCH to override the file.
#
#   port            - the fully modified 1.6.1-fideslib branch from the FIDESlib
#                     fork (carlostriste/openfhe-development), which already
#                     contains the port.
#
# For every mode the install is built with static libraries (BUILD_STATIC=ON):
# FIDESLib links ${OpenFHE_LIBDIR}/libOPENFHE*_static.a, and OpenFHE's default
# (non-Emscripten) build is shared-only.
#
# Overridable variables (environment or CMake):
#   FIDESLIB_OPENFHE_SOURCE       patch | patch-minimal | port  (default: patch)
#   FIDESLIB_OPENFHE_RELEASE_TAG  patch mode release tag        (default: v1.6.1)
#   FIDESLIB_OPENFHE_PATCH        patch mode patch file         (default: <script dir>/fideslib-1.6.1-compile.patch,
#                                                               or ...-compile-minimal.patch for patch-minimal)
#   FIDESLIB_OPENFHE_PORT_BRANCH  port mode branch             (default: 1.6.1-fideslib)
#   FIDESLIB_OPENFHE_FORK_URL     port mode remote             (default: https://github.com/carlostriste/openfhe-development.git)
#   FIDESLIB_OPENFHE_RELEASE_URL  patch mode remote            (default: https://github.com/openfheorg/openfhe-development.git)
#   JOBS                          parallel build jobs          (default: nproc)

set -e
set -x

if [ -z "$1" ]; then
    echo "Usage: $0 <openfhe-install-prefix>" >&2
    exit 1
fi
INSTALL_PREFIX="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPENFHE_SOURCE="${FIDESLIB_OPENFHE_SOURCE:-patch}"
OPENFHE_FORK_URL="${FIDESLIB_OPENFHE_FORK_URL:-https://github.com/carlostriste/openfhe-development.git}"
OPENFHE_RELEASE_URL="${FIDESLIB_OPENFHE_RELEASE_URL:-https://github.com/openfheorg/openfhe-development.git}"
OPENFHE_PORT_BRANCH="${FIDESLIB_OPENFHE_PORT_BRANCH:-1.6.1-fideslib}"
OPENFHE_RELEASE_TAG="${FIDESLIB_OPENFHE_RELEASE_TAG:-v1.6.1}"
OPENFHE_PATCH="${FIDESLIB_OPENFHE_PATCH:-}"
JOBS="${JOBS:-$(nproc)}"

# Remove previous installation.
rm -rf openfhe-install
rm -rf openfhe-src

# Target installation directory.
mkdir -p "$INSTALL_PREFIX"
git submodule update --init --recursive --remote

# Source submodule.
cd openfhe-src

case "$OPENFHE_SOURCE" in
    patch|patch-minimal)
        # Resolve the patch file default per mode; FIDESLIB_OPENFHE_PATCH overrides.
        if [ -z "$OPENFHE_PATCH" ]; then
            if [ "$OPENFHE_SOURCE" = "patch-minimal" ]; then
                OPENFHE_PATCH="${SCRIPT_DIR}/fideslib-1.6.1-compile-minimal.patch"
            else
                OPENFHE_PATCH="${SCRIPT_DIR}/fideslib-1.6.1-compile.patch"
            fi
        fi
        echo "Fetching OpenFHE release ${OPENFHE_RELEASE_TAG} from ${OPENFHE_RELEASE_URL}"
        git fetch "${OPENFHE_RELEASE_URL}" "tag" "${OPENFHE_RELEASE_TAG}"
        git checkout -f FETCH_HEAD
        echo "Applying FIDESlib compile patch ${OPENFHE_PATCH}"
        git apply "${OPENFHE_PATCH}"
        ;;
    port)
        echo "Fetching FIDESlib OpenFHE port ${OPENFHE_PORT_BRANCH} from ${OPENFHE_FORK_URL}"
        git fetch "${OPENFHE_FORK_URL}" "${OPENFHE_PORT_BRANCH}"
        git checkout -f FETCH_HEAD
        ;;
    *)
        echo "Unknown FIDESLIB_OPENFHE_SOURCE='${OPENFHE_SOURCE}' (expected 'patch', 'patch-minimal' or 'port')" >&2
        exit 1
        ;;
esac

# Compilation and installation.
mkdir -p build
cd build
echo "Installing into ${INSTALL_PREFIX}"
# FIDESLib links the static OpenFHE archives (${OpenFHE_LIBDIR}/libOPENFHE*_static.a),
# so force BUILD_STATIC=ON: OpenFHE's non-Emscripten default is shared-only, which builds
# and installs but leaves FIDESLib unable to link.
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_STATIC=ON -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" ..
make -j"${JOBS}"
make install -j"${JOBS}"
