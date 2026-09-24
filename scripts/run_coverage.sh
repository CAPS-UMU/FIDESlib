#!/usr/bin/env bash
#
# FIDESlib coverage pipeline.
#
# Builds the test suite with gcov instrumentation, runs it, and produces gcovr
# reports (HTML for browsing + Cobertura XML for Codecov). Optionally enforces
# a minimum line-coverage threshold and uploads the result to Codecov.
#
# Usage:
#   ./scripts/run_coverage.sh                                          # full run (configure+build+test+report)
#   COVERAGE_MIN_LINE=60 ./scripts/run_coverage.sh                     # fail if line coverage < 60%
#   CODECOV_TOKEN=xxxx ./scripts/run_coverage.sh                       # also upload to Codecov
#   BUILD_DIR=build-coverage-openfhe ./scripts/run_coverage.sh         # custom build directory
#
# Requirements:
#   - A working CUDA environment (tests need a GPU).
#   - gcovr:  pip3 install gcovr
#   - codecov-cli (only for the upload step): pip3 install codecov-cli
#
# Scope / limitations:
#   gcov instruments the host side of .cpp and .cu files. Device code
#   (__device__/__global__ bodies) is NOT instrumented and will show as
#   uncovered; kernel-heavy files will therefore report low percentages.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# Configuration (overridable through the environment).
BUILD_DIR="${BUILD_DIR:-build-coverage}"
REPORT_DIR="${REPORT_DIR:-coverage}"
GCOVR_CONFIG="scripts/coverage.gcovr.cfg"
COVERAGE_MIN_LINE="${COVERAGE_MIN_LINE:-0}"
JOBS="${JOBS:-$(nproc)}"

# Honor a non-default OpenFHE install if requested.
OPENFHE_FLAGS=()
if [[ -n "${OPENFHE_INSTALL_PREFIX:-}" ]]; then
    OPENFHE_FLAGS+=("-DOPENFHE_INSTALL_PREFIX=${OPENFHE_INSTALL_PREFIX}")
fi

# Optionally pin CUDA architectures (must include the arch of the GPU that will
# run the tests; unset = portable list, FIDESLIB_ARCH=auto = detect via nvidia-smi).
ARCH_FLAGS=()
if [[ -n "${FIDESLIB_ARCH:-}" ]]; then
    ARCH_FLAGS+=("-DFIDESLIB_ARCH=${FIDESLIB_ARCH}")
fi

echo "==> Configuring coverage build (${BUILD_DIR})"
cmake -S . -B "${BUILD_DIR}" -G Ninja \
    -DFIDESLIB_ENABLE_COVERAGE=ON \
    -DFIDESLIB_COMPILE_BENCHMARKS=OFF \
    "${OPENFHE_FLAGS[@]}" \
    "${ARCH_FLAGS[@]}"

echo "==> Building fideslib-test"
cmake --build "${BUILD_DIR}" --target fideslib-test -j "${JOBS}"

echo "==> Clearing stale coverage counters"
find "${BUILD_DIR}" -name '*.gcda' -delete

echo "==> Running the test suite"
set +e
"${BUILD_DIR}/fideslib-test"
TESTS_RC=$?
set -e

echo "==> Generating coverage reports (${REPORT_DIR}/)"
mkdir -p "${REPORT_DIR}"
gcovr --root "${PROJECT_ROOT}" --object-directory "${BUILD_DIR}" \
    --config "${GCOVR_CONFIG}" \
    --gcov-ignore-errors=source_not_found \
    --html "${REPORT_DIR}/index.html" --html-details \
    --cobertura "${REPORT_DIR}/coverage.xml" \
    --print-summary
GCOVR_RC=$?

echo "==> Checking coverage threshold (${COVERAGE_MIN_LINE}% lines)"
set +e
gcovr --root "${PROJECT_ROOT}" --object-directory "${BUILD_DIR}" \
    --config "${GCOVR_CONFIG}" \
    --gcov-ignore-errors=source_not_found \
    --fail-under-line "${COVERAGE_MIN_LINE}"
THRESHOLD_RC=$?
set -e

# Upload to Codecov when a token is available.
if [[ -n "${CODECOV_TOKEN:-}" && -f "${REPORT_DIR}/coverage.xml" ]]; then
    echo "==> Uploading coverage to Codecov"
    codecov-cli upload-process \
        --token "${CODECOV_TOKEN}" \
        -f "${REPORT_DIR}/coverage.xml" \
        -n "coverage-$(date +%Y%m%d-%H%M%S)"
fi

echo
if [[ ${TESTS_RC} -ne 0 ]]; then
    echo "ERROR: test suite failed (exit code ${TESTS_RC}); coverage report may be incomplete."
    echo "       See ${REPORT_DIR}/index.html"
    exit 3
fi
if [[ ${GCOVR_RC} -ne 0 ]]; then
    echo "ERROR: gcovr could not produce a report (exit code ${GCOVR_RC})."
    exit 4
fi
if [[ ${THRESHOLD_RC} -ne 0 ]]; then
    echo "ERROR: line coverage is below the ${COVERAGE_MIN_LINE}% threshold."
    exit 5
fi

echo "OK: tests passed and line coverage >= ${COVERAGE_MIN_LINE}%"
echo "    Browse the report at ${REPORT_DIR}/index.html"
