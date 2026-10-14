#!/usr/bin/env bash
#
# Sharded coverage pipeline for FIDESlib.
#
# The monolithic run (run_coverage.sh) executes the whole suite in one process,
# which needs more GPU memory than a single test process holds and, importantly,
# cannot survive environments that kill long-running background jobs (e.g. a
# 60-minute wall-clock limit). This driver instead runs every test case in its
# own short-lived process:
#
#   - Each case starts with a fresh process, so peak memory is per-test and a
#     crash/abort in one case (rc 134/139/...) can't take the others down.
#   - gcov counters accumulate across the clean exits of each case process, so a
#     single gcovr pass at the end yields full-suite metrics.
#   - Progress is tracked in a done-file, so the run is resumable: relaunching
#     the script after it was killed continues from where it left off.
#   - NOTE: tests run as foreground children (no background subprocesses); this
#     sandbox terminates shells that spawn background children mid-flight.
#
# Usage:
#   ./scripts/run_coverage_sharded.sh                                     # full suite (minus benchmarks)
#   TEST_FILTER='OpenFHEInterfaceTests*:OpenFHECompatTests*' ./scripts/run_coverage_sharded.sh
#   CASE_FILE=/tmp/batch.txt ./scripts/run_coverage_sharded.sh            # run exactly the listed cases
#   BUILD_DIR=build-coverage-openfhe ./scripts/run_coverage_sharded.sh    # custom build dir
#
# Notes:
#   - Requires the coverage build already configured and built (see
#     scripts/run_coverage.sh); it does not build.
#   - The .gcda counters are NOT cleared: existing counters (e.g. from an
#     earlier run_coverage.sh) are merged with the sharded measurements.
#   - Benchmark and timing suites are skipped (P2PBenchmark, APIbench,
#     LLMTests, Microbench) like in run_coverage.sh's gcovr config scope.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

BUILD_DIR="${BUILD_DIR:-build-coverage}"
REPORT_DIR="${REPORT_DIR:-coverage}"
BIN="${BUILD_DIR}/fideslib-test"
STATS_DIR="${REPORT_DIR}/.stats"
DONE_FILE="${STATS_DIR}/done.log"
CASES_FILE="${STATS_DIR}/cases.txt"
CASE_FILE="${CASE_FILE:-}"
FULL_RESTART="${FULL_RESTART:-0}"
CASE_LOG_DIR="${STATS_DIR}/cases"
SUMMARY_FILE="${STATS_DIR}/sharded-summary.log"
TEST_FILTER="${TEST_FILTER:-}"

if [[ ! -x "${BIN}" ]]; then
    echo "ERROR: ${BIN} not found. Configure and build the coverage build first" >&2
    echo "       (e.g. BUILD_DIR=build-coverage ./scripts/run_coverage.sh)." >&2
    exit 2
fi

mkdir -p "${STATS_DIR}" "${CASE_LOG_DIR}"
: > "${SUMMARY_FILE}"
trap 'echo "trap $(date -u +%FT%T)" >> "${SUMMARY_FILE}"' TERM HUP INT

# Case list: an explicit CASE_FILE wins; otherwise enumerate (excluding
# benchmark/timing suites) and optionally narrow with TEST_FILTER tokens.
if [[ -n "${CASE_FILE}" ]]; then
    cp "${CASE_FILE}" "${CASES_FILE}"
    echo "using CASE_FILE: ${CASE_FILE} ($(wc -l < "${CASES_FILE}") cases)" | tee -a "${SUMMARY_FILE}"
else
    echo "    filter: ${TEST_FILTER:-<all unit tests>}" | tee -a "${SUMMARY_FILE}"
    python3 - "$CASES_FILE" "$TEST_FILTER" <<'PY'
import os, re, subprocess, sys
out = subprocess.run(['build-coverage/fideslib-test', '--gtest_list_tests'],
                     capture_output=True, text=True).stdout
names, suite = [], None
for line in out.splitlines():
    if line.startswith('  '):
        t = line.strip().split()[0]
        if suite:
            names.append(suite + '.' + t)
    else:
        # list_tests suite line keeps its trailing '.'; drop it so the full
        # name is single-dot joined.
        suite = line.rstrip().rstrip('.')
skip = ('P2PBenchmark.', 'APIbench.', 'LLMTests/', 'Microbench.', 'BtsTiming')
names = [n for n in names if not n.startswith(skip) and 'DISABLED' not in n]
flt = sys.argv[2]
if flt:
    # Translate an approximate TEST_FILTER into exact full names: keep the
    # cases whose full name contains every ':'-separated token as a substring.
    tokens = [t for t in flt.split(':') if t]
    names = [n for n in names if any(tok in n for tok in tokens)]
with open(sys.argv[1], 'w') as f:
    f.write('\n'.join(names) + '\n')
print(f"enumerated {len(names)} cases")
PY
fi

# Sanity: the full case list must be matched 1:1 by gtest, or the filters are wrong.
ALL_CASES="$(paste -sd: "${CASES_FILE}")"
NMATCH="$("${BIN}" --gtest_list_tests --gtest_filter="${ALL_CASES}" 2>/dev/null | grep -cE '^  [A-Za-z]')"
NCASES="$(wc -l < "${CASES_FILE}")"
echo "validation: ${NCASES} cases, gtest matched ${NMATCH}" | tee -a "${SUMMARY_FILE}"
if [[ "${NMATCH}" -ne "${NCASES}" ]]; then
    echo "ERROR: enumeration validation failed (${NMATCH} != ${NCASES})." | tee -a "${SUMMARY_FILE}"
    exit 96
fi

# Fresh runs start from an empty done-file; a re-run (e.g. after the job was
# killed) resumes by skipping the cases already in the done-file.
if [[ "${FULL_RESTART}" == "1" || ! -f "${DONE_FILE}" ]]; then
    : > "${DONE_FILE}"
fi

PASSED=0; FAILED=0; SKIPPED=0; CRASHED=0
while IFS= read -r name; do
    [[ -z "${name}" ]] && continue
    if grep -qxF "${name}" "${DONE_FILE}" 2>/dev/null; then continue; fi
    safe="$(printf '%s' "${name}" | tr '/.' '__')"
    LOG="${CASE_LOG_DIR}/${safe}.log"

    echo "RUN $(date +%s) ${name}" >> "${STATS_DIR}/timeline.csv"
    set +e
    "${BIN}" --gtest_filter="${name}" > "${LOG}" 2>&1
    RC=$?
    set -e
    echo "END $(date +%s) rc=${RC} ${name}" >> "${STATS_DIR}/timeline.csv"

    if grep -q "0 tests from 0 test suites" "${LOG}"; then
        echo "ERROR: case filter matched nothing: ${name}" | tee -a "${SUMMARY_FILE}"
        exit 97
    fi
    case "${RC}" in
        0) PASSED=$((PASSED+1)) ;;
        *) if grep -q '^\[  SKIPPED \]' "${LOG}"; then SKIPPED=$((SKIPPED+1)); else FAILED=$((FAILED+1)); fi ;;
    esac
    echo "${name}" >> "${DONE_FILE}"
done < "${CASES_FILE}"

echo "passed=${PASSED} failed=${FAILED} skipped=${SKIPPED} at $(date -u +%FT%T)" | tee -a "${SUMMARY_FILE}"

# Merged report from the accumulated counters.
echo "=== generating merged report $(date -u +%FT%T) ===" | tee -a "${SUMMARY_FILE}"
gcovr --root "${PROJECT_ROOT}" \
    --object-directory "${BUILD_DIR}" \
    --config scripts/coverage.gcovr.cfg \
    --gcov-ignore-errors=source_not_found \
    --html "${REPORT_DIR}/index.html" --html-details \
    --cobertura "${REPORT_DIR}/coverage.xml" \
    --print-summary >> "${SUMMARY_FILE}" 2>&1
echo "gcovr rc=$? at $(date -u +%FT%T)" | tee -a "${SUMMARY_FILE}"

# Static kernel-usage report (complements the host-only gcov numbers).
if command -v python3 >/dev/null 2>&1; then
    echo "=== kernel-usage report $(date -u +%FT%T) ===" | tee -a "${SUMMARY_FILE}"
    python3 scripts/kernel_usage.py --root "${PROJECT_ROOT}" \
        --coverage-xml "${REPORT_DIR}/coverage.xml" \
        --output "${KERNEL_REPORT:-KERNEL_USAGE.md}" | tee -a "${SUMMARY_FILE}" 2>&1
fi
echo "=== sharded run end $(date -u +%FT%T) ===" | tee -a "${SUMMARY_FILE}"
