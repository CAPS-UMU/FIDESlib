#!/bin/bash
# Run variant /0 of each OpenFHEBootstrapTest case individually.
set -u
BIN=build/fideslib-test
OUTDIR="/tmp/opencode/boot_tests"
mkdir -p "$OUTDIR"

TEST_CASES="ApproxModEval ApproxModEvalSparse LinearTransform CoeffsToSlots SlotsToCoeffs OpenFHEBootstrapCPUsetup OpenFHEBootstrap OpenFHEBootstrapManualPrescale OpenFHEBootstrapLT OpenFHEBootstrapDense"

for test in $TEST_CASES; do
  LOG="$OUTDIR/${test}.log"
  echo "=== RUN $test ===" >> "$LOG"
  timeout 3000 ./"$BIN" --gtest_filter="OpenFHEBootstrapTests/OpenFHEBootstrapTest.${test}/0" >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then
    echo "PASS  $test"
  elif [ $RC -eq 134 ]; then
    echo "ABORT $test (uncaught exception / signal 6)"
  elif [ $RC -eq 124 ]; then
    echo "TIMEOUT $test"
  else
    echo "FAIL  $test (rc=$RC)"
  fi
done
