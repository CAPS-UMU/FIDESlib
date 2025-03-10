#!/usr/bin/env bash

# Must provide the path to the benchmarking tool
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <path-to-benchmarking-tool>"
	exit 1
fi
if [ ! -f "$1" ]; then
	echo "Error: $1 not found"
	exit 1
fi

# Run the benchmarking tool
$1 --benchmark_out=result.json --benchmark_out_format=json --benchmark_counters_tabular=true