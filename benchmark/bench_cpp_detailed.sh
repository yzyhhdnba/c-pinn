#!/bin/bash
# ==============================================================
# Comprehensive C++ PINN benchmark runner
# Runs multiple configurations, captures per-run timing and loss
# Outputs structured JSON for comparison with PyTorch
# ==============================================================

set -e

BUILD_DIR="/Users/hhd/Desktop/test/c-pinn/build"
BENCHMARK_DIR="/Users/hhd/Desktop/test/c-pinn/benchmark"
EXAMPLES_DIR="$BUILD_DIR/examples"

cd /Users/hhd/Desktop/test/c-pinn

# Ensure we have a fresh build
echo "=== Building C++ PINN (Release) ==="
cd "$BUILD_DIR"
make -j$(sysctl -n hw.ncpu) 2>&1 | tail -3
cd /Users/hhd/Desktop/test/c-pinn
echo ""

EQUATIONS=("kdv" "sine_gordon" "allen_cahn")
EXECUTABLES=("example_pure_c_kdv" "example_pure_c_sine_gordon" "example_pure_c_allen_cahn")

# ===================================================
# Test 1: Default config, 5 runs each (statistical)
# ===================================================
echo "============================================================"
echo "TEST 1: Default config × 5 runs (statistical)"
echo "============================================================"

for idx in 0 1 2; do
    eq=${EQUATIONS[$idx]}
    exe=${EXECUTABLES[$idx]}
    echo "--- $eq ---"
    for run in 1 2 3 4 5; do
        echo -n "  Run $run: "
        # Capture output and timing
        output=$( { /usr/bin/time -l "$EXAMPLES_DIR/$exe" 2>&1; } 2>&1 )
        # Extract final loss (last "Loss:" line)
        final_loss=$(echo "$output" | grep "Loss:" | tail -1 | awk '{print $NF}')
        # Extract all loss lines
        losses=$(echo "$output" | grep "Loss:")
        # Extract real time (user time from time -l output)
        user_time=$(echo "$output" | grep "user" | head -1 | awk '{print $1}')
        # Extract peak RSS
        rss=$(echo "$output" | grep "maximum resident" | awk '{print $1}')
        rss_mb=$(echo "scale=2; $rss / 1048576" | bc 2>/dev/null || echo "N/A")
        echo "loss=$final_loss  time=${user_time}s  RSS=${rss_mb}MB"
    done
    echo ""
done

echo "============================================================"
echo "Detailed per-iteration data (single canonical run)"
echo "============================================================"

for idx in 0 1 2; do
    eq=${EQUATIONS[$idx]}
    exe=${EXECUTABLES[$idx]}
    echo ""
    echo "=== $eq ==="
    /usr/bin/time -l "$EXAMPLES_DIR/$exe" 2>&1
    echo ""
done
