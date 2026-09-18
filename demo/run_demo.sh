#!/usr/bin/env bash
# ==============================================================================
# SIH 26126 - TerrainSight UGV Demonstration Launcher
# Organization: Bharat Electronics Limited (BEL)
# System: Vision-Based Autonomous Navigation for Outdoor UGV
# ==============================================================================

set -e

# Detect script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=================================================================="
echo "    TERRAINSIGHT UGV - SIH 26126 DEMONSTRATION RUNNER"
echo "    Bharat Electronics Limited (BEL) | Vision-Only Navigation"
echo "=================================================================="

# 1. Detect Python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 was not found in PATH. Please install Python 3.9+."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# 2. Dependency check
echo "[INFO] Verifying core runtime dependencies..."
$PYTHON_CMD -c "
import sys
required = ['numpy', 'cv2', 'yaml']
missing = []
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print(f'[ERROR] Missing required packages: {missing}')
    print('[INFO] Run: pip install -r requirements.txt')
    sys.exit(1)
print('[OK] Core dependencies verified.')
"

# 3. Parse command-line arguments
MODE="dashboard"
PORT=5000
SCENARIO="scenario_1_open_path"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cli)
            MODE="cli"
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./demo/run_demo.sh [OPTIONS]"
            echo "Options:"
            echo "  --dashboard       Launch Mission Control Web Dashboard (Default)"
            echo "  --cli             Run headless terminal replay pipeline"
            echo "  --port <PORT>     Dashboard server port (Default: 5000)"
            echo "  --scenario <NAME> Scenario to replay in CLI mode (Default: scenario_1_open_path)"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "[WARN] Unknown argument: $1"
            shift
            ;;
    esac
done

# 4. Launch based on mode
if [ "$MODE" = "cli" ]; then
    echo "[INFO] Running headless CLI pipeline on demo/sample_data/${SCENARIO}..."
    $PYTHON_CMD scripts/run_replay.py --dataset-root "demo/sample_data" --scenario "${SCENARIO}"
else
    echo "------------------------------------------------------------------"
    echo "[INFO] Starting Mission Control Dashboard Server on port ${PORT}..."
    echo "[INFO] Open your web browser at: http://localhost:${PORT}"
    echo "[INFO] Press Ctrl+C to cleanly stop the server."
    echo "------------------------------------------------------------------"

    # Attempt to open browser if xdg-open or open exists (background)
    if command -v xdg-open &>/dev/null; then
        (sleep 2 && xdg-open "http://localhost:${PORT}") &
    elif command -v open &>/dev/null; then
        (sleep 2 && open "http://localhost:${PORT}") &
    fi

    # Launch dashboard server
    exec $PYTHON_CMD src/visualization/dashboard_server.py --port "${PORT}"
fi
