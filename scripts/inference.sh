#!/usr/bin/env bash
# Inference script for fine-tuned Nemotron model
# Usage:
#   bash scripts/inference.sh                          # Use latest epoch with default question
#   bash scripts/inference.sh 18                       # Use specific epoch
#   bash scripts/inference.sh 18 "Your question here"  # Custom epoch and question
#   EPOCH=15 bash scripts/inference.sh                 # Use environment variable

set -e

# =============================================================================
# Configuration
# =============================================================================

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Python executable
PYTHON="/home/clara/.conda/envs/mamba_temp/bin/python"

# Default settings
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/output/nemotron-sft-trained"
DEFAULT_BASE_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"

# Parse arguments
if [ -n "$1" ]; then
    EPOCH_NUM="$1"
else
    # Find latest epoch if not specified
    if [ -n "$EPOCH" ]; then
        EPOCH_NUM="$EPOCH"
    else
        # Auto-detect latest epoch
        LATEST_EPOCH=$(ls -d ${DEFAULT_OUTPUT_DIR}/epoch-* 2>/dev/null | sed 's/.*epoch-//' | sort -n | tail -1)
        if [ -z "$LATEST_EPOCH" ]; then
            echo "❌ Error: No trained epochs found in ${DEFAULT_OUTPUT_DIR}"
            echo "Please train the model first using: bash scripts/train.sh"
            exit 1
        fi
        EPOCH_NUM="$LATEST_EPOCH"
    fi
fi

ADAPTER_PATH="${DEFAULT_OUTPUT_DIR}/epoch-${EPOCH_NUM}"

# Check if adapter exists
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "❌ Error: Adapter not found at: $ADAPTER_PATH"
    echo ""
    echo "Available epochs:"
    ls -d ${DEFAULT_OUTPUT_DIR}/epoch-* 2>/dev/null | sed 's/.*epoch-/  • epoch-/' || echo "  (none)"
    exit 1
fi

# Question (from argument or default)
if [ -n "$2" ]; then
    QUESTION="$2"
else
    QUESTION="${QUESTION:-Git reset --soft và --hard khác nhau thế nào?}"
fi

# Generation settings
MAX_LENGTH="${MAX_LENGTH:-512}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
USE_4BIT="${USE_4BIT:-true}"

# =============================================================================
# Display Configuration
# =============================================================================

echo "======================================================================="
echo "🧪 Nemotron Model Inference"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  • Base Model: ${DEFAULT_BASE_MODEL}"
echo "  • Adapter: epoch-${EPOCH_NUM}"
echo "  • Path: ${ADAPTER_PATH}"
echo ""
echo "Question:"
echo "  ${QUESTION}"
echo ""
echo "Generation Settings:"
echo "  • Max Length: ${MAX_LENGTH}"
echo "  • Temperature: ${TEMPERATURE}"
echo "  • Top-p: ${TOP_P}"
echo "  • 4-bit Quantization: ${USE_4BIT}"
echo ""
echo "======================================================================="
echo ""

# =============================================================================
# Run Inference
# =============================================================================

cd "${PROJECT_ROOT}"

CMD="$PYTHON -m src.inferences \
  --base-model ${DEFAULT_BASE_MODEL} \
  --adapter ${ADAPTER_PATH} \
  --question \"${QUESTION}\" \
  --max-length ${MAX_LENGTH} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P}"

if [ "$USE_4BIT" = "false" ]; then
    CMD="${CMD} --no-4bit"
fi

eval $CMD

EXIT_CODE=$?

echo ""
echo "======================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference completed successfully!"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
fi
echo "======================================================================="

exit $EXIT_CODE
