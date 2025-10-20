#!/usr/bin/env bash
# Dynamic training launcher
# Usage: 
#   bash train.sh                    # Default: GPU 0, wandb enabled
#   bash train.sh 1                  # GPU 1, wandb enabled
#   bash train.sh 0,1                # Multi-GPU 0,1, wandb enabled
#   GPU=2 bash train.sh              # GPU 2, wandb enabled
#   WANDB_OFF=1 bash train.sh        # GPU 0, wandb disabled
#   GPU=1 WANDB_OFF=1 bash train.sh  # GPU 1, wandb disabled

# =============================================================================
# Configuration (can be overridden by environment variables)
# =============================================================================

# GPU devices (default: 0)
GPU_DEVICES="${1:-${GPU:-0}}"

# Wandb settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
if [ ! -z "$WANDB_OFF" ]; then
    WANDB_ENABLED="false"
fi
WANDB_PROJECT="${WANDB_PROJECT:-nemotron-sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-sft-$(date +%Y%m%d-%H%M%S)}"

# Training hyperparameters
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Nano-12B-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/nemotron-sft-trained_en_gsm8k_ver12B}"
DATA_JSONL="${DATA_JSONL:-./data/vi_gsm8k.jsonl}"
PRESET="${PRESET:-mlp_only}"
RANK="${RANK:-8}"
ALPHA="${ALPHA:-16}"
DROPOUT="${DROPOUT:-0.05}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
EPOCHS="${EPOCHS:-20}"
# Optional explicit max epochs variable (can override EPOCHS)
MAX_EPOCHS="${MAX_EPOCHS:-${EPOCHS}}"
# Number of checkpoints to keep (-1 for all, N for last N)
NUM_CHECKPOINT="${NUM_CHECKPOINT:-3}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# =============================================================================
# Advanced Optimization Parameters (inspired by nanochat)
# =============================================================================

# Gradient optimization
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOGITS_SOFTCAP="${LOGITS_SOFTCAP:-15.0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

# Multi-optimizer learning rates
UNEMBEDDING_LR="${UNEMBEDDING_LR:-2e-5}"  # lr * 0.1
EMBEDDING_LR="${EMBEDDING_LR:-4e-4}"      # lr * 2.0
MATRIX_LR="${MATRIX_LR:-2e-4}"            # lr

# Muon optimizer (advanced)
USE_MUON="${USE_MUON:-false}"
MUON_LR="${MUON_LR:-0.02}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"

# Learning rate scheduling
WARMUP_STEPS="${WARMUP_STEPS:-100}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"

# =============================================================================
# Environment setup
# =============================================================================

# Set environment variables for Triton/Mamba compilation (if needed)
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDA_HOME=/usr/local/cuda
export TRITON_LIBCUDA_PATH=/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=${GPU_DEVICES}

# Use conda's python directly
PYTHON="/home/clara/.conda/envs/mamba_temp/bin/python"

# =============================================================================
# Display configuration
# =============================================================================

echo "======================================================================="
echo "🚀 Starting Nemotron SFT Training"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  • Model: ${MODEL_ID}"
echo "  • Dataset: ${DATA_JSONL}"
echo "  • Output: ${OUTPUT_DIR}"
echo "  • GPU Devices: ${GPU_DEVICES}"
echo ""
echo "LoRA Settings:"
echo "  • Preset: ${PRESET}"
echo "  • Rank: ${RANK}, Alpha: ${ALPHA}, Dropout: ${DROPOUT}"
echo ""
echo "Training Settings:"
echo "  • Batch Size: ${BATCH_SIZE} × Grad Accum: ${GRAD_ACCUM}"
echo "  • Epochs: ${EPOCHS} (max_epochs=${MAX_EPOCHS}), Learning Rate: ${LR}"
echo "  • Checkpoints: Keep ${NUM_CHECKPOINT} latest (-1 = all)"
echo "  • Sequence Length: ${SEQ_LEN}"
echo "  • Using: 4-bit quantization + BF16 + Chat Template"
echo ""
echo "Advanced Optimizations:"
echo "  • Gradient Clipping: ${GRAD_CLIP}"
echo "  • Logits Softcap: ${LOGITS_SOFTCAP}"
echo "  • Weight Decay: ${WEIGHT_DECAY}"
echo "  • Multi-Optimizer LRs:"
echo "    - Unembedding: ${UNEMBEDDING_LR}"
echo "    - Embedding: ${EMBEDDING_LR}"
echo "    - Matrix: ${MATRIX_LR}"
echo "  • Muon Optimizer: ${USE_MUON} (lr=${MUON_LR}, momentum=${MUON_MOMENTUM})"
echo "  • LR Scheduling: Warmup=${WARMUP_STEPS}, Total=${TOTAL_STEPS}"
echo ""
echo "Wandb Settings:"
echo "  • Enabled: ${WANDB_ENABLED}"
if [ "$WANDB_ENABLED" = "true" ]; then
    echo "  • Project: ${WANDB_PROJECT}"
    echo "  • Run Name: ${WANDB_RUN_NAME}"
fi
echo ""
echo "======================================================================="
echo ""

# =============================================================================
# Build command
# =============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run as Python module from project root
cd "${PROJECT_ROOT}"

CMD="$PYTHON -m src.sft \
  --model-id ${MODEL_ID} \
  --output-dir ${OUTPUT_DIR} \
  --data-jsonl ${DATA_JSONL} \
  --messages-key messages \
  --preset ${PRESET} \
  --rank ${RANK} \
  --alpha ${ALPHA} \
  --dropout ${DROPOUT} \
  --load-in-4bit \
  --bf16 \
  --seq-len ${SEQ_LEN} \
  --batch-size ${BATCH_SIZE} \
  --grad-accum ${GRAD_ACCUM} \
  --grad-checkpoint \
  --epochs ${EPOCHS} \
  --max-epochs ${MAX_EPOCHS} \
  --num-checkpoint ${NUM_CHECKPOINT} \
  --lr ${LR} \
  --use-chat-template \
  --append-eos \
  --num-workers ${NUM_WORKERS} \
  --gpu ${GPU_DEVICES} \
  --grad-clip ${GRAD_CLIP} \
  --logits-softcap ${LOGITS_SOFTCAP} \
  --weight-decay ${WEIGHT_DECAY} \
  --unembedding-lr ${UNEMBEDDING_LR} \
  --embedding-lr ${EMBEDDING_LR} \
  --matrix-lr ${MATRIX_LR} \
  --warmup-steps ${WARMUP_STEPS} \
  --total-steps ${TOTAL_STEPS}"

# Add Muon optimizer if enabled
if [ "$USE_MUON" = "true" ]; then
    CMD="${CMD} --use-muon --muon-lr ${MUON_LR} --muon-momentum ${MUON_MOMENTUM}"
fi

# Add wandb settings
if [ "$WANDB_ENABLED" = "true" ]; then
    CMD="${CMD} --wandb-project ${WANDB_PROJECT} --wandb-run-name ${WANDB_RUN_NAME}"
else
    CMD="${CMD} --wandb-off"
fi

# =============================================================================
# Run training
# =============================================================================

eval $CMD

EXIT_CODE=$?

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "======================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully!"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}/"
    echo ""
    echo "Adapter files:"
    ls -lh ${OUTPUT_DIR}/
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
fi
echo "======================================================================="

exit $EXIT_CODE

exit $EXIT_CODE
