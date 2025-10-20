#!/usr/bin/env bash
# Quick inference test script

cd /home/clara/manhhd/nvidia-nemotron-h-training || exit 1

PYTHON="/home/clara/.conda/envs/mamba_temp/bin/python"

echo "======================================================================="
echo "🧪 Testing Fine-tuned Nemotron Model"
echo "======================================================================="
echo ""

# Test question from dataset
QUESTION="Git reset --soft và --hard khác nhau thế nào?"

echo "Test Question: $QUESTION"
echo ""
echo "Expected Response Pattern:"
echo "  <think>[Reasoning about state-space models...]</think>"
echo "  [Answer about Mamba2...]"
echo ""
echo "======================================================================="
echo ""

$PYTHON inference.py \
  --base-model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --adapter ./output/nemotron-sft-trained \
  --question "$QUESTION" \
  --max-length 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --load-in-4bit

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Inference test completed!"
else
    echo "✗ Inference failed with exit code: $EXIT_CODE"
fi
