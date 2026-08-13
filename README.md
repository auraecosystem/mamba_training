# NVIDIA Nemotron Fine-tuning with LoRA

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

Fine-tune NVIDIA Nemotron models using LoRA (Low-Rank Adaptation) with PyTorch Lightning and Weights & Biases integration.

## 🌟 Features

- ✅ **LoRA Fine-tuning** - Efficient parameter tuning with PEFT
- ✅ **4-bit Quantization** - Train large models on consumer GPUs
- ✅ **Wandb Integration** - Track experiments and metrics
- ✅ **Per-Epoch Checkpointing** - Save adapters after each epoch
- ✅ **Flexible Configuration** - Easy-to-use bash scripts with environment variables
- ✅ **Multi-GPU Support** - Train on single or multiple GPUs
- ✅ **Chat Template Support** - Format conversations properly
- ✅ **Checkpoint Management** - Control number of saved checkpoints

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Inference](#-inference)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Advanced Usage](#-advanced-usage)
- [Troubleshooting](#-troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ or 12.x
- 16GB+ VRAM (for 4-bit quantization)
- Conda or virtualenv

### Setup

```bash
# Clone the repository
git clone <[[your-repo-url](https://github.com/auraecosystem/lmlmodel.git)](https://github.com/auraecosystem/mamba_training.git)>
cd mamba_training

# Create conda environment
conda create -n mamba_temp python=3.10
conda activate mamba_temp

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft bitsandbytes accelerate
pip install pytorch-lightning wandb
pip install datasets jsonlines
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## ⚡ Quick Start

### 1. Prepare Your Data

Create a JSONL file with conversation format:

```json
{"messages": [{"role": "user", "content": "What is Git?"}, {"role": "assistant", "content": "Git is a distributed version control system..."}]}
{"messages": [{"role": "user", "content": "Explain Python decorators"}, {"role": "assistant", "content": "Python decorators are..."}]}
```

Save to `data/example.json`

### 2. Train the Model

```bash
# Basic training (default settings)
bash scripts/train.sh

# Custom training
MAX_EPOCHS=10 NUM_CHECKPOINT=3 bash scripts/train.sh
```

### 3. Run Inference

```bash
# Use latest trained epoch
bash scripts/inference.sh

# Use specific epoch
bash scripts/inference.sh 18

# Custom question
bash scripts/inference.sh 18 "Explain neural networks"
```

## 🎓 Training

### Basic Training

```bash
bash scripts/train.sh
```

This will:
- Load NVIDIA Nemotron Nano 9B model
- Apply LoRA to MLP layers
- Train for 20 epochs
- Save adapters per epoch
- Log to Weights & Biases

### Training Configuration

Configure via environment variables:

```bash
# Model and data
MODEL_ID="nvidia/NVIDIA-Nemotron-Nano-9B-v2" \
DATA_JSONL="./data/my_data.json" \
OUTPUT_DIR="./output/my-model" \
bash scripts/train.sh

# LoRA settings
PRESET="mlp_mamba" \
RANK=16 \
ALPHA=32 \
DROPOUT=0.1 \
bash scripts/train.sh

# Training hyperparameters
MAX_EPOCHS=30 \
BATCH_SIZE=2 \
GRAD_ACCUM=4 \
LR=3e-4 \
bash scripts/train.sh

# Checkpoint management
NUM_CHECKPOINT=5 \
bash scripts/train.sh

# Wandb settings
WANDB_PROJECT="my-project" \
WANDB_RUN_NAME="experiment-1" \
bash scripts/train.sh

# Disable wandb
WANDB_OFF=1 bash scripts/train.sh
```

### Available LoRA Presets

| Preset | Target Modules | Use Case |
|--------|---------------|----------|
| `mlp_only` | MLP up/down projections | Fast training, good for task-specific |
| `mamba_only` | Mamba2 in/out projections | Focus on state-space models |
| `mlp_mamba` | MLP + Mamba2 | Balanced, recommended |
| `attn_mlp_ssm` | Attention + MLP + Mamba2 | Best quality, slower |
| `llama` | Standard Llama layers | For testing with other models |

### Multi-GPU Training

```bash
# Use GPUs 0 and 1
bash scripts/train.sh 0,1

# Or via environment
GPU=0,1,2,3 bash scripts/train.sh
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ID` | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | HuggingFace model ID |
| `OUTPUT_DIR` | ./output/nemotron-sft-trained | Output directory |
| `DATA_JSONL` | ./data/example.json | Training data path |
| `PRESET` | mlp_only | LoRA target preset |
| `RANK` | 8 | LoRA rank |
| `ALPHA` | 16 | LoRA alpha |
| `DROPOUT` | 0.05 | LoRA dropout |
| `SEQ_LEN` | 2048 | Max sequence length |
| `BATCH_SIZE` | 1 | Per-device batch size |
| `GRAD_ACCUM` | 8 | Gradient accumulation steps |
| `EPOCHS` | 20 | Number of epochs (legacy) |
| `MAX_EPOCHS` | 20 | Maximum epochs to train |
| `NUM_CHECKPOINT` | -1 | Max checkpoints to keep (-1=all) |
| `LR` | 2e-4 | Learning rate |
| `NUM_WORKERS` | 2 | DataLoader workers |

### Monitoring Training

1. **Weights & Biases Dashboard**: https://wandb.ai (default enabled)
2. **Terminal Output**: Real-time progress and metrics
3. **CSV Logs**: `output/nemotron-sft-trained/lightning_logs/version_X/metrics.csv`

## 🔮 Inference

### Using Bash Script (Recommended)

```bash
# Auto-detect latest epoch
bash scripts/inference.sh

# Use specific epoch
bash scripts/inference.sh 18

# Custom question
bash scripts/inference.sh 18 "What is the difference between Git reset --soft and --hard?"

# With environment variables
QUESTION="Explain Python generators" \
TEMPERATURE=0.5 \
MAX_LENGTH=1024 \
bash scripts/inference.sh 18
```

### Using Python Directly

```bash
python -m src.inferences \
  --adapter ./output/nemotron-sft-trained/epoch-18 \
  --question "Your question here" \
  --max-length 512 \
  --temperature 0.7 \
  --top-p 0.9
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCH` | auto-detect | Epoch number to use |
| `QUESTION` | Git reset question | Question to ask |
| `MAX_LENGTH` | 512 | Max tokens to generate |
| `TEMPERATURE` | 0.7 | Sampling temperature (0-2) |
| `TOP_P` | 0.9 | Top-p/nucleus sampling |
| `USE_4BIT` | true | Enable 4-bit quantization |

### Temperature Guidelines

- **0.1-0.3**: Deterministic, factual responses
- **0.7-0.9**: Balanced creativity and coherence (default)
- **1.0-1.5**: More creative and diverse outputs
- **1.5-2.0**: Highly creative but less coherent

## ⚙️ Configuration

### Edit Configuration Directly

Modify `src/core/config.py` to change defaults:

```python
@dataclass
class ScriptArgs:
    model_id: str
    output_dir: str
    preset: str = "mlp_only"
    rank: int = 8
    alpha: int = 16
    # ... more settings
```

### Custom LoRA Targets

Add custom presets in `src/core/config.py`:

```python
PRESETS = {
    "my_custom": [
        "mixer.up_proj",
        "mixer.down_proj",
        "custom.layer",
    ],
}
```

## 📁 Project Structure

```
mamba_training/
├── README.md                    # This file
├── INFERENCE_GUIDE.md          # Detailed inference documentation
├── data/
│   └── example.json            # Training data
├── output/
│   └── nemotron-sft-trained/   # Training outputs
│       ├── epoch-0/            # Per-epoch adapters
│       ├── epoch-1/
│       └── lightning_logs/     # Training logs
├── scripts/
│   ├── train.sh               # Training script
│   ├── inference.sh           # Inference script
│   └── validation.sh          # Validation script
├── src/
│   ├── __init__.py
│   ├── sft.py                 # Main training script
│   ├── inferences.py          # Inference script
│   ├── orpo.py                # ORPO training (experimental)
│   ├── core/
│   │   ├── config.py          # Configuration dataclasses
│   │   ├── dataset.py         # Dataset loading
│   │   └── model.py           # LightningModule wrapper
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── wandb.py           # Wandb utilities
└── test_inference.py          # Quick inference test
```

### Output Structure

After training, your output directory will look like:

```
output/nemotron-sft-trained/
├── epoch-0/                    # Epoch 0 checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.safetensors  # LoRA weights (~31MB)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── epoch-1/                    # Epoch 1 checkpoint
├── ...
├── epoch-N/                    # Latest checkpoint
└── lightning_logs/
    └── version_X/
        └── metrics.csv         # Training metrics
```

## 🎯 Advanced Usage

### Resume Training from Checkpoint

```bash
# Training will automatically continue if output dir exists
OUTPUT_DIR="./output/my-existing-model" bash scripts/train.sh
```

### Compare Different Epochs

```bash
# Test epoch 10
bash scripts/inference.sh 10 "Test question" > results_epoch10.txt

# Test epoch 20
bash scripts/inference.sh 20 "Test question" > results_epoch20.txt

# Compare results
diff results_epoch10.txt results_epoch20.txt
```

### Batch Inference

Create a script to test multiple questions:

```bash
#!/bin/bash
EPOCH=18
for question in "Question 1" "Question 2" "Question 3"; do
    echo "Testing: $question"
    bash scripts/inference.sh $EPOCH "$question" >> results.txt
    echo "---" >> results.txt
done
```

### Custom Data Format

If your data has a different format, modify `src/core/dataset.py`:

```python
def load_raw_dataset(args):
    # Add your custom loading logic here
    pass
```

### Hyperparameter Search

Use Weights & Biases sweeps:

```bash
# Create sweep configuration
wandb sweep sweep.yaml

# Run sweep agent
wandb agent <sweep-id>
```

## 🐛 Troubleshooting

### CUDA Out of Memory

**Solutions:**
1. Enable 4-bit quantization (default)
2. Reduce batch size: `BATCH_SIZE=1 bash scripts/train.sh`
3. Increase gradient accumulation: `GRAD_ACCUM=16 bash scripts/train.sh`
4. Reduce sequence length: `SEQ_LEN=1024 bash scripts/train.sh`
5. Use gradient checkpointing (enabled by default)

### No Space Left on Device

**Solutions:**
1. Reduce `NUM_CHECKPOINT`: `NUM_CHECKPOINT=3 bash scripts/train.sh`
2. Clean old checkpoints: `rm -rf output/*/epoch-*`
3. The script now only saves adapters (~31MB each) instead of full checkpoints

### Wandb Not Working

**Solutions:**
1. Login: `wandb login`
2. Check API key: `wandb status`
3. Disable if needed: `WANDB_OFF=1 bash scripts/train.sh`

### Model Loading Errors

**Solutions:**
1. Clear HuggingFace cache: `rm -rf ~/.cache/huggingface/`
2. Check internet connection
3. Verify model ID is correct
4. Use `trust_remote_code=True` (already enabled)

### Slow Training

**Optimization tips:**
1. Use multi-GPU: `bash scripts/train.sh 0,1`
2. Increase batch size if VRAM allows
3. Reduce `NUM_WORKERS` if CPU is bottleneck
4. Use smaller LoRA rank: `RANK=4 bash scripts/train.sh`
5. Choose lighter preset: `PRESET=mlp_only bash scripts/train.sh`

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade transformers peft bitsandbytes
pip install --upgrade torch torchvision torchaudio

# Verify installation
python -c "import peft; import bitsandbytes; print('OK')"
```

## 📊 Performance Tips

### Memory Optimization

| Technique | VRAM Saved | Speed Impact |
|-----------|------------|--------------|
| 4-bit quantization | ~75% | Minimal |
| Gradient checkpointing | ~40% | -20% speed |
| Smaller batch size | Variable | Slower |
| Lower LoRA rank | ~10% | Minimal |
| Shorter sequences | Variable | Faster |

### Training Speed

- **Single GPU (RTX 5090)**: ~15 samples/min (batch_size=1, grad_accum=8)
- **Multi-GPU (2x RTX 5090)**: ~25 samples/min
- **Epoch time**: ~1-2 minutes for 15 samples

### Recommended Settings

**Fast experimentation:**
```bash
RANK=4 ALPHA=8 PRESET=mlp_only MAX_EPOCHS=5 bash scripts/train.sh
```

**Production quality:**
```bash
RANK=16 ALPHA=32 PRESET=attn_mlp_ssm MAX_EPOCHS=50 bash scripts/train.sh
```

**Memory constrained:**
```bash
BATCH_SIZE=1 GRAD_ACCUM=16 SEQ_LEN=1024 bash scripts/train.sh
```

## 📚 Additional Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Weights & Biases](https://docs.wandb.ai/)
- [NVIDIA Nemotron Models](https://huggingface.co/nvidia)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the LICENSE file for details.

**Important Note**: AGPL-3.0 is a copyleft license that requires any derivative works or modifications to also be licensed under AGPL-3.0. If you use this software in a network service or web application, you must make the source code available to users.

## 🙏 Acknowledgments

- NVIDIA for the Nemotron models
- HuggingFace for transformers and PEFT
- PyTorch Lightning team
- Weights & Biases for experiment tracking

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to troubleshooting section

## 🔄 Updates

### Latest Changes

- ✅ Added per-epoch checkpoint saving
- ✅ Implemented checkpoint management (NUM_CHECKPOINT)
- ✅ Fixed "No space left on device" error
- ✅ Improved Wandb integration
- ✅ Added comprehensive inference scripts
- ✅ Added MAX_EPOCHS parameter
- ✅ Removed heavy full checkpoints, only save adapters
- ✅ Enhanced error handling and logging

---

**Happy Fine-tuning! 🚀**
