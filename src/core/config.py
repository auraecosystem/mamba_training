from dataclasses import dataclass 
from typing import List, Optional

PRESETS = {
    # MLP only: apply LoRA to MLP up/down projections
    "mlp_only": [
        "mixer.up_proj",
        "mixer.down_proj",
    ],
    # Mamba2 only: apply LoRA to Mamba2 in/out projections
    "mamba_only": [
        "mixer.in_proj",
        "mixer.out_proj",
    ],
    # MLP + Mamba2: Most common choice for Nemotron-H
    "mlp_mamba": [
        "mixer.up_proj",
        "mixer.down_proj",
        "mixer.in_proj",
        "mixer.out_proj",
    ],
    # All mixers: MLP + Mamba2 + Attention (recommended)
    "attn_mlp_ssm": [
        # MLP projections
        "mixer.up_proj",
        "mixer.down_proj",
        # Mamba2/SSM projections
        "mixer.in_proj",
        "mixer.out_proj",
        # Attention projections (only in some layers)
        "mixer.q_proj",
        "mixer.k_proj",
        "mixer.v_proj",
        "mixer.o_proj",
    ],
    # Standard preset for Llama-style models (for testing with other models)
    "llama": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
}


def get_targets(name: str):
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return PRESETS[name]

@dataclass
class ScriptArgs:
    model_id: str
    output_dir: str
    dataset: Optional[str] = None
    data_jsonl: Optional[str] = None
    text_key: str = "text"
    prompt_key: str = "prompt"
    response_key: str = "response"
    messages_key: str = "messages"  # list[{role, content}]
    preset: str = "attn_mlp_ssm"
    train_on_inputs: bool = False  # SFT: False => mask prompt (not used with chat template)
    append_eos: bool = True
    use_chat_template: bool = True  # Use tokenizer.apply_chat_template if available
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    load_in_4bit: bool = False
    bf16: bool = False
    seq_len: int = 4096
    batch_size: int = 1
    grad_accum: int = 16
    grad_checkpoint: bool = False
    epochs: int = 1
    max_epochs: int = 1
    num_checkpoint: int = -1  # -1: keep all, N: keep last N checkpoints
    lr: float = 2e-4
    num_workers: int = 4
    gpu_devices: list = None  # GPU device(s) to use
    wandb_project: str = "nemotron-h"
    wandb_run_name: Optional[str] = None
    wandb_off: bool = False
    
    # New optimization parameters
    grad_clip: float = 1.0  # Gradient clipping threshold
    unembedding_lr: Optional[float] = None  # Learning rate for unembedding layer
    embedding_lr: Optional[float] = None  # Learning rate for embedding layer
    matrix_lr: Optional[float] = None  # Learning rate for matrix parameters
    use_muon: bool = False  # Use Muon optimizer for matrix parameters
    muon_lr: float = 0.02  # Muon learning rate
    muon_momentum: float = 0.95  # Muon momentum
    logits_softcap: float = 15.0  # Logits softcap for numerical stability
    weight_decay: float = 0.0  # Weight decay for AdamW
    warmup_steps: int = 100  # Learning rate warmup steps
    total_steps: int = 10000  # Total training steps for LR scheduling
