from .core.config import ScriptArgs, get_targets, PRESETS
from .core.dataset import load_raw_dataset, build_text_from_messages, collate_sft, SFTDataset
from .core.model import LitSFT

import argparse
import os
from peft import LoraConfig, get_peft_model
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

torch.cuda.empty_cache()
try:
    torch.cuda.set_per_process_memory_fraction(0.95, device=0)
except Exception:
    pass



def parse_args() -> ScriptArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-jsonl", default=None)
    p.add_argument("--text-key", default="text")
    p.add_argument("--prompt-key", default="prompt")
    p.add_argument("--response-key", default="response")
    p.add_argument("--messages-key", default="messages")
    p.add_argument("--preset", default="attn_mlp_ssm")
    p.add_argument("--train-on-inputs", action="store_true")
    p.add_argument("--append-eos", action="store_true")
    p.add_argument("--use-chat-template", action="store_true", default=True,
                   help="Use tokenizer.apply_chat_template for messages format")
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Optional max epochs (overrides --epochs if provided)")
    p.add_argument("--num-checkpoint", type=int, default=-1,
                   help="Max number of checkpoints to keep (-1 for all, N for last N)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpu", type=str, default="0", help="GPU device(s) to use (e.g., '0' or '0,1')")
    p.add_argument("--wandb-project", default="nemotron-h")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping threshold")
    p.add_argument("--unembedding-lr", type=float, default=None,
                   help="Learning rate for unembedding layer (default: lr * 0.1)")
    p.add_argument("--embedding-lr", type=float, default=None,
                   help="Learning rate for embedding layer (default: lr * 2.0)")
    p.add_argument("--matrix-lr", type=float, default=None,
                   help="Learning rate for matrix parameters (default: lr)")
    p.add_argument("--use-muon", action="store_true",
                   help="Use Muon optimizer for matrix parameters")
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="Muon optimizer learning rate")
    p.add_argument("--muon-momentum", type=float, default=0.95,
                   help="Muon optimizer momentum")
    p.add_argument("--logits-softcap", type=float, default=15.0,
                   help="Logits softcap for numerical stability")
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="Weight decay for AdamW")
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Learning rate warmup steps")
    p.add_argument("--total-steps", type=int, default=10000,
                   help="Total training steps for LR scheduling")
    p.add_argument("--wandb-off", action="store_true")
    a = p.parse_args()
    
    # Parse GPU devices
    # if "," in a.gpu:
    #     gpu_devices = [int(x.strip()) for x in a.gpu.split(",")]
    # else:
    #     gpu_devices = [int(a.gpu)]
    # Lấy GPU từ biến môi trường CUDA_VISIBLE_DEVICES nếu có
    import os
    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_devices = [int(x.strip()) for x in gpu_str.split(",") if x.strip()]
    
    return ScriptArgs(
        model_id=a.model_id,
        output_dir=a.output_dir,
        dataset=a.dataset,
        data_jsonl=a.data_jsonl,
        text_key=a.text_key,
        prompt_key=a.prompt_key,
        response_key=a.response_key,
        messages_key=a.messages_key,
        preset=a.preset,
        train_on_inputs=a.train_on_inputs,
        append_eos=a.append_eos,
        use_chat_template=a.use_chat_template,
        rank=a.rank,
        alpha=a.alpha,
        dropout=a.dropout,
        load_in_4bit=a.load_in_4bit,
        bf16=a.bf16,
        seq_len=a.seq_len,
        batch_size=a.batch_size,
        grad_accum=a.grad_accum,
        grad_checkpoint=a.grad_checkpoint,
        epochs=a.epochs,
        lr=a.lr,
        num_workers=a.num_workers,
        gpu_devices=gpu_devices,
        wandb_project=a.wandb_project,
        wandb_run_name=a.wandb_run_name,
        wandb_off=a.wandb_off,
        max_epochs=a.max_epochs if hasattr(a, "max_epochs") and a.max_epochs is not None else a.epochs,
        num_checkpoint=a.num_checkpoint if hasattr(a, "num_checkpoint") else -1,
        # New optimization parameters
        grad_clip=a.grad_clip,
        unembedding_lr=a.unembedding_lr,
        embedding_lr=a.embedding_lr,
        matrix_lr=a.matrix_lr,
        use_muon=a.use_muon,
        muon_lr=a.muon_lr,
        muon_momentum=a.muon_momentum,
        logits_softcap=a.logits_softcap,
        weight_decay=a.weight_decay,
        warmup_steps=a.warmup_steps,
        total_steps=a.total_steps,
    )

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.load_in_4bit:
        load_kwargs.update(dict(load_in_4bit=True))
        if args.bf16 and torch.cuda.is_available():
            load_kwargs.update(dict(bnb_4bit_compute_dtype=torch.bfloat16))
    elif args.bf16:
        load_kwargs.update(dict(torch_dtype=torch.bfloat16))


    model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map=None,
            trust_remote_code=True,
            **load_kwargs,
        )
    
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:
                pass

    # PEFT / LoRA
    target_modules = get_targets(args.preset)
    lconf = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)

    # Data
    raw_ds = load_raw_dataset(args)
    sft_ds = SFTDataset(raw_ds, tokenizer, args)

    def _collate(b):
        return collate_sft(b, tokenizer)
    

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": _collate,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers and args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(sft_ds, **loader_kwargs)

    precision = "bf16" if (args.bf16 and torch.cuda.is_available()) else 32
    lit_model = LitSFT(
        model, 
        lr=args.lr,
        grad_clip=args.grad_clip,
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        use_muon=args.use_muon,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        weight_decay=args.weight_decay
    )
    
    # Set additional parameters for LR scheduling
    lit_model.warmup_steps = args.warmup_steps
    lit_model.total_steps = args.total_steps

    wandb_logger = None
    use_wandb = not getattr(args, "wandb_off", False)
    if use_wandb:
        project = os.getenv("WANDB_PROJECT", args.wandb_project if hasattr(args, "wandb_project") else "nemotron-h")
        run_name = os.getenv("WANDB_RUN_NAME", getattr(args, "wandb_run_name", None))
        print(f"\n🔗 Initializing Wandb Logger...")
        print(f"   Project: {project}")
        print(f"   Run Name: {run_name}")
        try:
            wandb_logger = WandbLogger(project=project, name=run_name, log_model=False)
            print(f"✓ Wandb initialized successfully!")
            print(f"   View run at: https://wandb.ai/{wandb_logger.experiment.entity}/{project}/runs/{wandb_logger.experiment.id}")
            wandb_logger.log_hyperparams({
                "model_id": args.model_id,
                "preset": args.preset,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "epochs": args.epochs,
                "lr": args.lr,
                "rank": args.rank,
                "alpha": args.alpha,
                "dropout": args.dropout,
                "load_in_4bit": args.load_in_4bit,
                "bf16": args.bf16,
                # New optimization parameters
                "grad_clip": args.grad_clip,
                "unembedding_lr": args.unembedding_lr,
                "embedding_lr": args.embedding_lr,
                "matrix_lr": args.matrix_lr,
                "use_muon": args.use_muon,
                "muon_lr": args.muon_lr,
                "muon_momentum": args.muon_momentum,
                "logits_softcap": args.logits_softcap,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "total_steps": args.total_steps,
            })
        except Exception as e:
            print(f"✗ Failed to initialize Wandb: {e}")
            print(f"   Continuing without wandb logging...")
            wandb_logger = None
    else:
        print("\n⚠ Wandb is disabled (--wandb-off)")


    
    # GPU devices from args
    gpu_devices = getattr(args, "gpu_devices", [0])
    
    # Custom callback to save adapter + tokenizer at the end of each epoch
    class EpochEndSaver(Callback):
        """Save adapter + tokenizer at the end of each epoch with epoch index in filename."""
        def __init__(self, output_dir, num_checkpoint=-1):
            self.output_dir = output_dir
            self.num_checkpoint = num_checkpoint
            self.saved_epochs = []
        
        def on_train_epoch_end(self, trainer, pl_module):
            try:
                epoch = trainer.current_epoch
                out_dir = os.path.join(self.output_dir, f"epoch-{epoch}")
                os.makedirs(out_dir, exist_ok=True)
                
                # Save adapters (PEFT) and tokenizer
                try:
                    pl_module.model.save_pretrained(out_dir)
                except Exception:
                    # If model is wrapped, try accessing underlying model
                    try:
                        pl_module.model.module.save_pretrained(out_dir)
                    except Exception:
                        pass
                try:
                    tokenizer.save_pretrained(out_dir)
                except Exception:
                    pass
                
                # Track saved epochs
                self.saved_epochs.append(epoch)
                
                # Remove old checkpoints if num_checkpoint is set
                if self.num_checkpoint > 0 and len(self.saved_epochs) > self.num_checkpoint:
                    epochs_to_remove = self.saved_epochs[:-self.num_checkpoint]
                    for old_epoch in epochs_to_remove:
                        old_dir = os.path.join(self.output_dir, f"epoch-{old_epoch}")
                        if os.path.exists(old_dir):
                            import shutil
                            shutil.rmtree(old_dir)
                            print(f"Removed old checkpoint: {old_dir}")
                    self.saved_epochs = self.saved_epochs[-self.num_checkpoint:]
            except Exception as e:
                print(f"Error saving checkpoint at epoch {epoch}: {e}")

    epoch_saver_cb = EpochEndSaver(args.output_dir, args.num_checkpoint)

    num_devices = len(gpu_devices)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        accumulate_grad_batches=args.grad_accum,
        precision=precision,
        enable_checkpointing=False,
        callbacks=[epoch_saver_cb],
        devices=list(range(num_devices)),
        accelerator="gpu",
        strategy="auto" if num_devices == 1 else "ddp",
        logger=wandb_logger if wandb_logger is not None else None,
    )

    if wandb_logger is not None:
        try:
            wandb_logger.watch(lit_model.model, log="all", log_freq=100)
        except Exception:
            pass
    trainer.fit(lit_model, train_dataloaders=train_loader)

    # Save adapter and tokenizer
    lit_model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
    

        