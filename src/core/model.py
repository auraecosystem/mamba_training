
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import (
    AutoModelForCausalLM
)
from typing import Optional, Dict, Any

class LitSFT(pl.LightningModule):
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        lr: float,
        grad_clip: float = 1.0,
        unembedding_lr: Optional[float] = None,
        embedding_lr: Optional[float] = None,
        matrix_lr: Optional[float] = None,
        use_muon: bool = False,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        weight_decay: float = 0.0,
        compute_loss_func: Optional[callable] = None
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.grad_clip = grad_clip
        self.unembedding_lr = unembedding_lr or lr * 0.1
        self.embedding_lr = embedding_lr or lr * 2.0
        self.matrix_lr = matrix_lr or lr
        self.use_muon = use_muon
        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.weight_decay = weight_decay
        self.compute_loss_func = compute_loss_func
        
        # Track training metrics
        self.train_losses = []
        self.num_tokens_seen = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss following HuggingFace Trainer's approach exactly.
        
        Args:
            model: The model to compute loss for
            inputs: Dictionary containing input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs along with loss
            
        Returns:
            loss or (loss, outputs) if return_outputs=True
        """
        if self.compute_loss_func is not None:
            # Custom loss function provided (like UnslothSFTTrainer)
            outputs = model(**inputs)
            loss = self.compute_loss_func(model, inputs, outputs)
            return (loss, outputs) if return_outputs else loss
        
        # Pure HuggingFace approach: model handles everything
        outputs = model(**inputs)
        
        # Extract loss from outputs (HuggingFace standard)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        else:
            raise ValueError("Model did not return a loss and no custom loss function provided")
        
        return (loss, outputs) if return_outputs else loss

    def training_step(self, batch, batch_idx):
        # print(f" • Current training batch index: {batch_idx}")
        # Use pure HuggingFace-style compute_loss
        loss = self.compute_loss(self.model, batch, return_outputs=False)
        
        # Count valid tokens for logging
        valid_tokens = (batch['labels'] != -100).sum()
        self.num_tokens_seen += valid_tokens
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("num_tokens", valid_tokens, on_step=True, on_epoch=True)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Apply gradient clipping after backward pass"""
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

    def configure_optimizers(self):
        """Configure multi-optimizer setup similar to nanochat"""
        # Separate parameters by type
        unembedding_params = []
        embedding_params = []
        matrix_params = []
        
        for name, param in self.named_parameters():
            if 'lm_head' in name or 'score' in name:
                unembedding_params.append(param)
            elif 'embed' in name or 'wte' in name or 'wpe' in name:
                embedding_params.append(param)
            else:
                matrix_params.append(param)
        
        optimizers = []
        
        # AdamW optimizer for all parameters
        param_groups = []
        if unembedding_params:
            param_groups.append({
                'params': unembedding_params, 
                'lr': self.unembedding_lr,
                'weight_decay': self.weight_decay
            })
        if embedding_params:
            param_groups.append({
                'params': embedding_params, 
                'lr': self.embedding_lr,
                'weight_decay': self.weight_decay
            })
        if matrix_params:
            param_groups.append({
                'params': matrix_params, 
                'lr': self.matrix_lr,
                'weight_decay': self.weight_decay
            })
        
        adamw_optimizer = torch.optim.AdamW(param_groups)
        optimizers.append(adamw_optimizer)
        
        # Add Muon optimizer for matrix parameters if enabled
        if self.use_muon and matrix_params:
            try:
                from .muon import Muon
                muon_optimizer = Muon(
                    matrix_params, 
                    lr=self.muon_lr, 
                    momentum=self.muon_momentum
                )
                optimizers.append(muon_optimizer)
            except ImportError:
                print("Warning: Muon optimizer not available, using AdamW only")
        
        # Store initial learning rates for scheduling
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        
        # Return single optimizer for PyTorch Lightning compatibility
        # Multi-optimizer support can be added later if needed
        return optimizers[0]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        """Custom optimizer step with learning rate scheduling"""
        # Apply learning rate multiplier (can be customized)
        lr_multiplier = self.get_lr_multiplier(self.global_step)
        
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lr_multiplier
        
        # Call the default optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx)

    def get_lr_multiplier(self, step: int) -> float:
        """Learning rate scheduling - can be customized"""
        # Simple warmup + cosine decay
        warmup_steps = getattr(self, 'warmup_steps', 100)
        total_steps = getattr(self, 'total_steps', 10000)
        
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
