from datasets import load_dataset
from .config import ScriptArgs
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch 

def load_raw_dataset(args: ScriptArgs):
    if args.dataset:
        ds = load_dataset(args.dataset)
        ds = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        return ds
    if args.data_jsonl:
        ds = load_dataset("json", data_files=args.data_jsonl)["train"]
        return ds
    raise ValueError("Provide --dataset <hf_id> or --data-jsonl <path>")

def build_text_from_messages(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer, use_chat_template: bool = True) -> str:
    """
    Convert messages to text using tokenizer's chat template.
    Falls back to simple format if chat template is not available.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        use_chat_template: Whether to try using apply_chat_template
    
    Returns:
        Formatted text string
    """
    if use_chat_template:
        try:
            # Use HuggingFace's apply_chat_template (preferred method)
            # tokenize=False returns the formatted string, not token IDs
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False,
            ) + tokenizer.eos_token
            return text
        except Exception:
            # Fallback to simple format if chat template is not available
            pass
    
    # Fallback format
    text_parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            text_parts.append(f"<|system|>\n{content}<|end|>")
        elif role == "user":
            text_parts.append(f"<|user|>\n{content}<|end|>")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>\n{content}<|end|>")
    
    full_text = "\n".join(text_parts)
    if tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token
    
    return full_text



class SFTDataset(Dataset):
    """Dataset that converts all formats to text, following HuggingFace approach"""
    def __init__(self, ds, tokenizer: AutoTokenizer, args: ScriptArgs):
        self.ds = ds
        self.tok = tokenizer
        self.args = args
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __len__(self):
        return len(self.ds)

    def _convert_to_text(self, ex: Dict[str, Any]) -> str:
        """Convert any format to plain text"""
        # Prefer messages format
        if self.args.messages_key in ex and isinstance(ex[self.args.messages_key], list):
            return build_text_from_messages(
                ex[self.args.messages_key], 
                self.tok, 
                use_chat_template=self.args.use_chat_template
            )
        
        # Then prompt/response format
        if self.args.prompt_key in ex and self.args.response_key in ex:
            prompt = ex[self.args.prompt_key]
            response = ex[self.args.response_key]
            text = f"{prompt}\n{response}"
            if self.args.append_eos and self.tok.eos_token and not text.endswith(self.tok.eos_token):
                text += self.tok.eos_token
            return text
        
        # Then plain text
        if self.args.text_key in ex:
            text = ex[self.args.text_key]
            if self.args.append_eos and self.tok.eos_token and not text.endswith(self.tok.eos_token):
                text += self.tok.eos_token
            return text
        
        # Fallback: join all string fields
        text = " ".join(str(v) for v in ex.values() if isinstance(v, str))
        if self.args.append_eos and self.tok.eos_token and not text.endswith(self.tok.eos_token):
            text += self.tok.eos_token
        return text

    def __getitem__(self, idx):
        ex = self.ds[idx]
        text = self._convert_to_text(ex)
        
        # Tokenize using HuggingFace standard approach
        # This will be used with DataCollatorForLanguageModeling
        tokenized = self.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.args.seq_len,
            return_tensors=None,  # Return lists, not tensors
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }


def collate_sft(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer):
    """
    HuggingFace-style collator for causal language modeling.
    Creates labels by shifting input_ids, matching the standard CLM approach.
    """
    # Pad to max length in batch
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    
    input_ids = torch.tensor([pad(x["input_ids"], pad_id) for x in batch], dtype=torch.long)
    attention_mask = torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long)
    
    # Create labels following HuggingFace approach:
    # labels = input_ids.clone() where we want to compute loss
    # labels = -100 where we want to ignore (padding positions)
    labels = input_ids.clone()
    
    # Mask padding tokens in labels
    labels[labels == pad_id] = -100
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }