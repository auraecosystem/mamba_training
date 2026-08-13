from datasets import load_dataset
from .config import ScriptArgs
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
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

def format_reasoning_data(example):
    """
    Format reasoning/chain-of-thought data cho Nemotron.
    Giữ nguyên /think và <think></think> tags.
    """
    messages = example["messages"]
    text = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

    text += "<|eot_id|>"
    return {"text": text}

def build_text_from_messages(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer, use_chat_template: bool = True) -> str:
    """
    Convert messages to text using custom Nemotron format for reasoning data.
    Preserves /think and <think></think> tags.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: HuggingFace tokenizer
        use_chat_template: Whether to use custom format (always True for reasoning)
    
    Returns:
        Formatted text string
    """
    # Use custom Nemotron format for reasoning data
    text = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()

        if role == "system":
            text += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "user":
            text += f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
        elif role == "assistant":
            text += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

    text += "<|eot_id|>"
    return text



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


def get_data_collator(tokenizer: AutoTokenizer, mlm: bool = False):
    """
    Sử dụng trực tiếp DataCollatorForLanguageModeling của Hugging Face
    để đảm bảo logic hoàn toàn chính xác.
    
    Args:
        tokenizer: HuggingFace tokenizer
        mlm: Whether to use masked language modeling (False for causal LM)
    
    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,  # False for causal language modeling (SFT)
        mlm_probability=0.15 if mlm else None,
        pad_to_multiple_of=None,
        return_tensors="pt"
    )


# Deprecated: Giữ lại để tương thích ngược
def collate_sft(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer):
    """
    DEPRECATED: Sử dụng get_data_collator() thay thế.
    
    HuggingFace-style collator for causal language modeling using tokenizer.
    Uses tokenizer.pad() for proper padding and tensor conversion.
    """
    # Sử dụng DataCollatorForLanguageModeling thay thế
    data_collator = get_data_collator(tokenizer, mlm=False)
    return data_collator(batch)
