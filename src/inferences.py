#!/usr/bin/env python3
"""
Inference script for fine-tuned Nemotron model with LoRA adapter
Tests the model's capabilities after training
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Nemotron")
    parser.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-Nano-9B-v2", 
                        help="Base model ID")
    parser.add_argument("--adapter", default="./output/nemotron-sft-trained/epoch-2",
                        help="Path to LoRA adapter (e.g., ./output/nemotron-sft-trained/epoch-2)")
    parser.add_argument("--question", default="Git reset --soft và --hard khác nhau thế nào?",
                        help="User question")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (1.0 = no penalty, >1.0 = reduce repetition)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit quantization (default: True)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to load the model on (e.g., 'cuda', 'cpu', 'auto')")
    return parser.parse_args()

def format_prompt(question: str, tokenizer) -> str:
    """Format question using chat template"""
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": question}
    ]
    return messages

def main():
    args = parse_args()
    
    # Handle no-4bit flag
    if args.no_4bit:
        args.load_in_4bit = False
    
    print("=" * 80)
    print("🚀 Nemotron Fine-tuned Model Inference")
    print("=" * 80)
    print(f"\n📦 Base Model: {args.base_model}")
    print(f"🎯 Adapter: {args.adapter}")
    print(f"❓ Question: {args.question}")
    print(f"\n⚙️  Generation params:")
    print(f"   • Temperature: {args.temperature}")
    print(f"   • Top-p: {args.top_p}")
    print(f"   • Repetition penalty: {args.repetition_penalty}")
    print(f"   • Max length: {args.max_length}")
    print(f"   • 4-bit quantization: {args.load_in_4bit}")
    print("\n" + "=" * 80)
    
    # Check if adapter exists
    if not os.path.exists(args.adapter):
        print(f"\n❌ Error: Adapter path does not exist: {args.adapter}")
        print("\nAvailable epochs:")
        output_base = os.path.dirname(args.adapter)
        if os.path.exists(output_base):
            epochs = [d for d in os.listdir(output_base) if d.startswith('epoch-')]
            for epoch in sorted(epochs):
                print(f"   • {os.path.join(output_base, epoch)}")
        return
    
    # Load tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load base model
    print("\n[2/4] Loading base model...")
    load_kwargs = {"trust_remote_code": True}
    
    if args.load_in_4bit:
        # Use BitsAndBytesConfig for proper 4-bit loading
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = args.device
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = args.device
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **load_kwargs
    )
    print("✓ Base model loaded")
    
    # Load LoRA adapter
    print(f"\n[3/4] Loading LoRA adapter from {args.adapter}...")
    try:
        model = PeftModel.from_pretrained(base_model, args.adapter)
        # Don't force move to cuda:0, let device_map handle it
        print("✓ Adapter loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load adapter: {e}")
        print("Using base model without adapter...")
        model = base_model
    
    model.eval()
    
    # Debug device information
    print(f"\n🔍 Device Information:")
    print(f"   • Model device: {next(model.parameters()).device}")
    print(f"   • Model dtype: {next(model.parameters()).dtype}")
    
    # Prepare input
    print("\n[4/4] Generating response...")
    messages = format_prompt(args.question, tokenizer)
    
    # Try to use chat template if available
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,  # Ensure we get string, not tokens
            add_generation_prompt=True,
        )
        # Ensure it's a string
        if isinstance(prompt_text, list):
            prompt_text = tokenizer.decode(prompt_text)
    except Exception as e:
        # Fallback format
        print(f"Note: Using fallback prompt format (chat template error: {e})")
        prompt_text = f"<|user|>\n{args.question}<|end|>\n<|assistant|>\n"
    
    print("\n" + "-" * 80)
    print("📝 Formatted Prompt:")
    print("-" * 80)
    print(prompt_text)
    print("-" * 80)
    
    # Tokenize
    inputs = tokenizer(prompt_text, return_tensors="pt")
    print('HERE', tokenizer.eos_token)
    
    # Ensure inputs are on the same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"   • Input IDs shape: {inputs['input_ids'].shape}")
    
    print(f"\n🔍 Input Device Information:")
    print(f"   • Target device: {device}")
    print(f"   • Input IDs device: {inputs['input_ids'].device}")
    print(f"   • Attention mask device: {inputs['attention_mask'].device}")
    
    # Generate
    print("\n🤖 Generating response...\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the new generation (after prompt)
    response = generated_text
    # Try to extract just the assistant's response
    if "<|assistant|>" in generated_text:
        parts = generated_text.split("<|assistant|>")
        if len(parts) > 1:
            response = parts[-1].replace("<|end|>", "").strip()
    
    # Display results
    print("=" * 80)
    print("✨ MODEL RESPONSE")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    print("\n✅ Inference completed successfully!")
    
    # Show token statistics
    print(f"\n📊 Statistics:")
    print(f"   • Input tokens: {inputs['input_ids'].shape[1]}")
    print(f"   • Output tokens: {outputs.shape[1]}")
    print(f"   • Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

if __name__ == "__main__":
    main()
