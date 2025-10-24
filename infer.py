from unsloth import FastLanguageModel
import torch
from torchao.quantization import quantize_
from torchao.quantization.qat import QATConfig
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--think", action="store_true", default=True)
parser.add_argument("--epoch", type=int, default=1, help="Epoch of the model that using for inference")
parser.add_argument("--seq_length", type=int, default=4000, help="Sequence length of the model")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")

args = parser.parse_args()

# # ====== LOAD MODEL ======
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"/home/clara/manhhd/mamba_training/output/merged-medqa_reasoning_nemotron_200/epoch-{args.epoch}",
    max_seq_length=args.seq_length,
    dtype=None,
    trust_remote_code=True,
)
print("100% Model and tokenizer loaded.")
# Chuẩn bị model cho inference
# FastLanguageModel.for_inference(model)
# Use the exact same config as QAT (convenient function)

# model.save_pretrained_torchao(
#     model, "tokenizer", 
#     torchao_config = model._torchao_config.base_config,
# )
# print("✓ Model saved in TorchAO format.")
# # Int4 QAT
# from torchao.quantization import Int4WeightOnlyConfig
# model.save_pretrained_torchao(
#     model, "tokenizer",
# torchao_config = Float8DynamicActivationFloat8WeightConfig(granularity = PerRow())
# )

# # Int8 QAT
# from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
# model.save_pretrained_torchao(
#     "model",   # đường dẫn thư mục bạn muốn lưu
#     tokenizer,
#     torchao_config=Float8DynamicActivationFloat8WeightConfig(),
# )


# # Save to TorchAO int4:
# if False:
#     from torchao.quantization import Int4WeightOnlyConfig
#     model.save_pretrained_torchao("model", tokenizer, torchao_config = Int4WeightOnlyConfig())

# if False: # Pushing to HF Hub
#     from torchao.quantization import Int4WeightOnlyConfig
#     model.save_pretrained_torchao(
#         "hf/model", # Change hf to your username!
#         tokenizer,
#         torchao_config = Int4WeightOnlyConfig(),
#         push_to_hub = True,
#         token = "", # Get a token at https://huggingface.co/settings/tokens
#     )

# # Save to TorchAO int8:
# if False:
#     from torchao.quantization import Int8DynamicActivationInt8WeightConfig
#     model.save_pretrained_torchao("model", tokenizer, torchao_config = Int8DynamicActivationInt8WeightConfig(),)

# if False: # Pushing to HF Hub
#     from torchao.quantization import Int8DynamicActivationInt8WeightConfig
#     model.save_pretrained_torchao(
#         "hf/model", # Change hf to your username!
#         tokenizer,
#         torchao_config = Int8DynamicActivationInt8WeightConfig(),
#         push_to_hub = True,
#         token = "", # Get a token at https://huggingface.co/settings/tokens
#     )


# ====== INFERENCE ======
def generate_response(prompt: str, max_tokens: int = 5000):
    """
    Generate response từ model
    
    Args:
        prompt: Input prompt
        max_tokens: Max tokens để generate
    
    Returns:
        Generated text
    """
    # Format prompt theo Nemotron chat template
    if args.think:
        messages = [
            {"role": "system", "content": "/think"},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": prompt}
        ]
    
    # Encode
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

def generate_response_streaming(prompt: str, max_tokens: int = 5000):
    """
    Generate response từ model theo kiểu streaming
    
    Args:
        prompt: Input prompt
        max_tokens: Max tokens để generate
    
    Yields:
        Generated text chunks (từng token hoặc từng từ)
    """
    # Format prompt theo Nemotron chat template
    messages = [
        {"role": "system", "content": "/think"},
        {"role": "user", "content": prompt}
    ]
    
    # Encode
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate với streaming
    with torch.no_grad():
        # Sử dụng TextIteratorStreamer để stream output
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Tạo generation thread
        generation_kwargs = {
            "input_ids": inputs,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Yield từng chunk từ streamer
        for new_text in streamer:
            yield new_text
        
        thread.join()

# ====== TEST ======
if __name__ == "__main__":
    # Ví dụ 1: Y khoa
    #prompt1 = "\nMột bệnh nhân đeo móc bột cổ (Colles cast) sau hai tuần xuất hiện không thể duỗi ngón cái. Nguyên nhân khả dĩ nhất của tình trạng này là gì?\n\n### Trả lời"
    with open("./input.txt", "r") as f:
        prompt1 = f.read()
    print("=== Input 1 ===")
    print(prompt1)
    
    # print("\n=== Output 1 (Normal) ===")
    # print(generate_response(prompt1))
    
    print("\n=== Output 1 (Streaming) ===")
    full_response = ""
    for chunk in generate_response_streaming(prompt1):
        print(chunk, end="", flush=True)  # In từng chunk ngay lập tức
        full_response += chunk
    
    with open("./output.txt", "w") as f:
        f.write(full_response + "\n")
    
    # # Ví dụ 2: Y khoa khác
    # prompt2 = "Hãy giải thích triệu chứng của bệnh tim hạn chế"
    # print("\n=== Input 2 ===")
    # print(prompt2)
    # print("\n=== Output 2 ===")
    # print(generate_response(prompt2))
    
    # Interactive mode (tùy chọn)
    # print("\n=== Interactive Mode ===")
    # while True:
    #     user_input = input("Nhập câu hỏi (hoặc 'quit' để thoát): ")
    #     if user_input.lower() == 'quit':
    #         break
    #     response = generate_response(user_input)
    #     print(f"Response: {response}\n")



