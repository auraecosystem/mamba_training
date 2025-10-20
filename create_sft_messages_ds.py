import json
import argparse
from pathlib import Path
from datasets import load_dataset

EXAMPLE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain what Mamba2 is in simple terms."},
    {"role": "assistant", "content": "Mamba2 is a sequence model that uses state-space layers to capture long-range dependencies efficiently, offering faster inference than traditional Transformers in some settings."},
]

EXAMPLES = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Viết một đoạn mô tả ngắn về Nemotron-H."},
        {"role": "assistant", "content": "Nemotron-H là mô hình kết hợp giữa Mamba2 và Transformer, giúp xử lý ngữ cảnh dài hiệu quả, đồng thời giữ được chất lượng sinh văn bản tốt."},
    ],
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me a Python one-liner to reverse a string."},
        {"role": "assistant", "content": "s[::-1]"},
    ],
    [
        {"role": "system", "content": "Bạn là trợ lý AI hữu ích."},
        {"role": "user", "content": "Thủ đô của Pháp là gì?"},
        {"role": "assistant", "content": "Paris."},
    ],
]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/vi_gsm8k.jsonl")
    ap.add_argument("--dataset", default="hllj/vi_gsm8k")
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        # Write a few fixed examples first
        if args.dataset:
            ds = load_dataset(args.dataset, split="train")
            for ex in ds:
                system_content = "/no_think"
                user_content = ex["question"]
                assistant_content = ex["explanation"]
                messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": "<think></think>" + assistant_content},
                ]
                rec = {"messages": messages}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            for msgs in EXAMPLES:
                rec = {"messages": msgs}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # Fill remaining with synthetic variants
            for i in range(max(0, args.n - len(EXAMPLES))):
                rec = {"messages": EXAMPLE}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        #print(f"Wrote {args.n} samples to {out}")


if __name__ == "__main__":
    main()