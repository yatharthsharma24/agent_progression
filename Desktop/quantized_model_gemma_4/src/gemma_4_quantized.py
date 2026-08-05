#!/usr/bin/env python3
"""
Quantizing Gemma 4 — worked examples.

Gemma 4 (google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it) is multimodal
(Gemma4ForConditionalGeneration): a text backbone plus a vision tower, an audio
tower on E2B/E4B/12B, and multimodal projectors. The 26B-A4B is Mixture-of-Experts.

Two rules that matter more than the choice of algorithm:
  1. Quantize the LANGUAGE MODEL ONLY. Leave vision/audio towers, projectors,
     embeddings, lm_head, and MoE routers in bf16. They are a small fraction of
     the weights and quantizing them wrecks multimodal quality / expert routing.
  2. Expect LESS than the naive 4x/2x size drop. That's the sign you did (1) right.

Usage:
    python quantize_gemma4.py bnb-4bit
    python quantize_gemma4.py bnb-8bit
    python quantize_gemma4.py fp8
    python quantize_gemma4.py w4a16
    python quantize_gemma4.py autoround
    python quantize_gemma4.py test --model ./gemma-4-12B-it-W4A16
"""

import argparse
import os

MODEL_ID = "google/gemma-4-12B-it"   # swap for E4B / 31B / 26B-A4B as needed

IGNORE = [
    "re:.*lm_head",
    "re:.*embed_tokens",
    "re:.*per_layer_embed.*",      # E2B/E4B Per-Layer Embeddings
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*multi_modal_projector.*",
    "re:.*router.*",               # 26B-A4B MoE router
    "re:.*gate$",
]


# ---------------------------------------------------------------------------
# 1. bitsandbytes — quantize at load time. Simplest path, good for fine-tuning
#    (QLoRA) and single-GPU experiments. Not the fastest for serving.
#    pip install -U transformers accelerate bitsandbytes torch
# ---------------------------------------------------------------------------
def bnb(bits=4):
    import torch
    from transformers import (
        AutoProcessor,
        AutoModelForMultimodalLM,
        BitsAndBytesConfig,
    )

    skip = [
        "lm_head", "embed_tokens", "vision_tower",
        "audio_tower", "multi_modal_projector", "router",
    ]

    if bits == 4:
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",             
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,         
            llm_int8_skip_modules=skip,
        )
        out_dir = "./gemma-4-12B-it-nf4"
    else:
        qcfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,                
            llm_int8_skip_modules=skip,
        )
        out_dir = "./gemma-4-12B-it-int8"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        quantization_config=qcfg,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",   # safer for Gemma's hybrid local/global attention
    )

    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")
    print(f"footprint: {model.get_memory_footprint() / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# 2. llm-compressor — writes a compressed-tensors checkpoint that vLLM loads
#    natively (no --quantization flag needed; it auto-detects).
#    pip install llmcompressor
#
#    2a. FP8 dynamic: no calibration data, ~2x smaller, near-lossless.
#        Needs Hopper/Ada (H100, L40S, RTX 4090) or newer for the fast kernels.
# ---------------------------------------------------------------------------
def fp8():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",   # per-channel weights, per-token dynamic activations
        ignore=IGNORE,
    )

    out_dir = "./gemma-4-12B-it-FP8-Dynamic"
    oneshot(model=model, recipe=recipe, output_dir=out_dir)
    processor.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")


# ---------------------------------------------------------------------------
#    2b. GPTQ W4A16: 4-bit weights, needs calibration data. ~3.5x smaller.
#        This is the one you want for memory-constrained serving.
# ---------------------------------------------------------------------------
def w4a16(num_samples=512, max_seq_len=2048):
    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # --- calibration set -----------------------------------------------------
    # Use data that looks like your real traffic. Generic web text works, but
    # domain data (your own logs, your own docs) recovers noticeably more accuracy.
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft"
    ).shuffle(seed=42).select(range(num_samples))

    def preprocess(example):
        # Apply the chat template so calibration activations match inference.
        text = processor.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return processor.tokenizer(
            text, truncation=True, max_length=max_seq_len, add_special_tokens=False
        )

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    recipe = GPTQModifier(
        targets="Linear",
        scheme="W4A16",                          # int4 weights, bf16 activations
        group_size=128,                          # 128 is the accuracy/size sweet spot
        dampening_frac=0.01,                     # bump to 0.05 if Hessian inversion fails
        sequential_targets=["Gemma4DecoderLayer"],
        ignore=IGNORE,
    )

    out_dir = "./gemma-4-12B-it-W4A16"
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_len,
        num_calibration_samples=num_samples,
        output_dir=out_dir,
    )
    processor.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")
    print("serve with:  vllm serve ./gemma-4-12B-it-W4A16 --max-model-len 32768")


# ---------------------------------------------------------------------------
# 3. Intel AutoRound — the fallback when llm-compressor/AutoAWQ trip over an
#    architecture. RTN mode (iters=0) is fast and was the path that worked on
#    Gemma 4 in the weeks right after launch.
#    pip install auto-round
# ---------------------------------------------------------------------------
def autoround():
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM
    from auto_round import AutoRound

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    ar = AutoRound(
        model,
        tokenizer=processor.tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        iters=0,          # 0 = round-to-nearest, no calibration. Use 200 for full AutoRound.
        layer_config={    # keep the non-text parts in bf16
            n: {"bits": 16}
            for n, _ in model.named_modules()
            if any(k in n for k in
                   ("vision_tower", "audio_tower", "multi_modal_projector",
                    "lm_head", "embed_tokens", "router"))
        },
    )

    out_dir = "./gemma-4-12B-it-W4A16-AutoRound"
    ar.quantize_and_save(out_dir, format="auto_gptq")
    processor.save_pretrained(out_dir)
    print(f"saved -> {out_dir}")


# ---------------------------------------------------------------------------
# 4. Sanity check — always run this. A quantized model that loads is not the
#    same as a quantized model that works.
# ---------------------------------------------------------------------------
def test(path):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForMultimodalLM.from_pretrained(
        path, dtype="auto", device_map="auto"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain KV-cache quantization in three sentences."},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    ).to(model.device)

    n = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs, max_new_tokens=256,
        temperature=1.0, top_p=0.95, top_k=64, do_sample=True,   # Gemma 4 defaults
    )
    print(processor.decode(out[0][n:], skip_special_tokens=True))
    print(f"\nfootprint: {model.get_memory_footprint() / 1e9:.2f} GB")


GGUF_NOTES = r"""
# 5. GGUF for llama.cpp / Ollama / LM Studio (CPU + Apple Silicon)

git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build -j

# bf16 -> GGUF
python convert_hf_to_gguf.py /path/to/gemma-4-12B-it --outfile gemma4-12b-bf16.gguf --outtype bf16

# (recommended) importance matrix from a calibration corpus — a few minutes,
# meaningfully better quality at 4 bits and below
./build/bin/llama-imatrix -m gemma4-12b-bf16.gguf -f calibration.txt -o gemma4.imatrix

# quantize
./build/bin/llama-quantize --imatrix gemma4.imatrix gemma4-12b-bf16.gguf gemma4-12b-Q4_K_M.gguf Q4_K_M

# run
./build/bin/llama-cli -m gemma4-12b-Q4_K_M.gguf -p "hello"
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=[
        "bnb-4bit", "bnb-8bit", "fp8", "w4a16", "autoround", "test", "gguf",
    ])
    p.add_argument("--model", default=MODEL_ID)
    args = p.parse_args()

    MODEL_ID = args.model
    {
        "bnb-4bit":  lambda: bnb(4),
        "bnb-8bit":  lambda: bnb(8),
        "fp8":       fp8,
        "w4a16":     w4a16,
        "autoround": autoround,
        "test":      lambda: test(args.model),
        "gguf":      lambda: print(GGUF_NOTES),
    }[args.method]()