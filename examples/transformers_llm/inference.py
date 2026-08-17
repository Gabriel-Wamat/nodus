from __future__ import annotations

import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from nodus.runtime import RuntimeRequest


def main() -> None:
    request = RuntimeRequest.from_cli()
    checkpoint = request.checkpoint()
    parameters = request.parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).to(device)
    inputs = tokenizer(parameters["prompt"], return_tensors="pt").to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(parameters["max_new_tokens"]),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000

    result = {
        "example": "transformers-tiny-gpt2",
        "prompt": parameters["prompt"],
        "generated_text": tokenizer.decode(generated[0], skip_special_tokens=True),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": int(generated.shape[-1]),
    }
    request.write_result(data=result)


if __name__ == "__main__":
    main()
