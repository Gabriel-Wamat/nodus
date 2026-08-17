from __future__ import annotations

import hashlib
import platform
import time
from pathlib import Path

import torch
from torchvision.models import resnet18

from nodus.runtime import RuntimeRequest


def load_ppm(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    header, width_height, maximum, pixels = raw.split(b"\n", 3)
    if header != b"P6" or maximum != b"255":
        raise ValueError("Expected a binary P6 PPM image")
    width, height = (int(value) for value in width_height.split())
    tensor = torch.frombuffer(bytearray(pixels), dtype=torch.uint8)
    return tensor.reshape(height, width, 3).permute(2, 0, 1).float().div(255).unsqueeze(0)


def main() -> None:
    request = RuntimeRequest.from_cli()
    checkpoint = request.checkpoint()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    image = load_ppm(request.input("image")).to(device)
    image = (image - 0.5) / 0.25

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    values, indices = logits.softmax(dim=1).topk(5)

    result = {
        "example": "pytorch-vision-resnet18",
        "parameters": request.parameters,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "elapsed_ms": round(elapsed_ms, 3),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "top5_indices": indices[0].cpu().tolist(),
        "top5_probabilities": [round(value, 8) for value in values[0].cpu().tolist()],
    }
    request.write_result(data=result)


if __name__ == "__main__":
    main()
