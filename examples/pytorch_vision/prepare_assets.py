from __future__ import annotations

import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".demo-cache"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def write_ppm(path: Path, width: int = 224, height: int = 224) -> None:
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend((x * 255 // (width - 1), y * 255 // (height - 1), 128))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def main() -> None:
    CACHE.mkdir(exist_ok=True)
    checkpoint = CACHE / "resnet18-f37072fd.pth"
    if not checkpoint.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, checkpoint)
    write_ppm(CACHE / "gradient.ppm")
    print(checkpoint)


if __name__ == "__main__":
    main()
