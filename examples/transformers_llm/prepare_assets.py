from __future__ import annotations

import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / ".demo-cache" / "tiny-gpt2"
BASE = "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main"
FILES = (
    "config.json",
    "merges.txt",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        destination = MODEL_DIR / name
        if not destination.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}", destination)
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
