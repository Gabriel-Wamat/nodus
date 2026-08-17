from __future__ import annotations

import os
from pathlib import Path

from nodus import Checkpoint, ClusterClient, Project, ResourceRequest, Venv

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / ".demo-cache" / "tiny-gpt2"


def main() -> None:
    venv = os.environ.get("NODUS_DEMO_VENV", "")
    if not venv:
        raise RuntimeError("Set NODUS_DEMO_VENV to a remote venv with torch and transformers")
    client = ClusterClient.from_env()
    model = client.model(
        name="nodus-transformers-llm",
        project=Project(HERE, "inference.py"),
        environment=Venv(path=venv),
        checkpoint=Checkpoint(MODEL_DIR),
        resources=ResourceRequest(
            min_vram_gb=4,
            gpu_count=1,
            cpus=4,
            ram_gb=16,
            time_limit="00:10:00",
        ),
    )
    job = model.submit(
        parameters={"prompt": "Nodus sends models to clusters", "max_new_tokens": 12}
    )
    print(job.id)
    print(job.wait(timeout=1800).download(HERE / "results"))


if __name__ == "__main__":
    main()
