from __future__ import annotations

import os
from pathlib import Path

from nodus import Checkpoint, ClusterClient, Project, ResourceRequest, Venv

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".demo-cache"


def main() -> None:
    venv = os.environ.get("NODUS_DEMO_VENV", "")
    if not venv:
        raise RuntimeError("Set NODUS_DEMO_VENV to a remote venv with torch and torchvision")
    client = ClusterClient.from_env()
    model = client.model(
        name="nodus-pytorch-vision",
        project=Project(HERE, "inference.py"),
        environment=Venv(path=venv),
        checkpoint=Checkpoint(CACHE / "resnet18-f37072fd.pth"),
        resources=ResourceRequest(
            min_vram_gb=4,
            gpu_count=1,
            cpus=4,
            ram_gb=16,
            time_limit="00:10:00",
        ),
    )
    job = model.submit(
        inputs={"image": CACHE / "gradient.ppm"},
        parameters={"model": "resnet18"},
    )
    print(job.id)
    print(job.wait(timeout=1800).download(HERE / "results"))


if __name__ == "__main__":
    main()
