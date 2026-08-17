# PyTorch vision example

Runs a pretrained torchvision ResNet-18 on a deterministic synthetic PPM image. The official
checkpoint is downloaded locally, passed to Nodus, cached by SHA-256 on the cluster, loaded on
the selected GPU, and used for a real forward pass.

```bash
python prepare_assets.py
export NODUS_DEMO_VENV="$HOME/envs/gpu-api"
python run.py
```

The remote process reads the standard Nodus environment variables and writes `result.json`.
