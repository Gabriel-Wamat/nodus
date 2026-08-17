# Transformers LLM example

Runs the small pretrained `sshleifer/tiny-gpt2` causal language model on a GPU. The complete
Hugging Face model snapshot is downloaded locally and passed as a directory checkpoint, so
Nodus demonstrates content-addressed caching for multi-file model weights too.

```bash
python prepare_assets.py
export NODUS_DEMO_VENV="$HOME/envs/gpu-api"
python run.py
```
