from __future__ import annotations

from nodus.session_runtime import SessionContext, SessionRequest


def load_model(context: SessionContext) -> dict[str, object]:
    """Replace this object with a PyTorch, Transformers, Diffusers, or vLLM model."""
    return {"checkpoint": str(context.checkpoint_path or ""), "calls": 0}


def infer(model: dict[str, object], request: SessionRequest) -> dict[str, object]:
    model["calls"] = int(model["calls"]) + 1
    return {
        "calls": model["calls"],
        "prompt": request.parameters.get("prompt", ""),
        "checkpoint": model["checkpoint"],
    }
