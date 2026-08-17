import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["CLUSTER_RUNNER_REQUEST"]).read_text())
checkpoint = os.environ.get("CLUSTER_RUNNER_CHECKPOINT", "")
output = Path(os.environ["CLUSTER_RUNNER_OUTPUT_DIR"])
output.mkdir(parents=True, exist_ok=True)

# Load the model and checkpoint here. This example only proves the contract.
(output / "result.json").write_text(
    json.dumps({"parameters": request["parameters"], "checkpoint": checkpoint}, indent=2)
)
