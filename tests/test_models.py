from pathlib import Path

import pytest

from cluster_model_runner.models import JobRequest


def test_job_request_from_mapping_builds_typed_contract(tmp_path):
    payload = {
        "name": "demo",
        "command": ["python", "main.py"],
        "project_dir": str(tmp_path),
        "parameters": {"prompt": "hello"},
        "resources": {"min_vram_gb": 24, "ram_gb": 64},
    }
    request = JobRequest.from_mapping(payload)
    assert request.project_dir == Path(tmp_path)
    assert request.resources.min_vram_gb == 24
    assert request.parameters["prompt"] == "hello"


def test_job_request_rejects_shell_string_command():
    with pytest.raises(ValueError, match="list of strings"):
        JobRequest.from_mapping({"name": "bad", "command": "python main.py"})
