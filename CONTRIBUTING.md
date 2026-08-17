# Contributing

Nodus targets Python 3.10+ and keeps its orchestration core independent of any specific cluster.

## Development setup

```bash
python -m pip install -e ".[dev]"
ruff check src tests examples
mypy src/cluster_model_runner src/nodus
pytest -q
python -m build
```

Tests must run without SSH or a live SLURM cluster. Use fake transport fixtures for scheduler
behavior and keep real-cluster checks optional. Do not commit credentials, workstation paths,
private endpoints, generated inventories, model weights or local state.

Changes to public contracts require type hints, tests and matching documentation.
Cluster-specific behavior belongs in configuration or an inventory adapter, never in the core.
