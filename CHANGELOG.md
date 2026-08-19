# Changelog

## Unreleased

- Added persistent SLURM model sessions with one-time model loading, resumable SQLite state,
  atomic shared-filesystem requests, authenticated SSH tunneling, automatic fallback, progress
  reporting, result download, logs, and explicit shutdown.
- Added the typed `SessionContext` and `SessionRequest` remote worker contract.

## 0.1.0

- Added the typed `ClusterClient`, SLURM batch backend, SQLite job persistence, remote venvs,
  content-addressed project and checkpoint caches, and dynamic GPU selection.
- Added high-level `Project`, `Checkpoint`, `Venv`, and `Model` contracts.
- Added named inputs and the remote `RuntimeRequest` result contract.
- Added resumable job handles, discovery TTL, CLI commands, and offline simulated integration
  tests.
- Added scheduler-independent transport, discovery and backend contracts with dependency injection.
- Added persistent cluster fingerprinting and optional representative/full GPU probes without a
  built-in GPU model table.
- Removed cluster-specific endpoint, partition, QoS, node and GPU assumptions from the core.
- Sanitized remote manifests so workstation paths and environment values are not disclosed.
