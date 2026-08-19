# Evidence matrix

This matrix maps each repository claim to reviewable code and cluster-independent tests.

| Requirement | Evidence |
| --- | --- |
| Importable, typed SDK | Canonical `nodus` plus compatible `cluster_model_runner` namespaces, high-level model contracts, `py.typed` in both packages, strict mypy configuration, and wheel inspection in CI. |
| CLI on the same core | `cluster_model_runner.cli` constructs `JobRequest` and calls `ClusterClient`; `test_public_api_cli.py` guards that contract. |
| SHA-256 code and checkpoint cache | `ArtifactCache` plus file/directory hashing and second-upload cache-hit tests. |
| Restart-resumable jobs | SQLite `JobStore`, SLURM correlation comments, lookup through queue/accounting, remote status markers, and `test_recovery.py`. |
| Real `sinfo`, `scontrol`, and `sacct` discovery | `SlurmDiscovery` and `SlurmBackend` invoke the native commands; fixtures cover queue, accounting and control fallbacks. |
| Smallest compatible GPU | `ResourceSelector` tests generic 24 GiB versus 80 GiB classes, safe capacity, exact selection and unknown-VRAM rejection. |
| Immutable venvs by dependency hash | Environment ID is full SHA-256 over requirements plus Python identity, builders are locked, and only `_READY` environments are reusable. Offline tests prove creation and reuse without admin access. |
| Atomic upload and concurrency protection | Per-digest remote locks, stale-lock recovery, temporary paths, atomic rename, `_READY`, cleanup, and concurrent-waiter tests. |
| Tests without cluster access | Fake transports exercise discovery, scheduling, caching, state recovery, high-level model submission, runtime results, download, CLI, and failures. CI runs the suite on Python 3.10 and 3.12 without SSH credentials. |
| Two real examples | `examples/pytorch_vision` uses official ResNet-18 weights; `examples/transformers_llm` uses a multifile Hugging Face model. |
| Decisions and limitations | ADRs under `docs/decisions`, `docs/architecture.md`, and `docs/limitations.md`. |
| Cluster-agnostic bootstrap | `ClusterBootstrapper`, topology fingerprints, persisted inventories, configurable probes, representative grouping, progress reporting, and offline probe/cache tests. |
| Persistent load-once sessions | Layered `sessions` package, standard-library remote worker, SQLite recovery, authenticated SSH plus `ProxyJump`, atomic filesystem fallback, request deduplication, and tests proving one load across repeated calls. |
| Repository hygiene | Generic `.env.example`, fictional inventory profile, sanitized manifests, injectable contracts, and tests rejecting cluster identifiers in the core. |

Local verification command:

```bash
ruff check src tests examples
mypy src/cluster_model_runner src/nodus
pytest -q
uv build
```

The current verified result is 55 passing offline tests, clean Ruff, clean strict mypy, and a
wheel containing both `nodus/py.typed` and `cluster_model_runner/py.typed`.
