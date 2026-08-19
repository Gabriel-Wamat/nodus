# Architecture

Nodus is a local control-plane SDK. It does not run a daemon on a login node and does not
require containers. The local process owns orchestration; SLURM owns placement and lifecycle.

```text
Python SDK / CLI
  -> OpenSSH discovery (sinfo + scontrol)
  -> optional SLURM GPU probes + fingerprinted inventory
  -> content-addressed rsync upload
  -> sbatch submission with correlation comment
  -> squeue / sacct / scontrol / remote marker polling
  -> rsync result download

Persistent session
  -> one long-lived SLURM allocation
  -> one model load inside the compute node
  -> authenticated SSH/HTTP control channel when routable
  -> atomic shared-filesystem queue as fallback
  -> repeated requests without reloading weights
```

## Boundaries

- `ClusterClient` is the public facade used by both SDK consumers and the CLI.
- `RemoteTransport` defines remote command and transfer capabilities; `OpenSSHTransport` is the
  default adapter.
- `ClusterDiscovery` defines inventory discovery; `SlurmDiscovery` is the default adapter.
- `SchedulerBackend` owns script rendering and lifecycle operations; `SlurmBackend` implements it.
- `ResourceSelector` converts minimum requirements into a SLURM-compatible placement.
- `ClusterBootstrapper` persists discovered topology and characterizes unknown GPUs on demand.
- `ArtifactCache` owns immutable code and checkpoint objects.
- `EnvironmentManager` owns immutable venvs.
- `JobStore` persists local-to-SLURM correlation in SQLite.
- `SessionService` owns persistent-worker orchestration without expanding the facade.
- `SessionStore` persists local-to-SLURM session correlation and reconnect metadata.
- `RequestChannel` separates transport policy from session lifecycle. The filesystem and SSH
  adapters implement the same request contract.
- `session_runtime` is a standard-library-only worker uploaded into the session directory. It
  loads the project entrypoint once and serializes inference calls against the in-memory model.
- Model projects own their inference code and dependencies.

`ClusterClient` accepts injected transport, discovery, scheduler and state-store implementations.
All adjustable cluster-specific hardware limits live in configuration or an inventory profile.
The orchestration core contains no cluster endpoint, node name, partition, QoS or GPU-model table.
