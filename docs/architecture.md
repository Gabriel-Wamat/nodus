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
- Model projects own their inference code and dependencies.

`ClusterClient` accepts injected transport, discovery, scheduler and state-store implementations.
All adjustable cluster-specific hardware limits live in configuration or an inventory profile.
The orchestration core contains no cluster endpoint, node name, partition, QoS or GPU-model table.
