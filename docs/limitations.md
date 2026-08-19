# Limitations

- Resource estimation is declarative: callers provide minimum VRAM, CPU and RAM. Automatic
  profiling from arbitrary model code is not reliable and is not claimed.
- Untyped GPU clusters need either an explicit inventory or permission to submit short probe
  jobs. `CLUSTER_AUTO_PROBE=never` disables probes completely.
- Persistent sessions require the project to expose the Nodus `load_model`/`infer` contract.
  Existing framework servers are not automatically adapted to that interface.
- The SSH channel requires the login host to route to the worker port on the allocated compute
  node. When that is forbidden, `auto` uses the shared-filesystem queue with higher control
  latency but identical load-once behavior.
- A session occupies its SLURM allocation until `close()`, cancellation, timeout, preemption, or
  scheduler failure. Users are responsible for choosing an appropriate time limit.
- A remote package index must be reachable to create new venvs with third-party dependencies.
- A password-only SSH setup is intentionally unsupported in environment variables. Use keys,
  `ssh-agent`, or a configured OpenSSH control connection.
- If SLURM accounting is unavailable and a pending job is cancelled outside Nodus, the SDK may
  not observe the terminal state. Cancelling through `JobHandle.cancel()` remains observable.
- Stale cache locks are reclaimed after two hours. Very long artifact uploads should adjust
  this policy in a future configurable lock lease.
- Nodus deliberately performs no global package installation, scheduler configuration, node
  mutation, or privileged command. All remote state is confined to the configured user-owned
  root.
- Representative probing assumes nodes with identical SLURM topology fields have equivalent
  GPUs. Use `probe_policy="all-nodes"` for exact per-node characterization.
