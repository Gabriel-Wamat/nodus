# ADR 0005: Persistent sessions use interchangeable control channels

## Status

Accepted.

## Context

Batch execution reloads model weights for every SLURM job. Long-running inference needs one GPU
allocation and one model process, while clusters differ in whether login hosts can route to
compute-node ports or whether compute nodes accept SSH through a jump host.

## Decision

A persistent session is one registered long-lived SLURM job. Its project entrypoint exports
`load_model(context)` and `infer(model, request)`. The worker invokes the first function once,
keeps its return value in memory, and serializes calls to the second function.

The application layer depends on a `RequestChannel` contract. The SSH adapter tries a local
forward through the login host, then a direct compute-node forward through OpenSSH `ProxyJump`.
Both use a random per-session bearer token. The filesystem adapter publishes requests and
responses atomically under the configured remote root. `auto` selects SSH when healthy and
falls back to the filesystem without restarting the worker.

Input files always use the existing SSH/rsync transfer layer. Control requests contain only
validated remote bindings and JSON parameters. Request IDs are idempotency keys: a completed
output is returned again rather than executing inference twice after an ambiguous connection
failure.

Session metadata and scheduler correlation are persisted in SQLite. The local capability token
is stored only in that user-owned database, whose permissions are restricted when supported.

## Consequences

- The model is loaded once per worker process, not once per request.
- Restricted clusters retain a no-admin path through the shared filesystem.
- SSH can reduce control latency, but never becomes a correctness dependency.
- One session occupies resources until explicitly closed or terminated by SLURM.
- Inference is serialized in the initial implementation; concurrent batching is a future policy.
