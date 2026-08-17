# ADR 0002: Content-addressed artifacts and environments

Status: accepted

Checkpoints, project snapshots, and venv definitions are identified by SHA-256. A `_READY`
marker is the publication boundary. Locks are acquired before transfer or installation, so a
concurrent caller waits rather than publishing a partial object.

Checkpoint and project publishers use temporary directories followed by rename. Venvs cannot
be safely renamed because scripts may contain absolute paths; they are built in their final
location under a lock and remain unavailable until `_READY` exists.
