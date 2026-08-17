# ADR 0001: Local control plane over OpenSSH

Status: accepted

Nodus uses the user's existing OpenSSH configuration and local `rsync`. It submits batch jobs
instead of installing a persistent API on a cluster login node.

This keeps credentials outside the package, works with SSH keys and agents, respects HPC login
node constraints, and makes the library cluster-agnostic. The tradeoff is that the local SDK
must poll and persist state across restarts.
