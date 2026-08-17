# ADR 0004: Dynamic cluster bootstrap

Status: accepted

Nodus does not embed a GPU model table or cluster topology in its orchestration core. It first
collects metadata exposed by SLURM, fingerprints that topology and persists a generated
inventory. When safe VRAM is still unknown and the resource policy needs it, Nodus may submit
short, user-owned `nvidia-smi` jobs according to an explicit `ProbePolicy`.

The default `when-needed` policy avoids scheduler usage when static discovery is sufficient.
Representative probing limits job count on homogeneous clusters; full probing is explicit.
Failures remain scoped to affected nodes, whose unknown capacity is excluded by VRAM-dependent
policies.
