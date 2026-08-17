# ADR 0003: GPU selection when GRES is untyped

Status: accepted

Some SLURM installations expose `gpu:N` without the GPU model. Nodus combines live `sinfo`
state with an optional hardware inventory. For `smallest-compatible`, it requests one node and
excludes nodes outside the chosen GPU class.

Passing every compatible node through `--nodelist` can be interpreted by SLURM as requiring all
listed nodes. The implementation instead uses `--nodes=1`, `--ntasks=1`, and `--exclude` for
incompatible nodes while leaving final placement to the scheduler.
