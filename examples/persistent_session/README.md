# Persistent session

This example demonstrates the session contract without assuming a model framework. Replace the
dictionary in `load_model` with the framework model and keep all weight loading in that function.
`infer` receives the same in-memory object for every request.

From this directory, after configuring the generic `CLUSTER_*` variables:

```bash
python run.py
```

The first printed result contains `calls: 1`; the second contains `calls: 2`. The SLURM job ID
and model process stay the same until `session.close()`.
