# Public API

The canonical import is `nodus`; `cluster_model_runner` remains available for compatibility.

## Model definition

```python
from nodus import Checkpoint, ClusterClient, Project, ResourceRequest, Venv

client = ClusterClient.from_env()
model = client.model(
    name="vision",
    project=Project(root=".", entrypoint="inference.py"),
    environment=Venv(requirements="requirements.lock", python="auto"),
    checkpoint=Checkpoint(path="model.safetensors"),
    resources=ResourceRequest(min_vram_gb=20),
)
```

`Model` is reusable. Each `submit()` call contains only inputs, scalar parameters, and an
optional resource override. It translates to the same `JobRequest` used by the lower-level API.

## Named inputs and runtime

```python
job = model.submit(inputs={"image": "scene.tif"}, parameters={"prompt": "buildings"})
```

Named inputs are staged below `jobs/<id>/inputs/<name>/`. Remote code reads the generated
manifest through `RuntimeRequest`:

```python
from nodus.runtime import RuntimeRequest

request = RuntimeRequest.from_cli()
image = request.input("image")
checkpoint = request.checkpoint("default")
request.write_result(data={"ok": True}, artifacts=["preview.png"])
```

The runtime helper is uploaded with each job, so the model venv does not need Nodus installed.
Results are written as `output/result.json` and optional files below `output/artifacts/`.

## Jobs and discovery

```python
snapshot = client.discover()          # cached according to CLUSTER_DISCOVERY_TTL
snapshot = client.refresh_cluster()   # explicit refresh
inventory = client.bootstrap()        # probes only when needed
inventory = client.bootstrap(probe_policy="all-nodes", refresh=True)

job = client.job("local-uuid")
jobs = client.list_jobs()
state = job.refresh()
text = job.logs()
job.cancel()
path = job.download("results")
```

`job.wait()` prints state changes and a periodic heartbeat by default. Pending output includes
the scheduler reason returned by `squeue`. Set `progress=False` to disable console feedback, or
pass an `on_update(JobStatus)` callback for programmatic reporting.

The original `client.submit(JobRequest(...))` API remains supported.

## Persistent sessions

```python
session = model.session(entrypoint="persistent.py", channel="auto")
session.wait_ready(timeout=900)
result = session.infer(inputs={"image": "scene.tif"}, parameters={"prompt": "roads"})
result.download("results")
session.close()
```

The session entrypoint exports `load_model(SessionContext)` and
`infer(model, SessionRequest)`. The first function is called exactly once per worker process.
`SessionRequest.write_result()` supports structured data and artifacts. A session can be
reattached after a local restart with `client.session(session_id)` and enumerated with
`client.list_sessions()`.

Channels are `auto`, `ssh`, and `filesystem`. `auto` tries a login-host local forward and a
direct compute-node forward through OpenSSH `ProxyJump`, then falls back to the shared-filesystem
queue if cluster routing prevents both. The fallback is visible as the selected channel and does
not reload the model.

## Resource overrides

`ResourceRequest(min_vram_gb=...)` is the caller's explicit workload requirement. Nodus uses
the discovered safe capacity only to filter and rank nodes; it does not replace the requested
value. `gpu_type` with `policy="exact"` provides a strict override.

`ProbePolicy` centralizes probe concurrency, scheduler time limit, wait timeout and the reserve
subtracted from physical VRAM. The generated inventory stores both physical and safe values.
