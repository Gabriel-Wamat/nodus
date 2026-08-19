from nodus import ClusterClient, ResourceRequest

client = ClusterClient.from_env()
model = client.model(
    name="persistent-example",
    project=".",
    entrypoint="persistent.py",
    resources=ResourceRequest(min_vram_gb=8, time_limit="01:00:00"),
)

session = model.session(channel="auto").wait_ready()
try:
    print(session.infer(parameters={"prompt": "first"}).data)
    print(session.infer(parameters={"prompt": "second"}).data)
finally:
    session.close()
