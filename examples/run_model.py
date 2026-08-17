from pathlib import Path

from nodus import ClusterClient, JobRequest, ResourceRequest

client = ClusterClient.from_env()

job = client.submit(
    JobRequest(
        name="example-inference",
        project_dir=Path(__file__).parent,
        inputs=[Path("image.png")],
        checkpoint=Path("model.safetensors"),
        requirements=Path("requirements.lock"),
        command=["python", "inference.py"],
        parameters={"prompt": "segment the building"},
        resources=ResourceRequest(min_vram_gb=20, gpu_count=1),
    )
)

result_dir = job.wait().download("results")
print(result_dir)
