from cluster_model_runner.models import JobState, ResolvedResources
from cluster_model_runner.slurm import SlurmBackend, render_sbatch
from cluster_model_runner.transport import CommandResult


def test_untyped_gres_uses_eligible_node_set():
    resources = ResolvedResources(
        partition="gpu-batch",
        qos="standard",
        gpu_count=1,
        gpu_type="gpu-large",
        cpus=8,
        ram_gb=64,
        time_limit="01:00:00",
        eligible_nodes=("gpu-a", "gpu-b"),
        excluded_nodes=("gpu-c", "gpu-d"),
        typed_gres=False,
    )
    script = render_sbatch(
        job_name="test",
        resources=resources,
        remote_dir="/remote/job",
        project_dir="/remote/project",
        command=["python", "main.py", "--prompt", "hello world"],
        venv="/remote/venv",
        python_module="Python/3.10.8",
        environment={"SAFE_VALUE": "hello world"},
    )
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --exclude=gpu-c,gpu-d" in script
    assert "#SBATCH --nodelist" not in script
    assert "python main.py --prompt 'hello world'" in script


class FakeStatusTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def run(self, argv, timeout=0):
        self.calls.append(argv)
        return next(self.responses)


def test_status_falls_back_from_squeue_and_sacct_to_scontrol():
    transport = FakeStatusTransport(
        [
            CommandResult(0, "", ""),
            CommandResult(1, "", "slurmdbd unavailable"),
            CommandResult(0, "JobId=42 JobState=COMPLETED ExitCode=0:0", ""),
        ]
    )

    assert SlurmBackend(transport).status("42") == JobState.SUCCEEDED
    assert [call[0] for call in transport.calls] == ["squeue", "sacct", "scontrol"]


def test_correlation_recovery_uses_sacct_when_job_left_queue():
    marker = "a" * 32
    transport = FakeStatusTransport(
        [
            CommandResult(0, "", ""),
            CommandResult(0, f"9001|nodus:{marker}|", ""),
        ]
    )

    assert SlurmBackend(transport).find_by_correlation_id(marker) == "9001"
    assert transport.calls[1][0] == "sacct"


def test_pending_status_includes_scheduler_reason_and_elapsed_time():
    transport = FakeStatusTransport([CommandResult(0, "PENDING|(Resources)|00:02:15|(null)", "")])

    info = SlurmBackend(transport).status_info("77")

    assert info.state == JobState.PENDING
    assert info.reason == "(Resources)"
    assert info.elapsed == "00:02:15"
