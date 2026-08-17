from pathlib import Path

from nodus import ClusterClient


def main() -> None:
    client = ClusterClient.from_env()
    requirements = Path(__file__).with_name("requirements.lock")
    first = client.prepare_environment(requirements, name="nodus-smoke")
    print(first.environment_id, first.slurm_id, first.reused)
    first.wait(timeout=1800)
    second = client.prepare_environment(requirements, name="nodus-smoke")
    print(second.environment_id, second.slurm_id, second.reused)


if __name__ == "__main__":
    main()
