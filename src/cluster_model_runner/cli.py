from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import ClusterClient
from .exceptions import ClusterRunnerError
from .models import JobRequest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nodus")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inspect", help="Discover nodes, GPUs and Python modules")
    discover = commands.add_parser("discover", help="Discover and persist cluster inventory")
    discover.add_argument("--full", action="store_true", help="Probe every GPU node")
    discover.add_argument("--refresh", action="store_true", help="Ignore cached inventory")
    discover.add_argument("--show", action="store_true", help="Show the persisted inventory")
    commands.add_parser("jobs", help="List locally persisted jobs")
    submit = commands.add_parser("submit", help="Submit a job described by a JSON manifest")
    submit.add_argument("manifest")
    submit.add_argument("--wait", action="store_true")
    submit.add_argument("--download")
    status = commands.add_parser("status", help="Show a job state")
    status.add_argument("job_id")
    logs = commands.add_parser("logs", help="Read remote SLURM logs")
    logs.add_argument("job_id")
    logs.add_argument("--lines", type=int, default=200)
    download = commands.add_parser("download", help="Download job outputs")
    download.add_argument("job_id")
    download.add_argument("--to", default="results")
    cancel = commands.add_parser("cancel", help="Cancel a submitted job")
    cancel.add_argument("job_id")
    env = commands.add_parser("env-create", help="Create or reuse a remote venv")
    env.add_argument("requirements")
    env.add_argument("--name", default="runtime")
    env.add_argument("--python-module", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        client = ClusterClient.from_env()
        if args.command == "inspect":
            print(json.dumps(client.inspect_cluster(), indent=2, default=list))
        elif args.command == "discover":
            policy = "all-nodes" if args.full else None
            result: object
            if args.full or args.refresh or args.show:
                result = client.bootstrap(probe_policy=policy, refresh=args.refresh)
            else:
                result = client.discover()
            print(json.dumps(result, indent=2, default=list))
        elif args.command == "jobs":
            print(json.dumps([job.record for job in client.list_jobs()], indent=2))
        elif args.command == "submit":
            payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            job = client.submit(JobRequest.from_mapping(payload))
            print(json.dumps({"id": job.id, "slurm_id": job.slurm_id}, indent=2))
            if args.wait or args.download:
                job.wait()
            if args.download:
                print(job.download(Path(args.download)))
        elif args.command == "status":
            print(client.status(args.job_id).value)
        elif args.command == "logs":
            print(client.logs(args.job_id, lines=args.lines))
        elif args.command == "download":
            print(client.download(args.job_id, Path(args.to)))
        elif args.command == "cancel":
            client.cancel(args.job_id)
            print("CANCELLED")
        elif args.command == "env-create":
            handle = client.prepare_environment(
                args.requirements, name=args.name, python_module=args.python_module
            )
            print(
                json.dumps(
                    {
                        "environment_id": handle.environment_id,
                        "path": handle.path,
                        "slurm_id": handle.slurm_id,
                        "reused": handle.reused,
                    },
                    indent=2,
                )
            )
        return 0
    except (ClusterRunnerError, OSError, ValueError, TypeError) as exc:
        print(f"nodus: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
