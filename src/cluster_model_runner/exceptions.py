class ClusterRunnerError(RuntimeError):
    """Base error raised by the package."""


class ConfigurationError(ClusterRunnerError):
    """Configuration is incomplete or unsafe."""


class RemoteCommandError(ClusterRunnerError):
    """A command executed through SSH failed."""


class JobNotFoundError(ClusterRunnerError):
    """A local job identifier is unknown."""


class DiscoveryError(ClusterRunnerError):
    """The remote cluster could not be inspected."""


class JobExecutionError(ClusterRunnerError):
    """A submitted job reached a terminal non-success state."""

    def __init__(self, job_id: str, state: str):
        super().__init__(f"Job {job_id} ended as {state}")
        self.job_id = job_id
        self.state = state
