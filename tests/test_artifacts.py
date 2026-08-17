from cluster_model_runner.artifacts import ArtifactCache, content_hash
from cluster_model_runner.transport import CommandResult


class FakeTransport:
    def __init__(self, *, ready=False, acquire=True):
        self.ready = ready
        self.acquire = acquire
        self.rsync_calls = []
        self.upload_calls = []
        self.remote_files = {}

    def run(self, argv, timeout=0):
        if argv[:2] == ["test", "-f"]:
            return CommandResult(0 if self.ready else 1, "", "")
        if argv[:1] == ["cat"] and argv[1] in self.remote_files:
            return CommandResult(0, self.remote_files[argv[1]].decode(), "")
        if argv[:1] == ["cat"] and argv[1].endswith("/metadata.json"):
            for path, content in self.remote_files.items():
                if path.endswith("/metadata.json"):
                    return CommandResult(0, content.decode(), "")
        return CommandResult(0, "", "")

    def checked(self, argv, timeout=0):
        return ""

    def copy_to(self, local, remote, excludes=()):
        self.rsync_calls.append((local, remote, excludes))

    def upload_bytes(self, content, remote):
        self.upload_calls.append((content, remote))
        self.remote_files[remote] = content

    def shell(self, script, data=None, timeout=0):
        if (
            "mkdir" in script
            and "checkpoint-" in script
            and "seq 1" not in script
            and not self.acquire
        ):
            return CommandResult(1, "", "locked")
        if "touch" in script and "_READY" in script:
            self.ready = True
        return CommandResult(0, "", "")


def test_checkpoint_upload_then_reuse(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"real-weights")
    transport = FakeTransport()
    cache = ArtifactCache(transport, "/remote")

    remote, digest, uploaded = cache.ensure_checkpoint(checkpoint)
    assert uploaded is True
    assert digest == content_hash(checkpoint)
    assert remote.endswith("/model.bin")
    assert len(transport.rsync_calls) == 1

    _, same_digest, uploaded_again = cache.ensure_checkpoint(checkpoint)
    assert same_digest == digest
    assert uploaded_again is False
    assert len(transport.rsync_calls) == 1


def test_concurrent_checkpoint_waiter_does_not_upload(tmp_path):
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"weights")
    transport = FakeTransport(acquire=False)
    cache = ArtifactCache(transport, "/remote")
    _, _, uploaded = cache.ensure_checkpoint(checkpoint)
    assert uploaded is False
    assert transport.rsync_calls == []


def test_equal_content_with_different_name_reuses_stored_object_path(tmp_path):
    first = tmp_path / "first.bin"
    second = tmp_path / "renamed.bin"
    first.write_bytes(b"same-content")
    second.write_bytes(b"same-content")
    transport = FakeTransport()
    cache = ArtifactCache(transport, "/remote")

    first_remote, digest, _ = cache.ensure_checkpoint(first)
    second_remote, second_digest, uploaded = cache.ensure_checkpoint(second)

    assert second_digest == digest
    assert uploaded is False
    assert second_remote == first_remote
