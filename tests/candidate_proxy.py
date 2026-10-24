"""Trusted test-side proxy: data crosses the boundary; candidate code never does."""

import atexit
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from wire_codec import pack, unpack
from runtime_observations import record_fault

MAX_WIRE = 16 * 1024 * 1024
HERE = Path(__file__).resolve().parent
PROCESSES = []


class RemoteCandidateError(Exception):
    """Unknown candidate error or invalid protocol, outside accepted families."""


def _build_exception_types():
    """Restore supported memberships without importing candidate-defined classes."""
    families = (NotImplementedError, ValueError, TypeError, RuntimeError)
    result = {0: RemoteCandidateError}
    for mask in range(1, 16):
        if mask & 1 and not mask & 8:
            continue  # NotImplementedError is also a RuntimeError.
        bases = tuple(
            base
            for index, base in enumerate(families)
            if mask & (1 << index) and not (index == 3 and mask & 1)
        )
        result[mask] = (
            bases[0]
            if len(bases) == 1
            else type("RemoteBuiltinFamilyError", bases, {"__module__": __name__})
        )
    return result


_EXCEPTION_TYPES = _build_exception_types()
_EXCEPTION_NAMES = ("NotImplementedError", "ValueError", "TypeError", "RuntimeError")


def exception_from_response(answer):
    """Reject malformed records; diagnostic class names never grant membership."""
    invalid = "Invalid candidate exception protocol"
    if type(answer) is not dict or answer.get("ok") is not False:
        return RemoteCandidateError(invalid)
    mask = answer.get("exception_mask")
    if type(mask) is not int or mask not in _EXCEPTION_TYPES:
        return RemoteCandidateError(invalid)
    expected = next(
        (name for index, name in enumerate(_EXCEPTION_NAMES) if mask & (1 << index)),
        None,
    )
    if "exception" not in answer or answer["exception"] != expected:
        return RemoteCandidateError(invalid)
    label = answer.get("exception_type")
    message = answer.get("message")
    if type(label) is not str or len(label) > 200:
        return RemoteCandidateError(invalid)
    if type(message) is not str or len(message) > 2000:
        return RemoteCandidateError(invalid)
    return _EXCEPTION_TYPES[mask](message)


def kill_all():
    """Stop tracked candidate groups at trusted pytest process exit."""
    for process in PROCESSES:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass


atexit.register(kill_all)


class Candidate:
    """Invoke candidate functions through an isolated, bounded JSON transport."""

    def __init__(self):
        """Allocate separate candidate scratch and public input directories."""
        self.process = None
        self.pending = b""
        self.folder = Path(tempfile.mkdtemp(prefix="candidate-"))
        self.folder.chmod(0o755)
        self.scratch = self.folder / "scratch"
        self.scratch.mkdir(mode=0o700)
        os.chown(self.scratch, 65534, 65534)
        self.inputs = self.folder / "inputs"
        self.inputs.mkdir(mode=0o755)
        self.abi = None
        self.sandbox_policy = None

    def _command(self, mode):
        return [
            sys.executable,
            "-I",
            str(HERE / "candidate_worker.py"),
            "/app/src",
            str(self.scratch),
            str(self.inputs),
            mode,
        ]

    def _env(self):
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HOME": str(self.scratch),
            "TMPDIR": str(self.scratch),
            "MPLCONFIGDIR": str(self.scratch / "mpl"),
            "MPLBACKEND": "Agg",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }

    def _track(self, process):
        PROCESSES.append(process)
        register = os.environ.get("CANDIDATE_PID_LEDGER")
        if register:
            with open(register, "a", encoding="ascii") as handle:
                handle.write(str(process.pid) + "\n")

    def _read(self, deadline):
        while b"\n" not in self.pending:
            wait = deadline - time.monotonic()
            if wait <= 0 or not select.select([self.process.stdout], [], [], wait)[0]:
                record_fault("timeout", "rpc_deadline")
                self.close()
                raise TimeoutError("Candidate deadline exceeded")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                error_tail = ""
                error_path = self.folder / "stderr.log"
                if error_path.is_file():
                    with error_path.open("rb") as error_log:
                        error_log.seek(max(0, error_path.stat().st_size - 3000))
                        error_tail = error_log.read().decode("utf-8", errors="replace")
                record_fault("crash", "rpc_eof")
                self.close()
                raise RemoteCandidateError(
                    "Candidate exited without a complete RPC result: " + error_tail
                )
            self.pending += block
            if len(self.pending) > MAX_WIRE:
                record_fault("protocol", "rpc_output_limit")
                self.close()
                raise RemoteCandidateError("Candidate output exceeds RPC limit")
        line, self.pending = self.pending.split(b"\n", 1)
        try:
            value = json.loads(line)
        except Exception as exc:
            record_fault("protocol", "rpc_json")
            self.close()
            raise RemoteCandidateError("Invalid candidate protocol") from exc
        if not isinstance(value, dict):
            record_fault("protocol", "rpc_json")
            self.close()
            raise RemoteCandidateError("Invalid candidate protocol")
        return value

    def _start(self, deadline):
        if self.process is not None:
            if self.process.poll() is None:
                return
            self.close()
        # Debug output is bounded by RLIMIT_FSIZE, separate from RPC/report data.
        self.stderr = open(self.folder / "stderr.log", "wb")
        try:
            self.process = subprocess.Popen(
                self._command("rpc"),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                env=self._env(),
                cwd="/",
                start_new_session=True,
            )
        except OSError:
            record_fault("bootstrap", "rpc_spawn")
            raise
        self._track(self.process)
        ready = self._read(deadline)
        if ready.get("ready") is not True or type(ready.get("landlock_abi")) is not int:
            record_fault("bootstrap", "sandbox_handshake")
            self.close()
            raise RemoteCandidateError("Candidate sandbox did not initialize")
        self.abi = ready["landlock_abi"]
        self.sandbox_policy = ready.get("sandbox_policy")

    def _publish_files(self, value):
        if isinstance(value, str) and value.startswith("/"):
            original = Path(value)
            if original.is_file():
                if original.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Fixture file too large")
                data = original.read_bytes()
                target = self.inputs / (
                    hashlib.sha256(data).hexdigest() + original.suffix
                )
                if not target.exists():
                    target.write_bytes(data)
                    target.chmod(0o444)
                if (self.sandbox_policy or {}).get("filesystem") == "chroot":
                    from runtime_root import publish_input

                    publish_input(target, self.inputs)
                return str(target)
        if type(value) in (list, tuple):
            return type(value)(self._publish_files(v) for v in value)
        if type(value) is dict:
            return {k: self._publish_files(v) for k, v in value.items()}
        return value

    def call(self, name, *args, timeout=60, **kwargs):
        """Run one bounded request and verify that input objects stay unchanged."""
        deadline = time.monotonic() + timeout
        self._start(deadline)
        args = self._publish_files(args)
        kwargs = self._publish_files(kwargs)
        before = {"args": pack(args), "kwargs": pack(kwargs)}
        request = dict(before, function=name)
        try:
            self.process.stdin.write(
                (json.dumps(request, allow_nan=False) + "\n").encode()
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            record_fault("crash", "rpc_broken_pipe")
            self.close()
            raise RemoteCandidateError(
                "Candidate exited before accepting an RPC request"
            ) from exc
        answer = self._read(deadline)

        def canonical(value):
            return json.dumps(
                value, allow_nan=False, sort_keys=True, separators=(",", ":")
            )

        try:
            changed = canonical(answer.get("args_after")) != canonical(before["args"])
            changed |= canonical(answer.get("kwargs_after")) != canonical(before["kwargs"])
        except Exception as exc:
            record_fault("protocol", "argument_echo")
            raise RemoteCandidateError("Invalid candidate argument echo") from exc
        if changed:
            raise AssertionError("Candidate modified its input arguments")
        if answer.get("ok") is False:
            raise exception_from_response(answer)
        if answer.get("ok") is not True:
            record_fault("protocol", "rpc_response")
            raise RemoteCandidateError("Invalid candidate response")
        try:
            return unpack(answer["value"])
        except Exception as exc:
            record_fault("protocol", "return_encoding")
            raise RemoteCandidateError("Invalid candidate return-value encoding") from exc

    def run(self, payload, timeout=90):
        """Invoke the public JSON task function without trusted imports."""
        return self.call("solve", payload, timeout=timeout)

    def native_policy(self, payload, timeout=90):
        """Request worker observations of the public native call itself."""
        return self.call("__native_policy_probe__", payload, timeout=timeout)

    def __getattr__(self, name):
        """Expose public candidate functions through the transport boundary."""
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *args, **kwargs: self.call(name, *args, **kwargs)

    def close(self):
        """Terminate a worker group and close its transport resources."""
        process, self.process = self.process, None
        self.pending = b""
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            finally:
                for stream in (
                    process.stdin,
                    process.stdout,
                    getattr(self, "stderr", None),
                ):
                    if stream is not None:
                        stream.close()
                if process in PROCESSES:
                    PROCESSES.remove(process)


candidate = Candidate()


def run_cli(input, timeout=20):
    """Run the public JSON command-line entry with bounded time and output."""
    fresh = Candidate()
    # Files, rather than PIPE buffering, also bound CLI output with RLIMIT_FSIZE.
    with (
        open(fresh.folder / "cli.out", "w+b") as out,
        open(fresh.folder / "cli.err", "w+b") as err,
    ):
        try:
            process = subprocess.Popen(
                fresh._command("cli"),
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                env=fresh._env(),
                cwd="/",
                start_new_session=True,
            )
        except OSError:
            record_fault("bootstrap", "cli_spawn")
            raise
        fresh._track(process)
        try:
            process.communicate(input.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            record_fault("timeout", "cli_deadline")
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise TimeoutError("Candidate CLI deadline exceeded") from exc
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(MAX_WIRE + 1), err.read(MAX_WIRE + 1)
        if len(stdout) > MAX_WIRE or len(stderr) > MAX_WIRE:
            record_fault("protocol", "cli_output_limit")
            raise RemoteCandidateError("Candidate CLI output exceeds limit")
        if process.returncode < 0:
            record_fault("crash", "cli_signal")
        return SimpleNamespace(
            returncode=process.returncode,
            stdout=stdout.decode("utf-8"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )
