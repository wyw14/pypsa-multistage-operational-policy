"""Prepare and finalize trusted scoring around a normal pytest shell command.

This module never executes candidate code. It authenticates test identities,
records infrastructure errors, cleans candidate process groups, and publishes
reward only after a complete successful pytest session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
LOGS = Path("/logs/verifier")
EXPECTED_PHASES = [["setup", "passed"], ["call", "passed"], ["teardown", "passed"]]


def atomic_write(path: Path, text: str) -> None:
    """Replace a root-owned result without exposing a partial or stale value."""
    descriptor, name = tempfile.mkstemp(prefix=".verifier-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        temporary.chmod(0o600)
        if path.exists():
            if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
                raise RuntimeError(
                    "Verifier output must be a regular file: " + str(path)
                )
            # Docker Desktop maps mode 0444 to a Windows read-only attribute.
            path.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict) -> None:
    """Publish structured verifier evidence atomically."""
    atomic_write(path, json.dumps(value, indent=2) + "\n")


def publish(record: dict) -> None:
    """Persist diagnostic evidence before granting any successful reward."""
    write_json(LOGS / "grading.json", record)
    atomic_write(LOGS / "reward.txt", str(record["reward"]) + "\n")
    print(json.dumps(record), flush=True)


def initialize() -> None:
    """Reset reward atomically, including a reused read-only regular file."""
    LOGS.mkdir(parents=True, exist_ok=True)
    atomic_write(LOGS / "reward.txt", "0\n")


def contract() -> dict:
    """Authenticate the complete expected test set and trusted implementation."""
    manifest = json.loads((HERE / "test-contract.json").read_text(encoding="utf-8"))
    expected = manifest["nodeids"]
    if not expected or len(expected) != len(set(expected)):
        raise RuntimeError("Expected test identities are empty or duplicated")
    for relative, expected_digest in manifest["test_files_sha256"].items():
        path = (HERE / relative).resolve()
        if not path.is_relative_to(HERE):
            raise RuntimeError("Test contract contains an escaping path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            raise RuntimeError("Trusted test checksum mismatch: " + relative)
    return manifest


def prepare(workspace: Path) -> None:
    """Check isolation and integrity before the shell launches pytest."""
    record = {"verifier_version": "pytest-cli-v2", "reward": 0, "timed_out": False}
    state = {"started_monotonic": time.monotonic(), "record": record}
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        atomic_write(LOGS / "reward.txt", "0\n")
        if os.geteuid() != 0:
            raise RuntimeError(
                "Verifier requires root bootstrap; candidate runs as uid 65534"
            )
        information = workspace.stat()
        if information.st_uid != 0 or stat.S_IMODE(information.st_mode) != 0o700:
            raise RuntimeError(
                "Verifier workspace must be a private root-owned directory"
            )
        manifest = contract()
        record["expected_tests"] = len(manifest["nodeids"])
        record["test_contract_sha256"] = hashlib.sha256(
            (HERE / "test-contract.json").read_bytes()
        ).hexdigest()
        sys.path.insert(0, str(HERE))
        from candidate_sandbox import prepare_sandbox

        record["sandbox"] = prepare_sandbox(str(HERE), str(LOGS), str(workspace))
        state["expected"] = manifest["nodeids"]
        state["prepared"] = True
    except Exception as error:
        state["preparation_error"] = type(error).__name__ + ": " + str(error)
        record["grader_error"] = state["preparation_error"]
        write_json(workspace / "state.json", state)
        publish(record)
        raise
    write_json(workspace / "state.json", state)
    write_json(LOGS / "preflight.json", record)


def cleanup(workspace: Path) -> None:
    """Kill candidate process groups that outlive pytest or a timeout."""
    ledger = workspace / "candidate-pids.txt"
    if ledger.exists():
        for line in ledger.read_text(encoding="ascii").splitlines():
            if line.isdigit():
                try:
                    os.killpg(int(line), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def finalize(workspace: Path, pytest_status: int, stage: str) -> None:
    """Accept only the exact full test collection and all three passing phases."""
    record = {"verifier_version": "pytest-cli-v2", "reward": 0, "timed_out": False}
    started = time.monotonic()
    try:
        state = json.loads((workspace / "state.json").read_text(encoding="utf-8"))
        record.update(state["record"])
        started = state["started_monotonic"]
        record["reward"] = 0
        record["completed_stage"] = stage
        record["pytest_exit_code"] = pytest_status
        record["timed_out"] = pytest_status in (124, 137)
        if state.get("preparation_error"):
            raise RuntimeError(state["preparation_error"])
        if stage != "pytest":
            raise RuntimeError("Pytest did not complete; failed stage: " + stage)
        # Recheck the trusted source after candidate execution as well.
        manifest = contract()
        if manifest["nodeids"] != state["expected"]:
            raise RuntimeError("Expected test identities changed during verification")
        report = workspace / "outcomes.json"
        if not report.is_file() or report.stat().st_size > 10_000_000:
            raise RuntimeError("Trusted pytest report is missing or oversized")
        data = json.loads(report.read_text(encoding="utf-8"))
        write_json(LOGS / "outcomes.json", data)
        expected = state["expected"]
        collected, outcomes = data["collected"], data["outcomes"]
        record["collected_tests"] = len(collected)
        record["completed_tests"] = len(outcomes)
        record["passed_tests"] = sum(
            phases == EXPECTED_PHASES for phases in outcomes.values()
        )
        complete = (
            not data["collection_errors"]
            and sorted(collected) == sorted(expected)
            and set(outcomes) == set(expected)
        )
        record["collection_complete"] = complete
        if (
            complete
            and not record["timed_out"]
            and pytest_status == data["exitstatus"] == 0
            and record["passed_tests"] == len(expected)
        ):
            record["reward"] = 1
    except Exception as error:
        record["reward"] = 0
        record["grader_error"] = type(error).__name__ + ": " + str(error)
    finally:
        try:
            cleanup(workspace)
        except Exception as error:
            record["reward"] = 0
            record["cleanup_error"] = type(error).__name__ + ": " + str(error)
        record["elapsed_seconds"] = time.monotonic() - started
        publish(record)


def main() -> None:
    """Expose separate preparation and finalization commands to test.sh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["initialize", "prepare", "finalize"])
    parser.add_argument("workspace", type=Path, nargs="?")
    parser.add_argument("--pytest-status", type=int, default=125)
    parser.add_argument("--stage", default="prepare")
    arguments = parser.parse_args()
    if arguments.action == "initialize":
        initialize()
    elif arguments.workspace is None:
        parser.error("prepare and finalize require a private workspace")
    elif arguments.action == "prepare":
        prepare(arguments.workspace)
    else:
        finalize(arguments.workspace, arguments.pytest_status, arguments.stage)


if __name__ == "__main__":
    main()
