"""Untrusted execution side. It never loads pytest, tests or expected results."""

import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wire_codec import pack, unpack
from candidate_sandbox import restrict
from native_state import native_policy_probe

MAX_WIRE = 16 * 1024 * 1024


# Capture real built-in families before importing candidate modules.
# Names are diagnostic only; membership is determined by actual inheritance.
_EXCEPTION_TYPE = type
_EX_IS_SUBCLASS = issubclass
_EX_ENUMERATE = enumerate
_EX_TEXT = str
_EXCEPTION_FAMILIES = (
    (NotImplementedError, "NotImplementedError"),
    (ValueError, "ValueError"),
    (TypeError, "TypeError"),
    (RuntimeError, "RuntimeError"),
)


def exception_response(exc):
    """Encode every supported built-in membership, including multiple bases."""
    actual_type = _EXCEPTION_TYPE(exc)
    family = None
    mask = 0
    for index, (base, name) in _EX_ENUMERATE(_EXCEPTION_FAMILIES):
        if _EX_IS_SUBCLASS(actual_type, base):
            mask |= 1 << index
            if family is None:
                family = name
    return {
        "ok": False,
        "exception": family,
        "exception_mask": mask,
        "exception_type": actual_type.__name__[:200],
        "message": _EX_TEXT(exc)[:2000],
    }


def main():
    """Apply mandatory isolation before loading or executing candidate code."""
    source, scratch, inputs, mode = sys.argv[1:]
    response = os.fdopen(os.dup(1), "w", buffering=1)
    # Candidate print() / os.write(1, ...) cannot become protocol messages.
    os.dup2(2, 1)
    policy = restrict(source, scratch, inputs)
    if mode == "cli":
        raw = sys.stdin.buffer.read(MAX_WIRE + 1)
        if len(raw) > MAX_WIRE:
            raise ValueError("CLI input too large")
        # exec preserves the filesystem and syscall policies and drops bootstrap.
        os.dup2(response.fileno(), 1)
        # Parent feeds the actual CLI input directly in this mode; no wrapper JSON.
        input_path = Path(scratch) / "stdin.json"
        input_path.write_bytes(raw)
        fd = os.open(input_path, os.O_RDONLY)
        os.dup2(fd, 0)
        os.close(fd)
        bootstrap = (
            "import runpy,sys;sys.path.insert(0,sys.argv[1]);"
            'runpy.run_path(sys.argv[1]+"/candidate.py",run_name="__main__")'
        )
        os.execv(sys.executable, [sys.executable, "-I", "-c", bootstrap, source])
    sys.path.insert(0, source)
    spec = importlib.util.spec_from_file_location(
        "candidate", Path(source) / "candidate.py"
    )
    candidate = importlib.util.module_from_spec(spec)
    sys.modules["candidate"] = candidate
    spec.loader.exec_module(candidate)
    response.write(
        json.dumps(
            {
                "ready": True,
                "landlock_abi": policy["landlock"]["abi"],
                "sandbox_policy": policy,
            }
        )
        + "\n"
    )
    for line in sys.stdin:
        if len(line) > MAX_WIRE:
            raise ValueError("RPC input too large")
        request = json.loads(line)
        args, kwargs = unpack(request["args"]), unpack(request["kwargs"])
        try:
            if request["function"] == "__native_policy_probe__":
                value = native_policy_probe(*args, **kwargs)
            else:
                value = getattr(candidate, request["function"])(*args, **kwargs)
        except Exception as exc:
            message = exception_response(exc)
        else:
            # Encoding failures are transport errors, not candidate exceptions.
            try:
                message = {"ok": True, "value": pack(value)}
            except Exception:
                message = {
                    "ok": False,
                    "exception": None,
                    "exception_mask": 0,
                    "exception_type": "TransportEncodingError",
                    "message": "Candidate returned an unsupported RPC value",
                }
        message["args_after"] = pack(args)
        message["kwargs_after"] = pack(kwargs)
        response.write(
            json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n"
        )


if __name__ == "__main__":
    main()
