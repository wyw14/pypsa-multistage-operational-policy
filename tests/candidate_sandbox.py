"""Filesystem isolation, mandatory seccomp and root-owned verifier assets."""

import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import resource
import socket
import stat
import subprocess
import sys
import tempfile

NOBODY = 65534
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _root_only(path):
    path = Path(path)
    if not path.exists():
        return True
    if path.is_symlink() or not path.is_dir():
        return False
    metadata = path.stat()
    return metadata.st_uid == 0 and stat.S_IMODE(metadata.st_mode) & 0o077 == 0


def _landlock(libc, source, scratch, public_inputs):
    """Prepare the filesystem allowlist when the host supports Landlock."""
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 3:
        reason = os.strerror(ctypes.get_errno()) if abi < 0 else "ABI below 3"
        return None, {"enabled": False, "abi": max(int(abi), 0), "reason": reason}
    read_access = (1 << 0) | (1 << 2) | (1 << 3)
    all_access = (1 << 15) - 1
    if abi >= 5:
        all_access |= 1 << 15

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    attr = Ruleset(all_access)
    rules = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if rules < 0:
        return None, {
            "enabled": False,
            "abi": int(abi),
            "reason": "landlock_create_ruleset: " + os.strerror(ctypes.get_errno()),
        }

    def allow(path, rights):
        path = Path(path)
        if not path.exists():
            return
        if not path.is_dir():
            rights &= ~((1 << 3) | sum(1 << i for i in range(4, 14)))
        descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = PathRule(rights, descriptor)
            if libc.syscall(445, rules, 1, ctypes.byref(rule), 0) != 0:
                raise OSError(ctypes.get_errno(), "landlock_add_rule")
        finally:
            os.close(descriptor)

    try:
        for path in (
            "/usr",
            "/lib",
            "/lib64",
            "/bin",
            "/etc/ld.so.cache",
            "/etc/localtime",
        ):
            allow(path, read_access)
        allow(source, read_access)
        allow(public_inputs, read_access)
        allow(scratch, all_access)
        allow("/dev/null", (1 << 1) | (1 << 2))
        allow("/dev/urandom", 1 << 2)
    except OSError as exc:
        os.close(rules)
        return None, {"enabled": False, "abi": int(abi), "reason": str(exc)}
    return rules, {"enabled": True, "abi": int(abi)}


def _seccomp(libc):
    """Block networking, tracing, signalling and privilege escapes."""

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint32),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    denied = [
        41,
        42,
        43,
        49,
        50,
        53,
        62,
        101,
        109,
        112,
        155,
        161,
        165,
        166,
        200,
        234,
        246,
        248,
        249,
        250,
        272,
        297,
        298,
        303,
        304,
        308,
        310,
        311,
        321,
        323,
        424,
        425,
        426,
        427,
        438,
    ]
    instructions = [
        (0x20, 0, 0, 4),
        (0x15, 1, 0, 0xC000003E),
        (0x06, 0, 0, 0x80000000),
        (0x20, 0, 0, 0),
        (0x35, 0, 1, 0x40000000),
        (0x06, 0, 0, 0x80000000),
    ]
    for number in denied:
        instructions.extend([(0x15, 0, 1, number), (0x06, 0, 0, 0x50000 | errno.EPERM)])
    instructions.append((0x06, 0, 0, 0x7FFF0000))
    filters = (Filter * len(instructions))(*(Filter(*x) for x in instructions))
    program = Program(len(instructions), filters)
    if libc.prctl(22, 2, ctypes.byref(program), 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Mandatory PR_SET_SECCOMP failed")


def restrict(source, scratch, public_inputs):
    """Apply isolation before importing candidate code; never run unrestricted."""
    if os.geteuid() != 0 or platform.machine() != "x86_64":
        raise RuntimeError("Verifier requires Linux/amd64 and root bootstrap")
    if Path(source).is_symlink() or str(Path(source).resolve()) != source:
        raise RuntimeError("Candidate source directory must not be a symlink")
    protected = (HERE, Path("/solution"), Path("/logs"))
    dac_ready = all(_root_only(path) for path in protected)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rules, landlock = _landlock(libc, source, scratch, public_inputs)
    filesystem = "landlock"
    identity = NOBODY
    if rules is None:
        from runtime_root import enter_root

        identity = enter_root(source, scratch, public_inputs)
        filesystem = "chroot"
    os.setgroups([])
    os.setgid(identity)
    os.setuid(identity)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
    if rules is not None:
        try:
            if libc.syscall(446, rules, 0) != 0:
                landlock = {
                    "enabled": False,
                    "abi": landlock["abi"],
                    "reason": "landlock_restrict_self: "
                    + os.strerror(ctypes.get_errno()),
                }
                # Identity has already been dropped, so no further privilege
                # transition can safely construct the alternate filesystem.
                raise RuntimeError("Landlock activation failed; candidate was not run")
        finally:
            os.close(rules)
    _seccomp(libc)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    os.chdir(scratch)
    return {
        "uid": os.getuid(),
        "gid": os.getgid(),
        "seccomp": True,
        "no_new_privileges": True,
        "dac_root_only": dac_ready,
        "landlock": landlock,
        "filesystem": filesystem,
    }


def _preflight(configuration):
    policy = restrict(
        configuration["source"], configuration["scratch"], configuration["inputs"]
    )
    probes = []
    for probe in configuration["protected"]:
        path = Path(probe["path"])
        try:
            if probe["kind"] == "directory":
                list(path.iterdir())
            else:
                with path.open("rb") as handle:
                    handle.read(1)
        except (PermissionError, FileNotFoundError) as error:
            probes.append(
                {"path": str(path), "read_denied": True, "denial": type(error).__name__}
            )
        else:
            raise RuntimeError("Candidate can read trusted asset: " + str(path))
    with open(configuration["source"] + "/candidate.py", "rb"):
        pass
    scratch_probe = Path(configuration["scratch"]) / "writable-probe"
    scratch_probe.write_text("probe", encoding="ascii")
    scratch_probe.unlink()
    outside = "/tmp/candidate-must-not-write-outside-scratch"
    try:
        descriptor = os.open(outside, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except PermissionError:
        outside_write_denied = True
    else:
        os.close(descriptor)
        os.unlink(outside)
        raise RuntimeError("Candidate can write outside its private scratch directory")
    try:
        descriptor = os.open(configuration["reward"], os.O_WRONLY)
    except PermissionError:
        reward_denied = True
    else:
        os.close(descriptor)
        raise RuntimeError("Candidate can write the reward file")
    try:
        network = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        network_denied = True
    else:
        network.close()
        raise RuntimeError("Candidate can create a network socket")
    policy.update(
        {
            "asset_probes": probes,
            "reward_write_denied": reward_denied,
            "network_socket_denied": network_denied,
            "outside_scratch_write_denied": outside_write_denied,
        }
    )
    return policy


def prepare_sandbox(tests_dir, logs_dir, private_dir):
    """Protect assets and probe actual open/scandir access as the candidate."""
    if os.geteuid() != 0 or platform.machine() != "x86_64":
        raise RuntimeError("Sandbox preparation requires Linux/amd64 root")
    roots = [
        Path(tests_dir),
        Path("/solution"),
        Path(logs_dir).parent,
        Path(private_dir),
    ]
    protection = []
    probes = []
    for path in roots:
        if not path.exists():
            protection.append({"path": str(path), "exists": False})
            continue
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("Trusted root must be a real directory: " + str(path))
        files = [
            entry
            for entry in path.iterdir()
            if entry.is_file() and not entry.is_symlink()
        ]
        try:
            os.chown(path, 0, 0)
            path.chmod(0o700)
        except OSError as exc:
            protection.append({"path": str(path), "dac": False, "reason": str(exc)})
        else:
            protection.append({"path": str(path), "dac": _root_only(path)})
        probes.append({"path": str(path), "kind": "directory"})
        if files:
            probes.append({"path": str(files[0]), "kind": "file"})
    reward = Path(logs_dir) / "reward.txt"
    if not reward.is_file() or reward.is_symlink():
        raise RuntimeError("A regular initial reward file is required")
    probes.append({"path": str(reward), "kind": "file"})
    folder = Path(tempfile.mkdtemp(prefix="sandbox-preflight-"))
    folder.chmod(0o755)
    scratch = folder / "scratch"
    scratch.mkdir(mode=0o700)
    os.chown(scratch, NOBODY, NOBODY)
    inputs = folder / "inputs"
    inputs.mkdir(mode=0o755)
    # Stage source only when a separate filesystem is needed. Dependency
    # libraries were provisioned at build, outside the verifier deadline.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rules, _ = _landlock(libc, "/app/src", str(scratch), str(inputs))
    if rules is None:
        from runtime_root import prepare_root

        prepare_root("/app/src")
    else:
        os.close(rules)
    configuration = {
        "source": "/app/src",
        "scratch": str(scratch),
        "inputs": str(inputs),
        "protected": probes,
        "reward": str(reward),
    }
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "HOME": str(scratch),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(HERE / "candidate_sandbox.py"),
            "--preflight",
            json.dumps(configuration),
        ],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Sandbox preflight failed: " + completed.stderr.strip()[-3000:]
        )
    policy = json.loads(completed.stdout)
    policy["trusted_asset_permissions"] = protection
    return policy


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--preflight":
        raise SystemExit("Use --preflight with the trusted configuration")
    print(json.dumps(_preflight(json.loads(sys.argv[2]))))
