"""Dependency-only chroot for hosts without a usable Landlock ruleset."""

import fcntl
import os
from pathlib import Path
import shutil
import stat

ROOT = Path("/opt/candidate-root")


def _worker_identity():
    """Allocate a non-reused UID so workers cannot write each other's scratch."""
    state = Path("/opt/candidate-worker-state")
    state.mkdir(mode=0o700, exist_ok=True)
    if state.is_symlink() or state.stat().st_uid != 0 or state.stat().st_mode & 0o077:
        raise RuntimeError("Unsafe worker identity state")
    descriptor = os.open(state / "next-uid", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = handle.read()
        uid = int(value) if value else 10000
        if not 10000 <= uid < 60000:
            raise RuntimeError("Candidate identity range exhausted")
        handle.seek(0)
        handle.write(str(uid + 1))
        handle.truncate()
        handle.flush()
    return uid


def prepare_root(source):
    """Stage only public candidate code; never copy trusted test material."""
    metadata = ROOT.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0:
        raise RuntimeError("Missing trusted dependency filesystem")
    if metadata.st_mode & 0o022:
        raise RuntimeError("Dependency filesystem is writable by candidate")
    source = Path(source)
    target = ROOT / "app/src"
    if target.is_symlink():
        raise RuntimeError("Candidate filesystem must not be a link")
    if target.exists():
        shutil.rmtree(target)
    # Validate special files without following candidate-controlled symlinks.
    for parent, dirs, files in os.walk(source, followlinks=False):
        for name in dirs + files:
            mode = (Path(parent) / name).lstat().st_mode
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                raise RuntimeError("Unsupported special file in candidate source")
    shutil.copytree(source, target, symlinks=True)
    for parent, dirs, files in os.walk(target, followlinks=False):
        Path(parent).chmod(0o755)
        for name in files:
            path = Path(parent) / name
            if not path.is_symlink():
                path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
    # Inaccessible empty roots make forbidden reads/writes unambiguous even
    # though the actual verifier assets are outside this filesystem entirely.
    for name in ("tests", "solution", "logs"):
        path = ROOT / name
        if path.is_symlink():
            raise RuntimeError("Protected filesystem root must not be a link")
        path.mkdir(mode=0o000, exist_ok=True)
        path.chmod(0o000)
    for name, minor in (("null", 3), ("urandom", 9)):
        device = ROOT / "dev" / name
        if not device.exists():
            os.mknod(device, stat.S_IFCHR | 0o666, os.makedev(1, minor))
        metadata = device.lstat()
        if not stat.S_ISCHR(metadata.st_mode) or metadata.st_rdev != os.makedev(1, minor):
            raise RuntimeError("Unexpected device in candidate filesystem")
        device.chmod(0o666 if name == "null" else 0o444)


def enter_root(source, scratch, inputs):
    """Enter the prepared root before permanently dropping the bootstrap UID."""
    if source != "/app/src" or not (ROOT / "app/src/candidate.py").is_file():
        raise RuntimeError("Candidate filesystem was not prepared")
    uid = _worker_identity()
    for path in (scratch, inputs):
        pure = Path(path)
        if not pure.is_absolute() or ".." in pure.parts or pure.parts[1] != "tmp":
            raise RuntimeError("Private worker path must be below /tmp")
        destination = ROOT / str(pure).lstrip("/")
        destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        destination.parent.chmod(0o755)
        if destination.is_symlink():
            raise RuntimeError("Worker filesystem path must not be a link")
        if destination.exists():
            # A closed RPC worker may be restarted with the same public proxy.
            # Its parent directory is root-owned and not writable by candidate.
            shutil.rmtree(destination)
        if path == scratch:
            destination.mkdir(mode=0o700)
            os.chown(destination, uid, uid)
        else:
            shutil.copytree(path, destination, symlinks=True)
            destination.chmod(0o755)
    os.chdir(ROOT)
    os.chroot(".")
    os.chdir("/")
    return uid


def publish_input(path, inputs):
    """Publish a new public fixture into a running worker's filesystem."""
    path, inputs = Path(path), Path(inputs)
    if path.is_symlink() or path.parent != inputs or not path.is_file():
        raise RuntimeError("Public input must be a regular file in its input directory")
    if not inputs.is_absolute() or ".." in inputs.parts or inputs.parts[1] != "tmp":
        raise RuntimeError("Public input directory must be below /tmp")
    parent = ROOT / str(inputs).lstrip("/")
    metadata = parent.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise RuntimeError("Public input directory is not protected")
    destination = parent / path.name
    data = path.read_bytes()
    if destination.is_symlink():
        raise RuntimeError("Public input destination cannot be a link")
    if destination.exists():
        metadata = destination.stat()
        if metadata.st_uid != 0 or metadata.st_mode & 0o222 or destination.read_bytes() != data:
            raise RuntimeError("Public input changed after publication")
        return
    with destination.open("xb") as handle:
        handle.write(data)
    destination.chmod(0o444)
