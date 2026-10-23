# Harbor task

The repository root is one standard Harbor task package.

## Verifier host compatibility

The task requires Linux/amd64 and a root bootstrap for its separate verifier.
Candidate computations run without root privileges, with mandatory seccomp and
no-new-privileges. When usable, Landlock provides the filesystem allowlist.
Otherwise the verifier enters a dependency-only chroot, with a distinct UID and
private scratch directory for each worker. The candidate can read its public
source and inputs; tests, reference solutions and scoring files stay outside
the candidate filesystem. Public input files added during an RPC session remain
available without restarting the computation. Neither path grants network access.

The fallback uses the standard Docker capabilities for chroot, device creation
and dropping identity (SYS_CHROOT, MKNOD, SETUID, SETGID). It does not require a
privileged container. The verifier checks actual read/write/socket access before
running tests and fails closed with diagnostics if isolation cannot be established.
Read-only tests/solution mounts are supported. Inspect verifier/preflight.json,
grading.json and pytest.log to distinguish environment failures from test failures.
