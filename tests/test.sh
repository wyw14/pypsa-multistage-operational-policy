#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
# Failure-safe initialization; only complete authenticated results can grant 1.
/usr/local/bin/python -I /tests/grade.py initialize
workspace="$(mktemp -d /tmp/trusted-verifier-XXXXXXXX)"
chmod 700 "$workspace"
stage=prepare
pytest_status=125
finalized=0

finish() {
    local shell_status=$?
    trap - EXIT HUP INT TERM
    if [[ "$shell_status" != 0 && "$stage" == pytest ]]; then
        stage=shell_failure
        pytest_status="$shell_status"
        finalized=0
    fi
    if [[ "$finalized" != 1 ]]; then
        /usr/local/bin/python -I /tests/grade.py finalize "$workspace" \
            --pytest-status "$pytest_status" --stage "$stage" || true
    fi
    exit "$shell_status"
}
interrupt() {
    stage=interrupted
    pytest_status="$1"
    finalized=0
    exit "$1"
}
trap finish EXIT
trap 'interrupt 129' HUP
trap 'interrupt 130' INT
trap 'interrupt 143' TERM

/usr/local/bin/python -I /tests/grade.py prepare "$workspace"
stage=install_dependencies
cd /tests/wheels
sha256sum -c /tests/wheels-sha256.txt
/usr/local/bin/python -I -m pip install --no-index --no-deps \
    --find-links /tests/wheels -r /tests/requirements-test.txt

stage=pytest
cd /
set +e
env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HOME="$workspace" \
    MPLCONFIGDIR="$workspace/mpl" MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONSAFEPATH=1 PYTHONPATH=/tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    CANDIDATE_PID_LEDGER="$workspace/candidate-pids.txt" \
    /usr/bin/timeout --signal=TERM --kill-after=5s 540s \
    /usr/local/bin/pytest /tests/test_task.py -q --tb=short \
    -p no:cacheprovider -p trusted_runner --trusted-ledger "$workspace/outcomes.json" \
    -c /tests/pytest.ini --confcutdir=/tests --rootdir=/tests \
    2>&1 | tee /logs/verifier/pytest.log
statuses=("${PIPESTATUS[@]}")
pytest_status="${statuses[0]}"
set -e
if [[ "${statuses[1]}" != 0 ]]; then
    stage=log_capture
fi
/usr/local/bin/python -I /tests/grade.py finalize "$workspace" \
    --pytest-status "$pytest_status" --stage "$stage"
finalized=1
