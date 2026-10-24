"""Pure aggregation of authenticated execution and explicit domain weights."""

from decimal import Decimal


def score_results(data, expected, policy, pytest_status):
    """Keep incomplete execution separate from an ordinary scientific failure."""
    failures = []
    entries = policy["tests"]
    weights = {entry["id"]: Decimal(str(entry["weight"])) for entry in entries}
    if len(weights) != len(entries) or set(weights) != set(expected):
        raise ValueError("Score map must cover every expected test exactly once")
    if (
        any(not w.is_finite() or w <= 0 for w in weights.values())
        or sum(weights.values()) != 1
    ):
        raise ValueError("Every test needs a positive finite weight; total must be one")
    if any(
        not isinstance(entry.get("reason"), str) or not entry["reason"].strip()
        for entry in entries
    ):
        raise ValueError("Every weight requires its scientific justification")
    if len(expected) != len(set(expected)) or not expected:
        raise ValueError("Expected test identities must be nonempty and unique")
    collected, outcomes = data.get("collected"), data.get("outcomes")
    complete = (
        type(collected) is list
        and len(collected) == len(expected)
        and len(set(collected)) == len(collected)
        and set(collected) == set(expected)
        and type(outcomes) is dict
        and set(outcomes) == set(expected)
        and data.get("collection_errors") == []
        and data.get("session_complete") is True
    )
    if not complete:
        failures.append("incomplete_collection_or_session")
    passed = set()
    if complete:
        for nodeid in expected:
            phases = outcomes[nodeid]
            if phases == [
                ["setup", "passed"],
                ["call", "passed"],
                ["teardown", "passed"],
            ]:
                passed.add(nodeid)
            elif phases != [
                ["setup", "passed"],
                ["call", "failed"],
                ["teardown", "passed"],
            ]:
                failures.append("incomplete_or_invalid_phases:" + nodeid)
    expected_exit = 0 if len(passed) == len(expected) else 1
    if (
        type(pytest_status) is not int
        or type(data.get("exitstatus")) is not int
        or pytest_status != data["exitstatus"]
        or pytest_status not in (0, 1)
        or (complete and pytest_status != expected_exit)
    ):
        failures.append("abnormal_or_inconsistent_pytest_exit")
    faults = data.get("runtime_faults")
    if type(faults) is not list or any(
        type(fault) is not dict
        or fault.get("kind") not in {"timeout", "crash", "protocol", "bootstrap"}
        or type(fault.get("operation")) is not str
        for fault in faults
    ):
        failures.append("missing_or_invalid_runtime_fault_record")
        faults = []
    elif faults:
        failures.append("candidate_runtime_fault")
    if policy.get("require_policy_result") is True:
        count = data.get("policy_results")
        if type(count) is not int or count <= 0:
            failures.append("no_complete_new_policy_result")
    weighted = {}
    for entry in entries:
        key = entry["behavior"]
        group = weighted.setdefault(key, {"possible": Decimal(0), "earned": Decimal(0)})
        group["possible"] += weights[entry["id"]]
        if entry["id"] in passed:
            group["earned"] += weights[entry["id"]]
    earned = sum(weights[nodeid] for nodeid in passed)
    return {
        "reward": 0.0 if failures else float(earned),
        "collection_complete": complete,
        "hard_gate_failures": failures,
        "runtime_faults": faults,
        "timed_out": pytest_status in (124, 137)
        or any(f["kind"] == "timeout" for f in faults),
        "policy_results": data.get("policy_results", 0),
        "weighted_behaviors": {
            name: {key: float(value) for key, value in group.items()}
            for name, group in weighted.items()
        },
    }
