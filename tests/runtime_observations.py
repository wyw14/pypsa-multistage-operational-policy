"""Trusted observations inaccessible to code in the candidate process."""

import math

_FAULTS = []
_POLICY_RESULTS = 0


def record_fault(kind, operation):
    """Record an observed transport failure, never infer one from candidate text."""
    if kind not in {"timeout", "crash", "protocol", "bootstrap"}:
        raise ValueError("Unknown runtime fault kind")
    _FAULTS.append({"kind": kind, "operation": operation})


def snapshot():
    """Return evidence to the trusted pytest ledger at session completion."""
    return {"runtime_faults": list(_FAULTS), "policy_results": _POLICY_RESULTS}


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _mapping(value, names):
    return (
        type(value) is dict
        and set(value) == set(names)
        and all(_finite(v) for v in value.values())
    )


def observe_policy(case, result):
    """Credit availability only after checking the complete public policy schema.

    This gate establishes that the new interface delivers a policy. Numerical
    equations, optimality and histories remain independently weighted tests.
    A candidate's status string or the existing deterministic adapter is not
    evidence of a delivered policy.
    """
    global _POLICY_RESULTS
    required = {
        "status",
        "root_objective",
        "expected_cost",
        "node_policy",
        "leaf_carry_state",
    }
    if (
        type(result) is not dict
        or not required.issubset(result)
        or result["status"] != "optimal"
    ):
        return
    if not _finite(result["root_objective"]) or not _finite(result["expected_cost"]):
        return
    network, tree = case["network"], case["tree"]
    names = {
        kind: {r["name"] for r in network.get(kind, [])}
        for kind in ("buses", "lines", "generators", "loads", "storage_units")
    }
    thermal = {
        r["name"] for r in network.get("generators", []) if r.get("committable", False)
    }
    keys = {
        "u": thermal,
        "start": thermal,
        "stop": thermal,
        "p": names["generators"],
        "charge": names["storage_units"],
        "discharge": names["storage_units"],
        "energy": names["storage_units"],
        "shed": names["loads"],
        "flow": names["lines"],
        "theta": names["buses"],
    }
    policies = result["node_policy"]
    nodeids = {node["id"] for node in tree}
    if type(policies) is not dict or not nodeids or set(policies) != nodeids:
        return
    for row in policies.values():
        if type(row) is not dict or not (
            set(keys) | {"local_cost", "risk_to_go"}
        ).issubset(row):
            return
        if not all(_mapping(row[key], values) for key, values in keys.items()):
            return
        if not _finite(row["local_cost"]) or not _finite(row["risk_to_go"]):
            return
    leaves = nodeids - {node["parent"] for node in tree}
    carry = result["leaf_carry_state"]
    if type(carry) is not dict or set(carry) != leaves:
        return
    for row in carry.values():
        if (
            type(row) is not dict
            or not {"generators", "energy"}.issubset(row)
            or not _mapping(row["energy"], names["storage_units"])
        ):
            return
        if type(row["generators"]) is not dict or set(row["generators"]) != thermal:
            return
        for state in row["generators"].values():
            required_state = {"status", "p", "hours", "remaining"}
            if (
                type(state) is not dict
                or not required_state.issubset(state)
                or not all(_finite(state[key]) for key in required_state)
            ):
                return
    _POLICY_RESULTS += 1


def observe_infeasible(independent_optimum, result):
    """Accept infeasibility only when independent enumeration has proved it."""
    global _POLICY_RESULTS
    if independent_optimum == math.inf and result == {"status": "infeasible"}:
        _POLICY_RESULTS += 1
