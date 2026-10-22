"""JSON transport for the native PyPSA policy-tree extension.

The optimization lives in pypsa.optimization.scenario_tree, reached through the
normal Network optimization accessor. This file only constructs native objects.
"""
import pypsa


def build_network(data):
    n = pypsa.Network()
    allowed = {
        "buses": ("Bus", {"name", "v_nom"}),
        "lines": ("Line", {"name", "bus0", "bus1", "x", "s_nom"}),
        "generators": ("Generator", {"name", "bus", "p_nom", "committable", "p_min_pu", "p_max_pu", "marginal_cost", "stand_by_cost", "start_up_cost", "shut_down_cost", "min_up_time", "min_down_time", "ramp_limit_up", "ramp_limit_down", "ramp_limit_start_up", "ramp_limit_shut_down"}),
        "loads": ("Load", {"name", "bus"}),
        "storage_units": ("StorageUnit", {"name", "bus", "p_nom", "p_min_pu", "p_max_pu", "max_hours", "efficiency_store", "efficiency_dispatch", "standing_loss", "state_of_charge_initial", "marginal_cost"}),
    }
    if not isinstance(data, dict) or set(data) - set(allowed):
        raise ValueError("Unsupported network component")
    for key, (kind, fields) in allowed.items():
        seen = set()
        for raw in data.get(key, []):
            if not isinstance(raw, dict) or set(raw) - fields:
                raise ValueError("Unsupported component attribute")
            item = dict(raw)
            name = item.pop("name", None)
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("Names must be unique nonempty strings")
            seen.add(name)
            n.add(kind, name, **item)
    return n


def solve(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected an object")
    operation = payload.get("operation", "policy")
    n = build_network(payload["network"])
    if operation == "deterministic":
        n.set_snapshots([0])
        for name, load in payload["load"].items():
            n.loads.at[name, "p_set"] = float(load)
        status, condition = n.optimize(solver_name="highs", log_to_console=False)
        return {"status": status, "condition": condition,
                "objective": float(n.objective),
                "generation": {str(g): float(n.generators_t.p.at[0, g]) for g in n.generators.index}}
    if operation != "policy":
        raise ValueError("Unsupported operation")
    return n.optimize.policy_tree(payload["tree"], payload["options"])
