"""Public input normalization; supplied so the task concerns energy modeling."""

import math

from pypsa.optimization.policy_numeric import SCALAR_LIMITS, validate_angle_budget


def number(value, low=None, high=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Expected a finite number")
    value = float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError("Number outside supported domain")
    return value


def positive(value, high=None):
    """A public strict-positive domain has no hidden numerical lower floor."""
    value = number(value, high=high)
    if value <= 0:
        raise ValueError("Expected a strictly positive number")
    return value


def normalize(n, tree, options):
    if len(n.buses) < 1 or len(n.buses) > 8 or len(n.lines) > 12:
        raise ValueError("Unsupported network size")
    buses = sorted(map(str, n.buses.index))
    gens = sorted(map(str, n.generators.index))
    loads = sorted(map(str, n.loads.index))
    lines = sorted(map(str, n.lines.index))
    stores = sorted(map(str, n.storage_units.index))
    thermal = [g for g in gens if bool(n.generators.at[g, "committable"])]
    renewable = [g for g in gens if g not in thermal]
    if len(thermal) > 3 or len(stores) > 2 or len(gens) > 8 or len(loads) > 8:
        raise ValueError("Unsupported component count")
    for b in buses:
        number(float(n.buses.at[b, "v_nom"]), *SCALAR_LIMITS["v_nom"])
    for group in (n.generators, n.loads, n.storage_units):
        if not set(map(str, group.bus)).issubset(buses):
            raise ValueError("Unknown bus")
    for l in lines:
        row = n.lines.loc[l]
        if row.bus0 not in buses or row.bus1 not in buses or row.bus0 == row.bus1:
            raise ValueError("Invalid line endpoints")
        if n.buses.at[row.bus0, "v_nom"] != n.buses.at[row.bus1, "v_nom"]:
            raise ValueError("Line voltage bases differ")
        number(float(row.x), *SCALAR_LIMITS["x"])
        number(float(row.s_nom), *SCALAR_LIMITS["s_nom"])
    for g in gens:
        row = n.generators.loc[g]
        number(float(row.p_nom), *SCALAR_LIMITS["p_nom"])
        number(float(row.p_min_pu), 0, 1)
        number(float(row.p_max_pu), float(row.p_min_pu), 1)
        if g in renewable and row.p_min_pu != 0:
            raise ValueError("Renewables must be curtailable to zero")
        for key in (
            "marginal_cost",
            "stand_by_cost",
            "start_up_cost",
            "shut_down_cost",
        ):
            number(float(row[key]), *SCALAR_LIMITS["cost"])
        for key in (
            "ramp_limit_up",
            "ramp_limit_down",
            "ramp_limit_start_up",
            "ramp_limit_shut_down",
        ):
            value = float(row[key])
            if not math.isnan(value):
                number(value, 0, 1)
        for key in ("min_up_time", "min_down_time"):
            value = number(float(row[key]), 0, 8)
            if value != int(value):
                raise ValueError("Minimum times must be integral")
    for s in stores:
        row = n.storage_units.loc[s]
        number(float(row.p_nom), *SCALAR_LIMITS["p_nom"])
        number(float(row.max_hours), *SCALAR_LIMITS["max_hours"])
        number(float(row.p_min_pu), -1, 0)
        number(float(row.p_max_pu), 0, 1)
        number(float(row.efficiency_store), *SCALAR_LIMITS["efficiency"])
        number(float(row.efficiency_dispatch), *SCALAR_LIMITS["efficiency"])
        number(float(row.standing_loss), 0, 1)
        number(float(row.state_of_charge_initial), 0, float(row.p_nom * row.max_hours))
        number(float(row.marginal_cost), *SCALAR_LIMITS["cost"])
    required = {
        "risk_weight",
        "alpha",
        "initial",
        "terminal_soc",
        "shed_cost",
        "throughput_cost",
    }
    if not isinstance(options, dict) or set(options) != required:
        raise ValueError("Invalid options fields")
    number(options["risk_weight"], 0, 1)
    number(options["alpha"], 0, 0.95)
    for key, names in (
        ("initial", thermal),
        ("terminal_soc", stores),
        ("shed_cost", loads),
        ("throughput_cost", stores),
    ):
        if not isinstance(options[key], dict) or set(options[key]) != set(names):
            raise ValueError("Incomplete option map")
    for g in thermal:
        h = options["initial"][g]
        if not isinstance(h, dict) or set(h) != {"status", "p", "hours", "root_status"}:
            raise ValueError("Invalid thermal initial state")
        for key in ("status", "root_status"):
            if isinstance(h[key], bool) or h[key] not in (0, 1):
                raise ValueError("Status must be 0 or 1")
        if (
            not isinstance(h["hours"], int)
            or isinstance(h["hours"], bool)
            or not 1 <= h["hours"] <= 10000
        ):
            raise ValueError("Invalid uninterrupted history")
        row = n.generators.loc[g]
        number(
            h["p"],
            float(row.p_nom * row.p_min_pu * h["status"]),
            float(row.p_nom * row.p_max_pu * h["status"]),
        )
    for s in stores:
        row = n.storage_units.loc[s]
        number(options["terminal_soc"][s], 0, float(row.p_nom * row.max_hours))
        number(options["throughput_cost"][s], *SCALAR_LIMITS["cost"])
    for l in loads:
        number(options["shed_cost"][l], *SCALAR_LIMITS["cost"])
    if not isinstance(tree, list) or not 1 <= len(tree) <= 25:
        raise ValueError("Invalid tree size")
    by_id = {}
    fields = {
        "id",
        "parent",
        "stage",
        "probability",
        "load",
        "renewable_pu",
        "line_status",
    }
    for node in tree:
        if not isinstance(node, dict) or set(node) != fields:
            raise ValueError("Invalid node fields")
        key = node["id"]
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError("Node ids must be distinct nonempty strings")
        if (
            not isinstance(node["stage"], int)
            or isinstance(node["stage"], bool)
            or not 0 <= node["stage"] <= 7
        ):
            raise ValueError("Invalid stage")
        positive(node["probability"], 1)
        for name, members in (
            ("load", loads),
            ("renewable_pu", renewable),
            ("line_status", lines),
        ):
            if not isinstance(node[name], dict) or set(node[name]) != set(members):
                raise ValueError("Incomplete observation map")
        for x in node["load"].values():
            number(x, 0, 1e5)
        for x in node["renewable_pu"].values():
            number(x, 0, 1)
        if any(not isinstance(x, bool) for x in node["line_status"].values()):
            raise ValueError("Line status must be boolean")
        by_id[key] = node
    roots = [v for v in tree if v["parent"] is None]
    if len(roots) != 1 or roots[0]["stage"] != 0 or roots[0]["probability"] != 1:
        raise ValueError("A unique probability-one stage-zero root is required")
    root = roots[0]["id"]
    children = {key: [] for key in by_id}
    for key, node in by_id.items():
        if key == root:
            continue
        parent = node["parent"]
        if parent not in by_id or by_id[parent]["stage"] + 1 != node["stage"]:
            raise ValueError("Invalid parent or stage")
        children[parent].append(key)
    leaves = [key for key, sons in children.items() if not sons]
    if len({by_id[key]["stage"] for key in leaves}) != 1:
        raise ValueError("All leaves must have the same stage")
    for sons in children.values():
        sons.sort()
        if sons and abs(sum(by_id[key]["probability"] for key in sons) - 1) > 1e-10:
            raise ValueError("Conditional probabilities must sum to one")
    validate_angle_budget(n, buses, lines, by_id)
    ordered = sorted(by_id, key=lambda key: (by_id[key]["stage"], key))
    return {
        "buses": buses,
        "gens": gens,
        "thermal": thermal,
        "renewable": renewable,
        "loads": loads,
        "lines": lines,
        "stores": stores,
        "nodes": by_id,
        "children": children,
        "leaves": sorted(leaves),
        "root": root,
        "ordered": ordered,
    }
