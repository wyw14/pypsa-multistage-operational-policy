"""Physical policy, causal history, topology and risk checked independently.

All cases are synthetic bounded research networks, not measured grid records.
Optimum fixtures enumerate discrete physical policies and solve risk-envelope
LPs; raw returned policies are checked without importing candidate code.
"""

import copy
import json
import math
import pytest
from candidate_proxy import candidate
from independent import exhaustive_optimum
from numeric_domain import independent_angle_budgets
from fractions import Fraction
from runtime_observations import observe_policy, observe_infeasible


def run_candidate(case):
    """题面：所有返回值是普通 JSON 数、字符串、数组或对象，不包含 NaN/Infinity。"""
    return json.loads(json.dumps(candidate.run(case), allow_nan=False))


def close(actual, expected, path="value"):
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
        f"{path}: expected a number, got {actual!r}"
    )
    assert math.isfinite(actual), f"{path}: nonfinite value {actual!r}"
    assert actual == pytest.approx(expected, abs=2e-5, rel=2e-6), (
        f"{path}: actual={actual!r}, expected={expected!r}, atol=2e-5, rtol=2e-6"
    )


def tail(values, probabilities, alpha):
    remaining, weighted = 1 - alpha, 0.0
    for i in sorted(range(len(values)), key=lambda j: values[j], reverse=True):
        take = min(remaining, probabilities[i])
        weighted += take * values[i]
        remaining -= take
        if remaining <= 0:
            break
    return weighted / (1 - alpha)


def base_case():
    return {
        "network": {
            "buses": [{"name": "A", "v_nom": 10}, {"name": "B", "v_nom": 10}],
            "lines": [{"name": "AB", "bus0": "A", "bus1": "B", "x": 1.7, "s_nom": 7}],
            "generators": [
                {
                    "name": "GT",
                    "bus": "A",
                    "p_nom": 8,
                    "committable": True,
                    "p_min_pu": 0.2,
                    "p_max_pu": 1,
                    "marginal_cost": 4,
                    "stand_by_cost": 2,
                    "start_up_cost": 15,
                    "shut_down_cost": 3,
                    "min_up_time": 2,
                    "min_down_time": 1,
                    "ramp_limit_up": 0.5,
                    "ramp_limit_down": 0.6,
                    "ramp_limit_start_up": 0.7,
                    "ramp_limit_shut_down": 0.7,
                },
                {
                    "name": "PV",
                    "bus": "B",
                    "p_nom": 6,
                    "committable": False,
                    "marginal_cost": 0,
                },
            ],
            "loads": [{"name": "LA", "bus": "A"}, {"name": "LB", "bus": "B"}],
            "storage_units": [
                {
                    "name": "BAT",
                    "bus": "B",
                    "p_nom": 3,
                    "max_hours": 2,
                    "efficiency_store": 0.9,
                    "efficiency_dispatch": 0.8,
                    "standing_loss": 0.05,
                    "state_of_charge_initial": 1.3,
                    "marginal_cost": 0.2,
                }
            ],
        },
        "tree": [],
        "options": {
            "risk_weight": 0.65,
            "alpha": 0.73,
            "initial": {"GT": {"status": 0, "p": 0, "hours": 3, "root_status": 0}},
            "terminal_soc": {"BAT": 0.4},
            "shed_cost": {"LA": 100, "LB": 120},
            "throughput_cost": {"BAT": 0.1},
        },
    }


def node(key, parent, stage, prob=1, la=2, lb=3, pv=0, on=True):
    return {
        "id": key,
        "parent": parent,
        "stage": stage,
        "probability": prob,
        "load": {"LA": la, "LB": lb},
        "renewable_pu": {"PV": pv},
        "line_status": {"AB": on},
    }


def no_storage(case):
    case["network"]["storage_units"] = []
    case["options"]["terminal_soc"] = {}
    case["options"]["throughput_cost"] = {}


def branched(storage=False):
    case = base_case()
    if not storage:
        no_storage(case)
    case["tree"] = [
        node("root", None, 0, la=1, lb=1, pv=1),
        node("low", "root", 1, 0.7, la=2, lb=1, pv=0.6),
        node("high", "root", 1, 0.3, la=2, lb=4, pv=0, on=False),
        node("low-end", "low", 2, la=2, lb=4, pv=0.1),
        node("high-end", "high", 2, la=2, lb=3, pv=0.1, on=True),
    ]
    return case


@pytest.mark.parametrize("malformed", ["network", "tree", "options", "collection", "parent"])
def test_malformed_adapter_structure_raises_value_error(malformed):
    """题面：结构或支持域错误抛 `ValueError`，不依赖异常消息。"""
    case = branched()
    if malformed == "collection":
        case["network"]["loads"] = None
    elif malformed == "parent":
        case["tree"][1]["parent"] = ["root"]
    else:
        del case[malformed]
    with pytest.raises(ValueError):
        run_candidate(case)


def validate(case, result, optimum=None):
    observe_policy(case, result)
    if optimum is not None and not math.isfinite(optimum):
        assert result == {"status": "infeasible"}
        observe_infeasible(optimum, result)
        return
    assert result["status"] == "optimal"
    tables = {
        kind: {r["name"]: r for r in case["network"].get(kind, [])}
        for kind in ("buses", "lines", "generators", "loads", "storage_units")
    }
    buses, lines, gens, loads, stores = [
        tables[k] for k in ("buses", "lines", "generators", "loads", "storage_units")
    ]
    thermal = [g for g, r in gens.items() if r.get("committable", False)]
    nodes = {r["id"]: r for r in case["tree"]}
    order = sorted(nodes, key=lambda a: (nodes[a]["stage"], a))
    kids = {a: [b for b in nodes if nodes[b]["parent"] == a] for a in nodes}
    policies = result["node_policy"]
    assert set(policies) == set(nodes)
    paths = {}
    for a in order:
        raw = nodes[a]
        parent = raw["parent"]
        paths[a] = (paths[parent] if parent is not None else []) + [a]
        out = policies[a]
        for field, names in (
            ("u", thermal),
            ("start", thermal),
            ("stop", thermal),
            ("p", gens),
            ("charge", stores),
            ("discharge", stores),
            ("energy", stores),
            ("shed", loads),
            ("flow", lines),
            ("theta", buses),
        ):
            assert set(out[field]) == set(names)
            for v in out[field].values():
                close(v, v)
        for g in thermal:
            h = case["options"]["initial"][g]
            r = gens[g]
            u = out["u"][g]
            assert u in (0, 1)
            before = h["status"] if parent is None else policies[parent]["u"][g]
            prevp = h["p"] if parent is None else policies[parent]["p"][g]
            close(out["start"][g], max(0, u - before))
            close(out["stop"][g], max(0, before - u))
            if parent is None:
                close(u, h["root_status"])
            for sibling in kids[a]:
                close(policies[sibling]["u"][g], policies[kids[a][0]]["u"][g])
            duration = 0
            for old in reversed(paths[a][:-1]):
                if policies[old]["u"][g] != before:
                    break
                duration += 1
            if duration == len(paths[a]) - 1 and before == h["status"]:
                duration += h["hours"]
            if u != before:
                assert duration >= r.get(
                    "min_up_time" if before else "min_down_time", 0
                )
            p, P = out["p"][g], r["p_nom"]
            assert p >= P * r.get("p_min_pu", 0) * u - 2e-5
            assert p <= P * r.get("p_max_pu", 1) * u + 2e-5
            up = P * (
                r.get("ramp_limit_up", 1) * before
                + r.get("ramp_limit_start_up", 1) * (u - before)
            )
            down = P * (
                r.get("ramp_limit_down", 1) * u
                + r.get("ramp_limit_shut_down", 1) * (before - u)
            )
            assert p - prevp <= up + 2e-5
            assert prevp - p <= down + 2e-5
        for g, r in gens.items():
            if g not in thermal:
                assert out["p"][g] >= -2e-5
                assert (
                    out["p"][g]
                    <= r["p_nom"] * r.get("p_max_pu", 1) * raw["renewable_pu"][g] + 2e-5
                )
        for s, r in stores.items():
            charge, discharge, e = (
                out["charge"][s],
                out["discharge"][s],
                out["energy"][s],
            )
            assert charge >= -2e-5 and discharge >= -2e-5
            assert min(charge, discharge) <= 2e-5
            assert charge <= -r.get("p_min_pu", -1) * r["p_nom"] + 2e-5
            assert discharge <= r.get("p_max_pu", 1) * r["p_nom"] + 2e-5
            assert -2e-5 <= e <= r["p_nom"] * r.get("max_hours", 1) + 2e-5
            before = (
                r.get("state_of_charge_initial", 0)
                if parent is None
                else policies[parent]["energy"][s]
            )
            close(
                e,
                (1 - r.get("standing_loss", 0)) * before
                + r.get("efficiency_store", 1) * charge
                - discharge / r.get("efficiency_dispatch", 1),
            )
            if not kids[a]:
                assert e >= case["options"]["terminal_soc"][s] - 2e-5
        graph = {b: set() for b in buses}
        for component_name, r in lines.items():
            flow = out["flow"][component_name]
            assert abs(flow) <= r["s_nom"] + 2e-5
            if raw["line_status"][component_name]:
                close(
                    flow,
                    buses[r["bus0"]].get("v_nom", 1) ** 2
                    / r["x"]
                    * (out["theta"][r["bus0"]] - out["theta"][r["bus1"]]),
                )
                graph[r["bus0"]].add(r["bus1"])
                graph[r["bus1"]].add(r["bus0"])
            else:
                close(flow, 0)
        unseen = set(buses)
        while unseen:
            reference = min(unseen)
            stack = [reference]
            unseen.remove(reference)
            while stack:
                for b in graph[stack.pop()]:
                    if b in unseen:
                        unseen.remove(b)
                        stack.append(b)
            close(out["theta"][reference], 0)
        for component_name in loads:
            assert (
                -2e-5
                <= out["shed"][component_name]
                <= raw["load"][component_name] + 2e-5
            )
        for b in buses:
            generation = sum(out["p"][g] for g, r in gens.items() if r["bus"] == b)
            storage = sum(
                out["discharge"][s] - out["charge"][s]
                for s, r in stores.items()
                if r["bus"] == b
            )
            demand = sum(
                raw["load"][component_name] - out["shed"][component_name]
                for component_name, r in loads.items()
                if r["bus"] == b
            )
            export = sum(
                out["flow"][component_name]
                for component_name, r in lines.items()
                if r["bus0"] == b
            ) - sum(
                out["flow"][component_name]
                for component_name, r in lines.items()
                if r["bus1"] == b
            )
            close(generation + storage, demand + export)
        cost = sum(out["p"][g] * r.get("marginal_cost", 0) for g, r in gens.items())
        cost += sum(
            out["u"][g] * gens[g].get("stand_by_cost", 0)
            + out["start"][g] * gens[g].get("start_up_cost", 0)
            + out["stop"][g] * gens[g].get("shut_down_cost", 0)
            for g in thermal
        )
        cost += sum(
            out["shed"][component_name] * case["options"]["shed_cost"][component_name]
            for component_name in loads
        )
        cost += sum(
            (out["charge"][s] + out["discharge"][s])
            * case["options"]["throughput_cost"][s]
            + out["discharge"][s] * r.get("marginal_cost", 0)
            for s, r in stores.items()
        )
        close(out["local_cost"], cost, f"node_policy[{a!r}].local_cost")
    expected = {}
    for a in reversed(order):
        out = policies[a]
        vals = [policies[b]["risk_to_go"] for b in kids[a]]
        qs = [nodes[b]["probability"] for b in kids[a]]
        risk = out["local_cost"]
        if vals:
            lam = case["options"]["risk_weight"]
            risk += (1 - lam) * sum(q * v for q, v in zip(qs, vals)) + lam * tail(
                vals, qs, case["options"]["alpha"]
            )
        close(out["risk_to_go"], risk, f"node_policy[{a!r}].risk_to_go")
        expected[a] = out["local_cost"] + sum(
            nodes[b]["probability"] * expected[b] for b in kids[a]
        )
    close(
        result["root_objective"],
        policies[order[0]]["risk_to_go"],
        "root_objective versus recomputed policy",
    )
    close(result["expected_cost"], expected[order[0]], "expected_cost")
    if optimum is not None:
        close(
            result["root_objective"],
            optimum,
            "root_objective versus independent optimum",
        )
    leaves = {a for a in nodes if not kids[a]}
    assert set(result["leaf_carry_state"]) == leaves
    for a in leaves:
        carry = result["leaf_carry_state"][a]
        assert set(carry["generators"]) == set(thermal)
        assert set(carry["energy"]) == set(stores)
        for s in stores:
            close(carry["energy"][s], policies[a]["energy"][s])
        for g in thermal:
            h = case["options"]["initial"][g]
            status = policies[a]["u"][g]
            hours = 0
            for b in reversed(paths[a]):
                if policies[b]["u"][g] != status:
                    break
                hours += 1
            if hours == len(paths[a]) and status == h["status"]:
                hours += h["hours"]
            minimum = gens[g].get("min_up_time" if status else "min_down_time", 0)
            state = carry["generators"][g]
            close(state["status"], status)
            close(state["p"], policies[a]["p"][g])
            close(state["hours"], hours)
            close(state["remaining"], max(0, minimum - hours))


def checked(case):
    optimum, _ = exhaustive_optimum(case)
    answer = run_candidate(case)
    if math.isfinite(optimum):
        validate(case, answer)
    observe_infeasible(optimum, answer)
    return answer, optimum


def test_native_deterministic_entry_still_works():
    """题面："保持入口、原生组件语义和已有 `n.optimize(...)` 可用。"。

    A native one-bus merit-order dispatch has exact cost six and supplies two MW.
    """
    case = {
        "operation": "deterministic",
        "network": {
            "buses": [{"name": "b", "v_nom": 10}],
            "generators": [{"name": "g", "bus": "b", "p_nom": 10, "marginal_cost": 3}],
            "loads": [{"name": "l", "bus": "b"}],
        },
        "load": {"l": 2},
    }
    answer = run_candidate(case)
    assert answer["status"] == "ok"
    assert answer["condition"] == "optimal"
    close(answer["objective"], 6)
    close(answer["generation"]["g"], 2)


def test_uc_initial_history_ramps_and_leaf_obligations():
    """题面："沿每条路径遵守最短开/停时间"。

    The prior unfinished on obligation, ramp bounds and finite-window carry state are checked against independent discrete policies.
    """
    case = base_case()
    no_storage(case)
    case["options"]["initial"]["GT"] = {
        "status": 1,
        "p": 2,
        "hours": 1,
        "root_status": 1,
    }
    case["network"]["generators"][0]["min_up_time"] = 3
    case["tree"] = [
        node("a", None, 0, la=2, lb=1, pv=0.1),
        node("b", "a", 1, la=2, lb=5, pv=0),
        node("c", "b", 2, la=1, lb=0, pv=1),
        node("d", "c", 3, la=2, lb=4, pv=0),
    ]
    answer, optimum = checked(case)
    assert answer["root_objective"] == pytest.approx(optimum, abs=2e-5, rel=2e-6), (
        f"root_objective={answer['root_objective']!r}; independent policy LP optimum={optimum!r}"
    )


def test_advance_commitment_cannot_use_the_next_observation():
    """题面："同一父节点的所有子节点对每台热机必须具有相同 u"。

    The quiet child has no sink for positive minimum generation; both children must inherit the single offline commitment and the exact causal cost is422.8.
    """
    case = base_case()
    no_storage(case)
    case["tree"] = [
        node("root", None, 0, la=1, lb=1, pv=1),
        node("quiet", "root", 1, 0.7, la=0, lb=0, pv=0),
        node("scarce", "root", 1, 0.3, la=2, lb=3, pv=0),
    ]
    optimum, schedules = exhaustive_optimum(case)
    # Positive minimum generation has nowhere to go on the quiet sibling.
    # A single prior commitment therefore leaves both children offline.
    close(optimum, 422.8)
    assert schedules == 2
    validate(case, run_candidate(case), optimum)


@pytest.mark.parametrize("weight,alpha", [(0, 0.8), (0.65, 0.73), (1, 0.5), (0.8, 0)])
def test_advance_commitment_and_nested_conditional_risk(weight, alpha):
    """题面："这是逐层条件风险，不是对整条叶路径总成本做一次静态 CVaR。"。

    A second branching level exercises conditional rather than whole-path tail loss, including expectation and pure-CVaR boundaries.
    """
    case = branched()
    case["options"]["risk_weight"], case["options"]["alpha"] = weight, alpha
    # A second split makes nesting structurally distinct from path CVaR.
    case["tree"][-2]["probability"] = 0.8
    case["tree"][-1]["probability"] = 0.6
    case["tree"] += [
        node("low-tail", "low", 2, 0.2, la=2, lb=6, pv=0),
        node("high-tail", "high", 2, 0.4, la=1, lb=5, pv=0.8),
    ]
    answer, optimum = checked(case)
    assert answer["root_objective"] == pytest.approx(optimum, abs=2e-5, rel=2e-6), (
        f"root_objective={answer['root_objective']!r}; independent policy LP optimum={optimum!r}"
    )


def test_storage_inheritance_islanding_and_reconnection():
    """题面："其余节点来自真实父节点结束电量。"。

    Storage is shared before the fork, then evolves on its actual branch through islanding and reconnection under advance commitments.
    """
    answer, optimum = checked(branched(storage=True))
    assert answer["root_objective"] == pytest.approx(optimum, abs=2e-5, rel=2e-6), (
        f"root_objective={answer['root_objective']!r}; independent storage-mode LP optimum={optimum!r}"
    )


def test_storage_losses_efficiencies_reserve_and_exclusive_modes():
    """题面："一个小时不能同时充放电，均为零允许。"。

    Enumerated idle/charge/discharge modes enforce losses, both conversion efficiencies and each terminal reserve.
    """
    case = base_case()
    case["options"]["terminal_soc"]["BAT"] = 1.7
    case["tree"] = [
        node("r", None, 0, la=0.5, lb=1, pv=1),
        node("s", "r", 1, la=2, lb=6, pv=0),
        node("t", "s", 2, la=2, lb=1, pv=0.2),
    ]
    answer, optimum = checked(case)
    assert answer["root_objective"] == pytest.approx(optimum, abs=2e-5, rel=2e-6), (
        f"root_objective={answer['root_objective']!r}; independent policy LP optimum={optimum!r}"
    )


def test_node_and_component_order_do_not_change_policy_problem():
    """题面："树数组不要求按时间或字典排序。"。

    Reversing component and node arrays retains the same parent-indexed feasible set and optimum, allowing different optimal actions.
    """
    case = branched(storage=True)
    baseline = run_candidate(case)
    validate(case, baseline)
    shuffled = copy.deepcopy(case)
    shuffled["tree"] = list(reversed(shuffled["tree"]))
    for records in shuffled["network"].values():
        records.reverse()
    result = run_candidate(shuffled)
    validate(shuffled, result)
    assert result["root_objective"] == pytest.approx(
        baseline["root_objective"], abs=2e-5, rel=2e-6
    ), (
        f"Reordering changed the optimum: original={baseline['root_objective']!r}, reversed={result['root_objective']!r}"
    )


def test_mesh_flow_and_independent_island_angle_gauges():
    """题面："停运线路 f=0 且不再关联两端相角。"。

    A nonreference island bus retains a genuine angle drop; the failed branch must not falsely tie it to a separate island.
    """
    case = branched()
    case["network"]["buses"].append({"name": "C", "v_nom": 10})
    case["network"]["loads"].append({"name": "LC", "bus": "C"})
    case["options"]["shed_cost"]["LC"] = 110
    case["network"]["lines"] += [
        {"name": "AC", "bus0": "A", "bus1": "C", "x": 1, "s_nom": 1.5},
        {"name": "CB", "bus0": "C", "bus1": "B", "x": 0.8, "s_nom": 2},
    ]
    for raw in case["tree"]:
        raw["load"]["LC"] = 1.1
        raw["line_status"]["AC"] = True
        raw["line_status"]["CB"] = raw["id"] != "high"
    answer, optimum = checked(case)
    assert answer["root_objective"] == pytest.approx(optimum, abs=2e-5, rel=2e-6), (
        f"root_objective={answer['root_objective']!r}; independent policy LP optimum={optimum!r}"
    )


@pytest.mark.parametrize(
    "kind",
    ["initial-obligation", "excess-minimum", "terminal-reserve", "storage-no-disposal"],
)
def test_true_infeasibility_is_reported(kind):
    """题面："合法输入但约束无解返回"。

    Initial commitment obligations, excess minimum power, terminal reserve and forbidden circulating storage each have genuinely infeasible physical schedules.
    """
    case = infeasible_case(kind)
    answer, optimum = checked(case)
    assert math.isinf(optimum), (
        f"Independent enumeration unexpectedly found {kind!r} feasible: objective={optimum!r}"
    )
    assert answer == {"status": "infeasible"}, (
        f"Physically infeasible {kind!r} returned {answer!r}"
    )


def infeasible_case(kind):
    case = base_case()
    case["tree"] = [node("r", None, 0, la=0, lb=0, pv=0)]
    if kind == "initial-obligation":
        case["network"]["generators"][0]["min_up_time"] = 3
        case["options"]["initial"]["GT"] = {
            "status": 1,
            "p": 2,
            "hours": 1,
            "root_status": 0,
        }
    elif kind == "excess-minimum":
        no_storage(case)
        case["options"]["initial"]["GT"] = {
            "status": 1,
            "p": 2,
            "hours": 3,
            "root_status": 1,
        }
    elif kind == "terminal-reserve":
        case["network"]["storage_units"][0]["state_of_charge_initial"] = 0
        case["options"]["terminal_soc"]["BAT"] = 2
    else:
        case["options"]["initial"]["GT"] = {
            "status": 1,
            "p": 2,
            "hours": 3,
            "root_status": 1,
        }
        case["network"]["storage_units"][0].update(
            p_nom=10, max_hours=1, state_of_charge_initial=10
        )
    return case


def test_twenty_five_node_supported_tree():
    """题面："1–25 个节点对象的数组"。

    An exact three-hour constant-demand cost verifies the declared upper node count and all three levels of observations.
    """
    case = base_case()
    no_storage(case)
    case["network"]["generators"] = [
        {
            "name": "PV",
            "bus": "B",
            "p_nom": 20,
            "committable": False,
            "marginal_cost": 2,
        }
    ]
    case["options"]["initial"] = {}
    case["tree"] = [node("root", None, 0, la=1, lb=2, pv=1)]
    for j in range(6):
        key = f"s{j}"
        case["tree"].append(node(key, "root", 1, 1 / 6, la=1, lb=2, pv=1))
        for k in range(3):
            case["tree"].append(node(f"{key}-{k}", key, 2, 1 / 3, la=1, lb=2, pv=1))
    answer = run_candidate(case)
    validate(case, answer)
    assert answer["root_objective"] == pytest.approx(18, abs=2e-5, rel=2e-6), (
        f"Three one-hour demands of 3 MW at cost 2 require exact objective18, got {answer['root_objective']!r}"
    )


@pytest.mark.parametrize(
    "kind", ["probability", "missing-parent", "unequal-depth", "unknown-attribute"]
)
def test_invalid_contract_inputs_raise_value_error(kind):
    """题面："结构或支持域错误抛 `ValueError`"。

    Invalid conditional probability, parent, depth or unsupported native attributes are rejected without constraining the exception message.
    """
    case = invalid_case(kind)
    with pytest.raises(ValueError):
        run_candidate(case)


def invalid_case(kind):
    case = branched()
    if kind == "probability":
        case["tree"][1]["probability"] = 0.4
    elif kind == "missing-parent":
        case["tree"][1]["parent"] = "absent"
    elif kind == "unequal-depth":
        case["tree"].pop()
    else:
        case["network"]["generators"][0]["p_nom_extendable"] = True
    return case


def analytic_network_case(*, probability=None, reactance=None, voltage=10.0):
    """A single-source research limit with an explicit primal/dual cost bound."""
    case = {
        "network": {
            "buses": [{"name": "A", "v_nom": voltage}],
            "generators": [
                {
                    "name": "G",
                    "bus": "A",
                    "p_nom": 2.0,
                    "committable": False,
                    "marginal_cost": 1.0,
                }
            ],
            "loads": [{"name": "L", "bus": "A"}],
        },
        "tree": [],
        "options": {
            "risk_weight": 0.5,
            "alpha": 0.5,
            "initial": {},
            "terminal_soc": {},
            "shed_cost": {"L": 10.0},
            "throughput_cost": {},
        },
    }
    flags = {}
    if reactance is not None:
        case["network"]["buses"].append({"name": "B", "v_nom": voltage})
        case["network"]["loads"][0]["bus"] = "B"
        case["network"]["lines"] = [
            {"name": "AB", "bus0": "A", "bus1": "B", "x": reactance, "s_nom": 2.0}
        ]
        flags = {"AB": True}

    def item(key, parent, stage, q):
        return {
            "id": key,
            "parent": parent,
            "stage": stage,
            "probability": q,
            "load": {"L": 1.0},
            "renewable_pu": {"G": 1.0},
            "line_status": dict(flags),
        }

    case["tree"] = [item("root", None, 0, 1.0)]
    if probability is not None:
        case["tree"] += [
            item("rare", "root", 1, probability),
            item("usual", "root", 1, 1 - probability),
        ]
    return case


@pytest.mark.parametrize(
    "probability", [1e-13, float.fromhex("0x0.0000000000001p-1022")]
)
def test_all_strictly_positive_representable_probabilities_are_valid(probability):
    """题面："probability: 从父节点到本节点的条件概率，严格正数，根为1"。

    Equal-cost children have an exact total of two; a positive branch remains
    physically represented even when its probability is a subnormal float.
    """
    case = analytic_network_case(probability=probability)
    result = run_candidate(case)
    validate(case, result, 2.0)
    assert set(result["node_policy"]) == {"root", "rare", "usual"}, (
        "A positive-probability branch must not disappear"
    )
    close(
        result["node_policy"]["rare"]["p"]["G"],
        1.0,
        "rare.p.G analytic supply certificate",
    )


def test_positive_reactance_has_no_hidden_lower_floor():
    """题面："`v_nom>0`（kV），`x>0`（欧姆），`s_nom>=0`（MW）。"

    The on-line two-bus certificate has G=f=1, no shedding, and thetaB=-x/v²;
    nonnegative costs give the matching one-unit lower bound independently.
    """
    case = analytic_network_case(reactance=1e-9)
    result = run_candidate(case)
    validate(case, result)
    assert result["root_objective"] == pytest.approx(1.0, abs=2e-5, rel=2e-6), (
        f"Analytic one-unit supply optimum is1, got {result['root_objective']!r}"
    )
    close(result["node_policy"]["root"]["flow"]["AB"], 1.0, "root.flow.AB")
    theta = result["node_policy"]["root"]["theta"]["B"]
    close(-theta * 1e11, 1.0, "root.theta.B through original-unit DC relation")


def test_finite_large_dc_coefficient_preserves_tiny_physical_angle():
    """题面："f = (v_nom(bus0)**2/x)*(theta_bus0-theta_bus1), |f| <= s_nom"。

    An explicit feasible optimum distinguishes electrical infeasibility from
    solver matrix errors; a tiny but necessary angle cannot be rounded to zero.
    """
    case = analytic_network_case(reactance=1e-6, voltage=1e5)
    result = run_candidate(case)
    validate(case, result)
    assert result["root_objective"] == pytest.approx(1.0, abs=2e-5, rel=2e-6), (
        f"Analytic one-unit supply optimum is1, got {result['root_objective']!r}"
    )
    theta = result["node_policy"]["root"]["theta"]["B"]
    close(-theta * 1e16, 1.0, "root.theta.B through original-unit DC relation")
    independent, _ = exhaustive_optimum(case)
    close(
        independent, 1.0, "independent resistance-form LP versus analytic certificate"
    )


def test_tiny_probability_keeps_a_material_tail_risk_contribution():
    """题面："边界原子只取所需质量。"；"同一父节点的所有子节点对每台热机必须具有相同 u"。

    A rare loss of 1e13 at probability1e-13 costs1.5 under the declared mixed
    tail risk, whereas the shared next-hour commitment costs exactly0.5.
    """
    case = analytic_network_case(probability=1e-13)
    gen = case["network"]["generators"][0]
    gen.update(
        committable=True, p_nom=1.0, marginal_cost=0.0, stand_by_cost=0.5, p_min_pu=0.0
    )
    case["options"]["initial"] = {
        "G": {"status": 0, "p": 0.0, "hours": 3, "root_status": 0}
    }
    case["options"]["shed_cost"]["L"] = 1e13
    for item in case["tree"]:
        item["renewable_pu"] = {}
        item["load"]["L"] = 1.0 if item["id"] == "rare" else 0.0
    result = run_candidate(case)
    validate(case, result, 0.5)
    assert (
        result["node_policy"]["rare"]["u"]["G"]
        == result["node_policy"]["usual"]["u"]["G"]
        == 1
    ), "Dropping the rare atom chooses the wrong advance commitment"
    independent, _ = exhaustive_optimum(case)
    close(independent, 0.5, "independent exact-probability risk-envelope optimum")


def test_mixed_reactances_preserve_a_required_transfer():
    """题面："f = (v_nom(bus0)**2/x)*(theta_bus0-theta_bus1), |f| <= s_nom"。

    Two consecutive lines differ in reactance by1e12, but each must transfer
    one MW. The explicit angle certificate and a cost lower bound establish
    optimum1 independently; the JSON-only resistance-form LP must agree.
    """
    case = analytic_network_case(reactance=1e-6)
    case["network"]["buses"].append({"name": "C", "v_nom": 10.0})
    case["network"]["loads"][0]["bus"] = "C"
    case["network"]["lines"].append(
        {"name": "BC", "bus0": "B", "bus1": "C", "x": 1e6, "s_nom": 2.0}
    )
    case["tree"][0]["line_status"]["BC"] = True
    result = run_candidate(case)
    validate(case, result)
    assert result["root_objective"] == pytest.approx(1.0, abs=2e-5, rel=2e-6), (
        f"The1MW source and two-line transfer attain matching lower bound1, got {result['root_objective']!r}"
    )
    close(
        result["node_policy"]["root"]["flow"]["AB"],
        1.0,
        "root.flow.AB required series transfer",
    )
    close(
        result["node_policy"]["root"]["flow"]["BC"],
        1.0,
        "root.flow.BC required series transfer",
    )
    independent, _ = exhaustive_optimum(case)
    close(independent, 1.0, "independent resistance-form LP versus series certificate")


NUMERIC_INTERVALS = [
    ("Bus.v_nom", 1e-3, 1e5),
    ("Line.x", 1e-12, 1e6),
    ("Line.s_nom", 0.0, 1e6),
    ("Generator.p_nom", 1e-6, 1e6),
    ("StorageUnit.p_nom", 1e-6, 1e6),
    ("StorageUnit.max_hours", 1e-6, 1e4),
    ("StorageUnit.efficiency_store", 1e-6, 1.0),
    ("StorageUnit.efficiency_dispatch", 1e-6, 1.0),
    ("Generator.marginal_cost", 0.0, 1e13),
    ("Generator.stand_by_cost", 0.0, 1e13),
    ("Generator.start_up_cost", 0.0, 1e13),
    ("Generator.shut_down_cost", 0.0, 1e13),
    ("StorageUnit.marginal_cost", 0.0, 1e13),
    ("options.shed_cost", 0.0, 1e13),
    ("options.throughput_cost", 0.0, 1e13),
]


def numeric_interval_case(field, value):
    """Idle physical certificate isolates admission of one declared interval."""
    case = analytic_network_case(reactance=1.0)
    case["tree"][0]["load"]["L"] = 0.0
    case["tree"][0]["renewable_pu"] = {}
    generator = case["network"]["generators"][0]
    generator.update(committable=True, marginal_cost=0.0)
    case["options"]["initial"] = {
        "G": {"status": 0, "p": 0.0, "hours": 3, "root_status": 0}
    }
    storage = {
        "name": "S",
        "bus": "A",
        "p_nom": 1.0,
        "max_hours": 1.0,
        "efficiency_store": 0.9,
        "efficiency_dispatch": 0.9,
        "state_of_charge_initial": 0.0,
        "standing_loss": 0.0,
    }
    case["network"]["storage_units"] = [storage]
    case["options"]["terminal_soc"] = {"S": 0.0}
    case["options"]["throughput_cost"] = {"S": 0.0}
    component, attribute = field.split(".")
    if component == "Bus":
        for bus in case["network"]["buses"]:
            bus[attribute] = value
    elif component == "Line":
        case["network"]["lines"][0][attribute] = value
    elif component == "Generator":
        generator[attribute] = value
    elif component == "StorageUnit":
        storage[attribute] = value
    else:
        name = "L" if attribute == "shed_cost" else "S"
        case["options"][attribute][name] = value
    return case


@pytest.mark.parametrize(
    "field,value",
    [(field, edge) for field, low, high in NUMERIC_INTERVALS for edge in (low, high)],
)
def test_public_numeric_interval_endpoints_are_inclusive(field, value):
    """题面："所有区间均含端点"。

    Every new scalar endpoint admits the exact idle certificate with zero
    demand, offline thermal unit and empty storage. This tests public domain
    admission, not numerical optimality at all combinations of extrema.
    """
    case = numeric_interval_case(field, value)
    budgets = independent_angle_budgets(case)
    assert all(error <= Fraction(1, 5_000_000) for _, _, error in budgets), (
        f"Endpoint fixture {field}={value!r} must satisfy the independently computed public angle budget"
    )
    result = run_candidate(case)
    validate(case, result)
    assert result["root_objective"] == pytest.approx(0.0, abs=2e-5, rel=2e-6), (
        f"{field}={value!r}: admissible zero-demand idle certificate has exact cost0"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        (field, math.nextafter(edge, direction))
        for field, low, high in NUMERIC_INTERVALS
        for edge, direction in ((low, -math.inf), (high, math.inf))
    ],
)
def test_first_values_outside_public_numeric_intervals_raise_value_error(field, value):
    """题面："范围或预算不符抛 `ValueError`"。

    The immediate representable value outside each newly published inclusive
    endpoint is rejected, without requiring any exception message text.
    """
    case = numeric_interval_case(field, value)
    with pytest.raises(ValueError):
        run_candidate(case)


def excessive_rounding_budget_case():
    case = analytic_network_case(reactance=1e6)
    case["network"]["buses"].append({"name": "C", "v_nom": 10.0})
    case["network"]["loads"][0]["bus"] = "C"
    case["network"]["lines"].append(
        {"name": "BC", "bus0": "B", "bus1": "C", "x": 1e-6, "s_nom": 2.0}
    )
    case["tree"][0]["line_status"]["BC"] = True
    return case


@pytest.mark.parametrize("bad_node", ["root", "later"])
def test_public_rounding_budget_checks_every_actual_topology(bad_node):
    """题面："全部节点都要满足该条件，停运线不参与"。

    A scalar-valid high common angle next to a stiff line exceeds the a priori
    serialization budget. It is checked at the root or a later topology,
    independently of whether the physical optimization would be easy.
    """
    case = excessive_rounding_budget_case()
    first = case["tree"][0]
    later = copy.deepcopy(first)
    later.update(id="later", parent="root", stage=1)
    first["line_status"]["AB"] = bad_node == "root"
    case["tree"].append(later)
    budgets = independent_angle_budgets(case)
    assert any(
        node_id == bad_node and error > Fraction(1, 5_000_000)
        for node_id, _, error in budgets
    ), f"Independent all-pairs rational budget must reject topology at {bad_node}"
    with pytest.raises(ValueError):
        run_candidate(case)


def test_unrepresentable_angle_is_outside_the_numeric_domain():
    """题面："Line.x"；"[1e-12,1e6] 欧姆"；"范围或预算不符抛 `ValueError`"。

    Minimal-subnormal reactance would require a 1 MW angle below the smallest
    representable value. The public interval and independent budget exclude
    it; positive subnormal probability remains a separate legal case.
    """
    case = analytic_network_case(reactance=float.fromhex("0x0.0000000000001p-1022"))
    budgets = independent_angle_budgets(case)
    assert budgets[0][2] == 2, (
        "Exact independent rounding-flow bound for this fixture is 2 MW"
    )
    with pytest.raises(ValueError):
        run_candidate(case)


@pytest.mark.parametrize("mode", ["outage", "zero-capacity"])
def test_rounding_budget_respects_outages_and_zero_capacity(mode):
    """题面："零容量在运线路的 w 为0，仍属于电气连通图。"。

    Removing the weak inter-island tie or setting its capacity to zero gives
    a valid topology-dependent budget. The zero-demand idle optimum is exact.
    """
    case = excessive_rounding_budget_case()
    case["tree"][0]["load"]["L"] = 0.0
    apply_zero_or_outage(case, mode)
    budgets = independent_angle_budgets(case)
    assert all(error <= Fraction(1, 5_000_000) for _, _, error in budgets), (
        f"{mode}: independent rational topology budget unexpectedly rejects the physical limit"
    )
    result = run_candidate(case)
    validate(case, result)
    assert result["root_objective"] == pytest.approx(0.0, abs=2e-5, rel=2e-6), (
        f"{mode}: zero-demand idle optimum must remain0"
    )


def apply_zero_or_outage(case, mode):
    if mode == "outage":
        case["tree"][0]["line_status"]["AB"] = False
    else:
        case["network"]["lines"][0]["s_nom"] = 0.0


def mixed_cost_scale_case():
    case = analytic_network_case()
    case["network"]["generators"] = [
        {"name": name, "bus": "A", "p_nom": 1.0,
         "committable": False, "marginal_cost": cost}
        for name, cost in (("G0", 2.0), ("G1", 1.0), ("G2", 1e13))
    ]
    case["tree"][0]["renewable_pu"] = {"G0": 1.0, "G1": 1.0, "G2": 1.0}
    return case


def test_single_bus_mixed_cost_has_unit_optimum():
    """“最小化 `V_root`。”

    One bus, one hour, demand1, generator prices2/1/1e13 and shedding price10.
    G1=1 attains1;
    every supply price is at least1, so the exact optimum is1, independently
    of the reference implementation and the enumerated LP.
    """
    case = mixed_cost_scale_case()
    case["options"].update(risk_weight=0.0, alpha=0.0)
    answer = run_candidate(case)
    validate(case, answer, 1.0)
    assert answer["root_objective"] == pytest.approx(1.0, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(1.0, abs=2e-5, rel=2e-6)


@pytest.mark.parametrize("weight,alpha", [(0.0, 0.0), (0.65, 0.73), (1.0, 0.5)])
def test_unused_expensive_capacity_cannot_erase_material_merit_order(weight, alpha):
    """“最小化 `V_root`。”

    At each hour the primal G1=1 attains one; every available supply/shedding
    price is at least one, proving the matching lower bound without an LP.
    Identical child costs make the exact two-hour risk objective two for all
    three declared risk regimes, despite an unused 1e13-price generator.
    """
    case = mixed_cost_scale_case()
    case["options"].update(risk_weight=weight, alpha=alpha)
    root = case["tree"][0]
    for name, q in (("a", 0.4), ("b", 0.6)):
        child = copy.deepcopy(root)
        child.update(id=name, parent="root", stage=1, probability=q)
        case["tree"].append(child)
    answer = run_candidate(case)
    validate(case, answer, 2.0)
    assert answer["root_objective"] == pytest.approx(2.0, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(2.0, abs=2e-5, rel=2e-6), (
        f"Risk-envelope LP must match the analytic primal/dual bound2, got {independent!r}"
    )


def test_tiny_load_with_material_cost_keeps_original_unit_balance():
    """“每母线：发电 + 放电 - 充电 + 削负荷 - 需求 = 向外线路净流出。”

    Demand1e-12 at generation price5e12 versus shedding price1e13 has exact
    optimum5. A zero primal residual in rescaled units cannot replace the
    original-unit balance or erase its material economic consequence.
    """
    case = analytic_network_case()
    case["network"]["generators"][0]["marginal_cost"] = 5e12
    case["tree"][0]["load"]["L"] = 1e-12
    case["options"]["shed_cost"]["L"] = 1e13
    result = run_candidate(case)
    validate(case, result, 5.0)
    assert result["node_policy"]["root"]["p"]["G"] * 1e12 == pytest.approx(
        1.0, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(5.0, abs=2e-5, rel=2e-6)


def test_mixed_cost_scale_respects_charge_and_terminal_energy():
    """“每个叶端独立满足最低终端储备。”

    The sole one-hour leaf must supply demand1 and charge1. G1 can supply
    exactly two at price1, while the unused unit costs1e13: primal and total
    energy lower bound both give objective2, with terminal energy exactly1.
    """
    case = mixed_cost_scale_case()
    for gen in case["network"]["generators"]:
        gen["p_nom"] = 2.0
    case["network"]["storage_units"] = [
        {"name": "S", "bus": "A", "p_nom": 1.0, "max_hours": 1.0,
         "efficiency_store": 1.0, "efficiency_dispatch": 1.0,
         "state_of_charge_initial": 0.0}
    ]
    case["options"]["terminal_soc"] = {"S": 1.0}
    case["options"]["throughput_cost"] = {"S": 0.0}
    result = run_candidate(case)
    validate(case, result, 2.0)
    assert result["node_policy"]["root"]["charge"]["S"] == pytest.approx(1, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(2.0, abs=2e-5, rel=2e-6)


def test_stiff_parallel_mesh_keeps_tiny_transfer_economically_material():
    """“f = (v_nom(bus0)**2/x)*(theta_bus0-theta_bus1), |f| <= s_nom”

    Parallel reactances1e-12/1e6 split flow in ratio1e18. The stiff line cap
    2e-13 admits total2e-13*(1+1e-18), leaving an analytic shedding cost8.
    Nodal rational transfer factors independently recover the same bound.
    """
    case = analytic_network_case(reactance=1e-12)
    case["network"]["lines"][0]["s_nom"] = 2e-13
    case["network"]["lines"].append(
        {"name": "parallel", "bus0": "A", "bus1": "B", "x": 1e6, "s_nom": 1e6})
    case["tree"][0]["line_status"]["parallel"] = True
    case["tree"][0]["load"]["L"] = 1e-12
    case["options"]["shed_cost"]["L"] = 1e13
    assert all(error <= Fraction(1, 5_000_000) for _, _, error in independent_angle_budgets(case))
    answer = run_candidate(case)
    validate(case, answer, 8.0)
    assert answer["node_policy"]["root"]["flow"]["AB"] * 1e13 == pytest.approx(2, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(8.0, abs=2e-5, rel=2e-6)


def test_pure_cvar_accepts_a_root_optimal_non_tail_policy():
    """“特别在 λ=1 时不额外要求非尾部分支各自条件最优。”

    A high branch needs10 but free generation is capped at1, proving root
    CVaR9. Shedding1 on the low branch stays outside the worst half mass and
    is equally root-optimal; the validator must accept that returned policy.
    """
    case = analytic_network_case(probability=0.5)
    case["options"].update(risk_weight=1.0, alpha=0.5)
    case["options"]["shed_cost"]["L"] = 1.0
    case["network"]["generators"][0].update(p_nom=1.0, marginal_cost=0.0)
    case["tree"][0]["load"]["L"] = 0.0
    case["tree"][1]["load"]["L"] = 10.0
    answer = run_candidate(case)
    validate(case, answer, 9.0)
    alternate = copy.deepcopy(answer)
    low = alternate["node_policy"]["usual"]
    low["p"]["G"], low["shed"]["L"] = 0.0, 1.0
    low["local_cost"], low["risk_to_go"] = 1.0, 1.0
    alternate["expected_cost"] = 5.0
    validate(case, alternate, 9.0)
    assert alternate["root_objective"] == pytest.approx(9.0, abs=2e-5, rel=2e-6)


@pytest.mark.parametrize("increment", [1.0, 0.75])
def test_large_soc_origin_cannot_erase_a_material_charging_requirement(increment):
    """“e = (1-standing_loss)*e_prev + efficiency_store*charge - discharge/efficiency_dispatch”

    SOC starts at1e10-1; terminal energy is higher by exactly1 or0.75.
    With no discharge and binary-exact efficiency2^-19, charge and generation
    are bounded below by increment*2^19, attaining that exact cost at price1.
    This checks a material increment on a large origin, including a reserve
    strictly inside the capacity boundary, independently of either LP solver.
    """
    initial = 1e10 - 1
    case = {
        "network": {
            "buses": [{"name": "A", "v_nom": 10}],
            "generators": [{"name": "G", "bus": "A", "p_nom": 1e6, "marginal_cost": 1}],
            "storage_units": [{"name": "S", "bus": "A", "p_nom": 1e6,
                "max_hours": 1e4, "efficiency_store": 2**-19,
                "efficiency_dispatch": 1.0, "p_max_pu": 0.0,
                "state_of_charge_initial": initial}],
        },
        "tree": [{"id": "r", "parent": None, "stage": 0, "probability": 1,
                  "load": {}, "renewable_pu": {"G": 1}, "line_status": {}}],
        "options": {"risk_weight": 0, "alpha": 0, "initial": {},
                    "terminal_soc": {"S": initial + increment}, "shed_cost": {},
                    "throughput_cost": {"S": 0}},
    }
    exact_cost = increment * 2**19
    answer = run_candidate(case)
    validate(case, answer, exact_cost)
    assert answer["node_policy"]["r"]["energy"]["S"] - initial == pytest.approx(
        increment, abs=2e-5, rel=2e-6)
    assert answer["node_policy"]["r"]["charge"]["S"] == pytest.approx(
        exact_cost, abs=2e-5, rel=2e-6)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(exact_cost, abs=2e-5, rel=2e-6)


def test_native_policy_preserves_public_network_state():
    """“函数不能改变传入的 network、tree 或 options”

    A real Network is constructed and observed by the isolated worker. The
    trusted parent compares all public component static/time-series tables,
    snapshots/weights and metadata, including NaN/index/dtype semantics. The
    policy must independently attain the same analytic unit-cost lower bound.
    """
    case = mixed_cost_scale_case()
    case["options"].update(risk_weight=0.0, alpha=0.0)
    diagnostic = candidate.native_policy(case)
    answer = json.loads(json.dumps(diagnostic["result"], allow_nan=False))
    validate(case, answer, 1.0)
    before, after = diagnostic["before"], diagnostic["after"]
    changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
    assert before == after, f"Native policy changed public Network state: {changed}"


def test_ordinary_down_ramp_has_an_attainable_eight_cost_lower_bound():
    """“p_prev - p <= P*(ramp_limit_down*u + ramp_limit_shut_down*(u_prev-u))”

    Initial and root commitment are on. Prior output10 and ordinary down limit2
    force thermal output at least8, although free renewable power could supply
    the entire10MW load. Thermal8 plus renewable2 attains the cost lower bound8.
    """
    case = {
        "network": {
            "buses": [{"name": "A", "v_nom": 10}],
            "generators": [
                {"name": "T", "bus": "A", "p_nom": 10, "committable": True,
                 "p_min_pu": 0, "marginal_cost": 1, "ramp_limit_down": 0.2},
                {"name": "R", "bus": "A", "p_nom": 10, "committable": False,
                 "marginal_cost": 0},
            ],
            "loads": [{"name": "L", "bus": "A"}],
        },
        "tree": [{"id": "r", "parent": None, "stage": 0, "probability": 1,
                  "load": {"L": 10}, "renewable_pu": {"R": 1}, "line_status": {}}],
        "options": {"risk_weight": 0, "alpha": 0,
                    "initial": {"T": {"status": 1, "p": 10, "hours": 3, "root_status": 1}},
                    "terminal_soc": {}, "shed_cost": {"L": 100}, "throughput_cost": {}},
    }
    answer = run_candidate(case)
    assert answer["status"] == "optimal"
    out = answer["node_policy"]["r"]
    assert 10 - out["p"]["T"] <= 2 + 2e-5, "ordinary down-ramp: previous10 minus current output must be<=2MW"
    assert out["p"]["T"] == pytest.approx(8, abs=2e-5, rel=2e-6)
    assert out["p"]["R"] == pytest.approx(2, abs=2e-5, rel=2e-6)
    assert answer["root_objective"] == pytest.approx(8, abs=2e-5, rel=2e-6)
    validate(case, answer, 8)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(8, abs=2e-5, rel=2e-6)


def test_active_zero_capacity_bridge_still_couples_island_angles():
    """“零容量在运线路的 w 为0，仍属于电气连通图。”

    Zero-capacity activeAC joins gaugeA to C while BC carries oneMW from C to B.
    Thus thetaC=thetaA=0 and thetaB=-0.01. DeletingAC incorrectly makes B a
    separate gauge, producing thetaC=0.01 despite the activeAC zero-flow law.
    G=1 has cost1 and attains the supply-price lower bound independently.
    """
    case = {
        "network": {
            "buses": [{"name": "A", "v_nom": 10}, {"name": "B", "v_nom": 10},
                      {"name": "C", "v_nom": 10}],
            "lines": [
                {"name": "AC", "bus0": "A", "bus1": "C", "x": 1, "s_nom": 0},
                {"name": "BC", "bus0": "B", "bus1": "C", "x": 1, "s_nom": 2},
            ],
            "generators": [{"name": "G", "bus": "C", "p_nom": 2,
                            "committable": False, "marginal_cost": 1}],
            "loads": [{"name": "L", "bus": "B"}],
        },
        "tree": [{"id": "r", "parent": None, "stage": 0, "probability": 1,
                  "load": {"L": 1}, "renewable_pu": {"G": 1},
                  "line_status": {"AC": True, "BC": True}}],
        "options": {"risk_weight": 0, "alpha": 0, "initial": {},
                    "terminal_soc": {}, "shed_cost": {"L": 10}, "throughput_cost": {}},
    }
    answer = run_candidate(case)
    assert answer["status"] == "optimal"
    out = answer["node_policy"]["r"]
    assert out["flow"]["AC"] == pytest.approx(
        100 * (out["theta"]["A"] - out["theta"]["C"]), abs=2e-5, rel=2e-6
    ), "active zero-capacity AC must retain its DC angle equation"
    assert out["theta"]["C"] == pytest.approx(0, abs=2e-5, rel=2e-6)
    assert out["theta"]["B"] == pytest.approx(-0.01, abs=2e-5, rel=2e-6)
    assert answer["root_objective"] == pytest.approx(1, abs=2e-5, rel=2e-6)
    validate(case, answer, 1)
    independent, _ = exhaustive_optimum(case)
    assert independent == pytest.approx(1, abs=2e-5, rel=2e-6)
