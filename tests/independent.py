"""Independent small-case optimum: enumerate physical discrete policies, solve LPs.

This works from JSON records, not PyPSA/reference functions. Conditional risk
uses capped-simplex risk-envelope vertices, not the reference eta/excess form.
Both numerical paths use HiGHS, so solver independence is not claimed.
"""

from itertools import product
from fractions import Fraction
import math
import numpy as np
from scipy.optimize import linprog


def dc_transfer_factors(buses, lines, node):
    """Exact reduced nodal Laplacian, independent of the reference cycle model.

    Each column injects one MW at a bus and withdraws it at its island gauge.
    Rational elimination avoids deleting a stiff-line coefficient. Voltage
    cancels because every connected line has the same voltage base.
    """
    graph = {b: set() for b in buses}
    active = {key: row for key, row in lines.items() if node["line_status"][key]}
    for row in active.values():
        graph[row["bus0"]].add(row["bus1"])
        graph[row["bus1"]].add(row["bus0"])
    factors = {key: {} for key in active}
    unseen = set(buses)
    while unseen:
        gauge = min(unseen)
        component, stack = {gauge}, [gauge]
        unseen.remove(gauge)
        while stack:
            for neighbor in graph[stack.pop()]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        free = sorted(component - {gauge})
        size = len(free)
        index = {b: i for i, b in enumerate(free)}
        matrix = [[Fraction(0) for _ in range(2 * size)] for _ in free]
        for i in range(size):
            matrix[i][size + i] = Fraction(1)
        for row in active.values():
            if row["bus0"] not in component:
                continue
            conductance = 1 / Fraction(float(row["x"]))
            for first, second in (
                (row["bus0"], row["bus1"]),
                (row["bus1"], row["bus0"]),
            ):
                if first != gauge:
                    i = index[first]
                    matrix[i][i] += conductance
                    if second != gauge:
                        matrix[i][index[second]] -= conductance
        for col in range(size):
            pivot = next(i for i in range(col, size) if matrix[i][col])
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            divisor = matrix[col][col]
            matrix[col] = [value / divisor for value in matrix[col]]
            for i in range(size):
                if i != col and matrix[i][col]:
                    factor = matrix[i][col]
                    matrix[i] = [x - factor * y for x, y in zip(matrix[i], matrix[col])]
        for key, row in active.items():
            if row["bus0"] not in component:
                continue
            for b in free:
                col = size + index[b]
                first = matrix[index[row["bus0"]]][col] if row["bus0"] != gauge else 0
                second = matrix[index[row["bus1"]]][col] if row["bus1"] != gauge else 0
                factors[key][b] = float((first - second) / Fraction(float(row["x"])))
    return factors


def risk_weights(probabilities, alpha, lam):
    q = np.array(probabilities, dtype=float)
    if len(q) == 1 or alpha == 0 or lam == 0:
        return [q]
    exact_q = [Fraction(float(value)) for value in probabilities]
    cap = [value / Fraction(1 - alpha) for value in exact_q]
    vertices = []
    for pivot in range(len(q)):
        other = [i for i in range(len(q)) if i != pivot]
        for flags in product((0, 1), repeat=len(other)):
            w = [Fraction(0) for _ in q]
            for i, flag in zip(other, flags):
                w[i] = cap[i] * flag
            w[pivot] = 1 - sum(w)
            if 0 <= w[pivot] <= cap[pivot]:
                vertices.append(
                    np.array(
                        [
                            float((1 - Fraction(lam)) * qq + Fraction(lam) * ww)
                            for qq, ww in zip(exact_q, w)
                        ]
                    )
                )
    return vertices


def bounded_risk_lp(
    names,
    bounds,
    equations,
    rhs,
    limits,
    ub,
    local_forms,
    nodes,
    order,
    kids,
    options,
    upper_risk,
):
    """Enumerated-policy LP with an incumbent-based economic unit system.

    This uses nodal transfer factors and risk-envelope vertices, not the
    reference MILP, event binaries, cycle rows, or eta/excess representation.
    A solver objective is checked against actual actions before it is trusted.
    """
    size, root = len(bounds), order[0]
    if any(lo is not None and hi is not None and lo > hi for lo, hi in bounds):
        return math.inf
    equality, inequality = np.array(equations), np.array(limits)
    base_rhs, base_ub = np.array(rhs, dtype=float), np.array(ub, dtype=float)
    vertices = {
        a: risk_weights(
            [nodes[c]["probability"] for c in kids[a]],
            options["alpha"],
            options["risk_weight"],
        )
        if kids[a]
        else [[]]
        for a in order
    }
    current = list(bounds)
    risk_units = dict(upper_risk)
    for attempt in range(5):
        scales = np.ones(size)
        offsets = np.array([lo if lo is not None else 0.0 for lo, _ in current])
        for key, i in names.items():
            lo, hi = current[i]
            if key[1] == "V":
                scales[i] = risk_units[key[0]] if risk_units[key[0]] > 0 else 1
            elif lo is not None and hi is not None and max(abs(lo), abs(hi)) > 0:
                scales[i] = max(abs(lo), abs(hi))
            if lo is not None and hi is not None and hi > lo:
                scales[i] = min(scales[i], hi - lo)
        equality_rhs, inequality_rhs = base_rhs.copy(), base_ub.copy()
        eqmat, ineqmat = equality.copy(), inequality.copy()
        fixed = [i for i, (lo, hi) in enumerate(current) if lo is not None and lo == hi]
        equality_rhs -= np.array(
            [
                math.fsum(value * offset for value, offset in zip(row, offsets))
                for row in eqmat
            ]
        )
        inequality_rhs -= np.array(
            [
                math.fsum(value * offset for value, offset in zip(row, offsets))
                for row in ineqmat
            ]
        )
        if fixed:
            eqmat[:, fixed] = 0
            ineqmat[:, fixed] = 0
        eqmat *= scales
        ineqmat *= scales
        eq_scale = np.maximum(np.max(np.abs(eqmat), axis=1), np.abs(equality_rhs))
        ineq_scale = np.maximum(np.max(np.abs(ineqmat), axis=1), np.abs(inequality_rhs))
        eq_scale[eq_scale == 0] = 1
        ineq_scale[ineq_scale == 0] = 1
        target = np.zeros(size)
        target[names[root, "V"]] = 1
        result = linprog(
            target,
            A_ub=ineqmat / ineq_scale[:, None],
            b_ub=inequality_rhs / ineq_scale,
            A_eq=eqmat / eq_scale[:, None],
            b_eq=equality_rhs / eq_scale,
            bounds=[
                (
                    (lo - offsets[i]) / scales[i] if lo is not None else None,
                    (hi - offsets[i]) / scales[i] if hi is not None else None,
                )
                for i, (lo, hi) in enumerate(current)
            ],
            method="highs",
        )
        if not result.success:
            if (
                result.status == 2
                and "infeasible" in result.message.lower()
                and "model error" not in result.message.lower()
            ):
                return math.inf
            raise RuntimeError(
                "Independent LP did not establish a result: " + result.message
            )
        point = result.x * scales + offsets
        risk = {}
        for a in reversed(order):
            local, constant = local_forms[a]
            cost = constant + math.fsum(
                coefficient * point[names[key]] for key, coefficient in local.items()
            )
            risk[a] = cost + max(
                math.fsum(w * risk[c] for w, c in zip(vertex, kids[a]))
                for vertex in vertices[a]
            )
        actual, modeled = risk[root], result.fun * scales[names[root, "V"]]
        if attempt and abs(actual - modeled) <= 2e-6 + 2e-7 * abs(actual):
            return float(actual)
        incumbent = Fraction(float(actual + max(1e-7, 64 * math.ulp(actual))))
        reach = {}
        for a in order:
            parent = nodes[a]["parent"]
            if parent is None:
                reach[a] = Fraction(1)
            else:
                total = sum(
                    Fraction(float(nodes[c]["probability"])) for c in kids[parent]
                )
                reach[a] = (
                    reach[parent]
                    * Fraction(float(nodes[a]["probability"]))
                    / max(1, total)
                )
            budget = incumbent / reach[a]
            cap = float(min(Fraction(float(upper_risk[a])), budget))
            risk_units[a] = cap
            current[names[a, "V"]] = (0, cap)
            local, _ = local_forms[a]
            for key, coefficient in local.items():
                if coefficient > 0:
                    i = names[key]
                    lo, hi = current[i]
                    current[i] = (
                        lo,
                        float(
                            min(
                                Fraction(float(hi)),
                                budget / Fraction(float(coefficient)),
                            )
                        ),
                    )
    raise RuntimeError(
        "Independent LP objective disagrees with its actual physical policy"
    )


def exhaustive_optimum(case):
    components = case["network"]
    nodes = {n["id"]: n for n in case["tree"]}
    order = sorted(nodes, key=lambda key: (nodes[key]["stage"], key))
    kids = {a: [b for b in order if nodes[b]["parent"] == a] for a in order}
    rows = {
        key: {r["name"]: r for r in components.get(key, [])}
        for key in ("buses", "lines", "generators", "loads", "storage_units")
    }
    buses, lines, gens, loads, stores = [
        rows[key] for key in ("buses", "lines", "generators", "loads", "storage_units")
    ]
    thermal = [g for g, r in gens.items() if r.get("committable", False)]
    decisions = [(a, g) for a in order if kids[a] for g in thermal]
    hist = {}
    for a in order:
        parent = nodes[a]["parent"]
        hist[a] = (hist[parent] if parent is not None else []) + [a]
    options = case["options"]
    transfer = {a: dc_transfer_factors(buses, lines, nodes[a]) for a in order}
    best = np.inf
    feasible_schedules = 0
    for bits in product((0, 1), repeat=len(decisions)):
        commands = dict(zip(decisions, bits))
        u, events, valid = {}, {}, True
        for a in order:
            for g in thermal:
                prior = options["initial"][g]
                parent = nodes[a]["parent"]
                u[a, g] = (
                    prior["root_status"] if parent is None else commands[parent, g]
                )
                previous = prior["status"] if parent is None else u[parent, g]
                events[a, g] = (max(0, u[a, g] - previous), max(0, previous - u[a, g]))
                duration = 0
                for old in reversed(hist[a][:-1]):
                    if u[old, g] != previous:
                        break
                    duration += 1
                if duration == len(hist[a]) - 1 and previous == prior["status"]:
                    duration += prior["hours"]
                minimum = gens[g].get("min_up_time" if previous else "min_down_time", 0)
                if u[a, g] != previous and duration < minimum:
                    valid = False
        if not valid:
            continue
        feasible_schedules += 1
        for modes in product((0, 1, 2), repeat=len(order) * len(stores)):
            storage_mode = dict(zip(product(order, stores), modes))
            names, bounds = {}, []

            def variable(key, lo=0, hi=None):
                names[key] = len(bounds)
                bounds.append((lo, hi))

            for a in order:
                generation_budget = math.fsum(nodes[a]["load"].values()) + math.fsum(
                    -r.get("p_min_pu", -1) * r["p_nom"] for r in stores.values()
                )
                for g, r in gens.items():
                    if g in thermal:
                        lo = r.get("p_min_pu", 0) * r["p_nom"] * u[a, g]
                        hi = r.get("p_max_pu", 1) * r["p_nom"] * u[a, g]
                    else:
                        lo = 0
                        hi = (
                            r.get("p_max_pu", 1)
                            * r["p_nom"]
                            * nodes[a]["renewable_pu"][g]
                        )
                    variable((a, "p", g), lo, min(hi, generation_budget))
                for s, r in stores.items():
                    mode = storage_mode[a, s]
                    variable(
                        (a, "c", s),
                        0,
                        -r.get("p_min_pu", -1) * r["p_nom"] if mode == 1 else 0,
                    )
                    variable(
                        (a, "d", s),
                        0,
                        r.get("p_max_pu", 1) * r["p_nom"] if mode == 2 else 0,
                    )
                    lower = options["terminal_soc"][s] if not kids[a] else 0
                    variable((a, "e", s), lower, r["p_nom"] * r.get("max_hours", 1))
                for component_name in loads:
                    variable(
                        (a, "shed", component_name), 0, nodes[a]["load"][component_name]
                    )
                for component_name, r in lines.items():
                    cap = r["s_nom"] if nodes[a]["line_status"][component_name] else 0
                    transfer_budget = generation_budget + math.fsum(
                        row.get("p_max_pu", 1) * row["p_nom"] for row in stores.values()
                    )
                    cap = min(cap, transfer_budget)
                    variable((a, "f", component_name), -cap, cap)
                variable((a, "V"))
            size = len(bounds)
            equations, rhs, limits, ub = [], [], [], []
            local_upper = {}
            local_forms = {}

            def eq(terms, value):
                row = np.zeros(size)
                for key, x in terms.items():
                    row[names[key]] += x
                equations.append(row)
                rhs.append(value)

            def le(terms, value):
                row = np.zeros(size)
                for key, x in terms.items():
                    row[names[key]] += x
                limits.append(row)
                ub.append(value)

            for a in order:
                parent = nodes[a]["parent"]
                for g in thermal:
                    r, h = gens[g], options["initial"][g]
                    P = r["p_nom"]
                    prior_u = h["status"] if parent is None else u[parent, g]
                    up = P * (
                        r.get("ramp_limit_up", 1) * prior_u
                        + r.get("ramp_limit_start_up", 1) * (u[a, g] - prior_u)
                    )
                    down = P * (
                        r.get("ramp_limit_down", 1) * u[a, g]
                        + r.get("ramp_limit_shut_down", 1) * (prior_u - u[a, g])
                    )
                    terms = {(a, "p", g): 1}
                    if parent is None:
                        le(terms, up + h["p"])
                        le({k: -v for k, v in terms.items()}, down - h["p"])
                    else:
                        terms[parent, "p", g] = -1
                        le(terms, up)
                        le({k: -v for k, v in terms.items()}, down)
                for s, r in stores.items():
                    terms = {
                        (a, "e", s): 1,
                        (a, "c", s): -r.get("efficiency_store", 1),
                        (a, "d", s): 1 / r.get("efficiency_dispatch", 1),
                    }
                    decay = 1 - r.get("standing_loss", 0)
                    if parent is None:
                        eq(terms, decay * r.get("state_of_charge_initial", 0))
                    else:
                        terms[parent, "e", s] = -decay
                        eq(terms, 0)
                # Dense rational nodal elimination, not a cycle/angle LP.
                for component_name, weights in transfer[a].items():
                    terms = {(a, "f", component_name): 1}
                    demand = 0.0
                    for b, weight in weights.items():
                        for g, r in gens.items():
                            if r["bus"] == b:
                                terms[a, "p", g] = -weight
                        for s, r in stores.items():
                            if r["bus"] == b:
                                terms[a, "d", s] = -weight
                                terms[a, "c", s] = weight
                        for load, r in loads.items():
                            if r["bus"] == b:
                                terms[a, "shed", load] = -weight
                                demand -= weight * nodes[a]["load"][load]
                    eq(terms, demand)
                for b in buses:
                    terms, demand = {}, 0
                    for g, r in gens.items():
                        if r["bus"] == b:
                            terms[a, "p", g] = 1
                    for component_name, r in loads.items():
                        if r["bus"] == b:
                            terms[a, "shed", component_name] = 1
                            demand += nodes[a]["load"][component_name]
                    for s, r in stores.items():
                        if r["bus"] == b:
                            terms[a, "d", s] = 1
                            terms[a, "c", s] = -1
                    for component_name, r in lines.items():
                        if r["bus0"] == b:
                            terms[a, "f", component_name] = -1
                        if r["bus1"] == b:
                            terms[a, "f", component_name] = 1
                    eq(terms, demand)
                # Exact local cost plus each risk-envelope extreme expectation.
                local, constant = {}, 0
                for g, r in gens.items():
                    local[a, "p", g] = r.get("marginal_cost", 0)
                    if g in thermal:
                        su, sd = events[a, g]
                        constant += (
                            r.get("stand_by_cost", 0) * u[a, g]
                            + r.get("start_up_cost", 0) * su
                            + r.get("shut_down_cost", 0) * sd
                        )
                for component_name in loads:
                    local[a, "shed", component_name] = options["shed_cost"][
                        component_name
                    ]
                for s, r in stores.items():
                    local[a, "c", s] = options["throughput_cost"][s]
                    local[a, "d", s] = options["throughput_cost"][s] + r.get(
                        "marginal_cost", 0
                    )
                local_upper[a] = constant + math.fsum(
                    coefficient * bounds[names[key]][1]
                    for key, coefficient in local.items()
                )
                local_forms[a] = (dict(local), constant)
                weights = (
                    risk_weights(
                        [nodes[c]["probability"] for c in kids[a]],
                        options["alpha"],
                        options["risk_weight"],
                    )
                    if kids[a]
                    else [[]]
                )
                for weights_row in weights:
                    terms = dict(local)
                    terms[a, "V"] = -1
                    for c, weight in zip(kids[a], weights_row):
                        terms[c, "V"] = float(weight)
                    le(terms, -constant)
            upper_risk = {}
            for a in reversed(order):
                vertices = (
                    risk_weights(
                        [nodes[c]["probability"] for c in kids[a]],
                        options["alpha"],
                        options["risk_weight"],
                    )
                    if kids[a]
                    else [[]]
                )
                upper_risk[a] = local_upper[a] + max(
                    math.fsum(
                        weight * upper_risk[c] for c, weight in zip(kids[a], vertex)
                    )
                    for vertex in vertices
                )
            value = bounded_risk_lp(
                names,
                bounds,
                equations,
                rhs,
                limits,
                ub,
                local_forms,
                nodes,
                order,
                kids,
                options,
                upper_risk,
            )
            best = min(best, value)
    return float(best), feasible_schedules
