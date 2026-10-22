"""Author reference implementation for the public finite policy-tree contract."""

import math
import time
from fractions import Fraction
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from pypsa.optimization.policy_schema import normalize


class Model:
    def __init__(self):
        self.keys = {}
        self.lower = []
        self.upper = []
        self.integer = []
        self.scales = []
        self.rows = []
        self.lo = []
        self.hi = []
        self.economic = []
        self.physical_tolerances = []

    def var(self, key, low=0, high=np.inf, binary=False, scale=1.0):
        i = len(self.lower)
        self.keys[key] = i
        self.lower.append(low)
        self.upper.append(high)
        self.integer.append(int(binary))
        self.scales.append(float(scale) if scale > 0 else 1.0)
        return i

    def constraint(
        self, terms, low=-np.inf, high=np.inf, economic=False, physical_tolerance=None
    ):
        self.rows.append({i: value for i, value in terms.items() if value})
        self.lo.append(low)
        self.hi.append(high)
        self.economic.append(economic)
        self.physical_tolerances.append(physical_tolerance)

    def solve(self, root, time_limit=75, feasibility=False):
        r, c, data = [], [], []
        # Work in increments from finite lower bounds. A large stored-energy
        # origin must not hide a small, economically material change in SOC.
        offsets = np.array([lo if np.isfinite(lo) else 0.0 for lo in self.lower])
        scales = np.asarray(self.scales).copy()
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                scales[i] = min(scales[i], hi - lo)
        self.scales = list(scales)
        lower = np.asarray(self.lo, dtype=float).copy()
        upper = np.asarray(self.hi, dtype=float).copy()
        for j, row in enumerate(self.rows):
            if feasibility and self.economic[j]:
                lower[j], upper[j] = -np.inf, np.inf
                continue
            fixed = math.fsum(value * offsets[i] for i, value in row.items())
            lower[j] -= fixed
            upper[j] -= fixed
            transformed = {
                i: value * self.scales[i]
                for i, value in row.items()
                if self.lower[i] != self.upper[i]
            }
            row_scale = max(
                [abs(value) for value in transformed.values()]
                + [abs(value) for value in (lower[j], upper[j]) if np.isfinite(value)]
                + [0.0]
            )
            row_scale = row_scale if row_scale > 0 else 1.0
            lower[j] /= row_scale
            upper[j] /= row_scale
            for i, value in transformed.items():
                r.append(j)
                c.append(i)
                data.append(value / row_scale)
        mat = coo_matrix(
            (data, (r, c)), shape=(len(self.rows), len(self.lower))
        ).tocsc()
        obj = np.zeros(len(self.lower))
        obj[root] = 0 if feasibility else 1
        result = milp(
            obj,
            integrality=self.integer,
            bounds=Bounds(
                (np.asarray(self.lower) - offsets) / scales,
                (np.asarray(self.upper) - offsets) / scales,
            ),
            constraints=LinearConstraint(mat, lower, upper),
            options={
                "time_limit": max(0.01, time_limit),
                "mip_rel_gap": 1e-9,
                "presolve": True,
            },
        )
        if result.x is not None:
            result.x = result.x * scales + offsets
            for i, integer in enumerate(self.integer):
                if integer:
                    result.x[i] = round(result.x[i])
            result.fun = result.fun * scales[root]
        return result

    def physical_residuals_valid(self, x):
        if x is None or not np.all(np.isfinite(x)):
            return False
        for key, i in self.keys.items():
            if key[1] in {"cost", "risk", "eta", "excess"}:
                continue
            value = x[i]
            if value < self.lower[i] - 2e-7 or value > self.upper[i] + 2e-7:
                return False
            if self.integer[i] and abs(value - round(value)) > 1e-7:
                return False
        for row, lo, hi, economic, explicit_tolerance in zip(
            self.rows, self.lo, self.hi, self.economic, self.physical_tolerances
        ):
            if economic:
                continue
            terms = [coefficient * x[i] for i, coefficient in row.items()]
            value = math.fsum(terms)
            if explicit_tolerance is not None:
                tolerance = explicit_tolerance
            elif lo == hi:
                magnitude = max([abs(lo)] + [abs(term) for term in terms])
                tolerance = max(64 * math.ulp(magnitude), 1e-10 * min(1, abs(lo)))
            else:
                tolerance = 2e-7
            if value < lo - tolerance or value > hi + tolerance:
                return False
        return True


def _add(terms, i, value):
    terms[i] = terms.get(i, 0) + value


def _rate(row, name):
    value = float(row[name])
    return 1.0 if math.isnan(value) else value


def _tail(values, probabilities, alpha):
    left, total = 1 - alpha, 0.0
    for value, q in sorted(zip(values, probabilities), reverse=True):
        part = min(left, q)
        total += part * value
        left -= part
        if left <= 0:
            break
    return total / (1 - alpha)


def _cost_bounds(n, d, options):
    """Physical upper bounds set units; they do not truncate probabilities."""
    local, risk, quantiles = {}, {}, {}
    for a in d["ordered"]:
        demand = math.fsum(d["nodes"][a]["load"].values())
        charging = math.fsum(
            -float(n.storage_units.at[s, "p_min_pu"])
            * float(n.storage_units.at[s, "p_nom"])
            for s in d["stores"]
        )
        capacity = math.fsum(
            float(n.generators.at[g, "p_nom"])
            * float(n.generators.at[g, "p_max_pu"])
            * (d["nodes"][a]["renewable_pu"][g] if g in d["renewable"] else 1)
            for g in d["gens"]
        )
        price = max(
            [float(n.generators.at[g, "marginal_cost"]) for g in d["gens"]] + [0.0]
        )
        cost = min(capacity, demand + charging) * price
        cost += math.fsum(
            float(n.generators.at[g, "stand_by_cost"])
            + max(
                float(n.generators.at[g, "start_up_cost"]),
                float(n.generators.at[g, "shut_down_cost"]),
            )
            for g in d["thermal"]
        )
        cost += math.fsum(
            d["nodes"][a]["load"][component_name] * options["shed_cost"][component_name]
            for component_name in d["loads"]
        )
        cost += math.fsum(
            options["throughput_cost"][s]
            * float(n.storage_units.at[s, "p_nom"])
            * max(
                -float(n.storage_units.at[s, "p_min_pu"]),
                float(n.storage_units.at[s, "p_max_pu"]),
            )
            + float(n.storage_units.at[s, "marginal_cost"])
            * float(n.storage_units.at[s, "p_nom"])
            * float(n.storage_units.at[s, "p_max_pu"])
            for s in d["stores"]
        )
        local[a] = cost
    lam, alpha = options["risk_weight"], options["alpha"]
    for a in reversed(d["ordered"]):
        children = d["children"][a]
        values = [risk[c] for c in children]
        qs = [d["nodes"][c]["probability"] for c in children]
        risk[a] = local[a] + (
            (1 - lam) * math.fsum(q * v for q, v in zip(qs, values))
            + lam * _tail(values, qs, alpha)
            if children
            else 0
        )
        if children:
            accumulated = 0.0
            quantiles[a] = max(values)
            for value, q in sorted(zip(values, qs)):
                accumulated += q
                if accumulated >= alpha:
                    quantiles[a] = value
                    break
    return local, risk, quantiles


def _local_cost_terms(n, d, options, node):
    terms = {}
    for g in d["gens"]:
        terms[node, "p", g] = float(n.generators.at[g, "marginal_cost"])
        if g in d["thermal"]:
            for field, column in (
                ("u", "stand_by_cost"),
                ("start", "start_up_cost"),
                ("stop", "shut_down_cost"),
            ):
                terms[node, field, g] = float(n.generators.at[g, column])
    for component_name in d["loads"]:
        terms[node, "shed", component_name] = options["shed_cost"][component_name]
    for s in d["stores"]:
        fee = options["throughput_cost"][s]
        terms[node, "charge", s] = fee
        terms[node, "discharge", s] = fee + float(
            n.storage_units.at[s, "marginal_cost"]
        )
    return terms


def _actual_risks(model, x, n, d, options):
    costs, risks = {}, {}
    for node in reversed(d["ordered"]):
        costs[node] = math.fsum(
            coefficient * x[model.keys[key]]
            for key, coefficient in _local_cost_terms(n, d, options, node).items()
        )
        children = d["children"][node]
        values = [risks[c] for c in children]
        qs = [d["nodes"][c]["probability"] for c in children]
        risks[node] = costs[node]
        if children:
            risks[node] += (1 - options["risk_weight"]) * math.fsum(
                q * v for q, v in zip(qs, values)
            ) + options["risk_weight"] * _tail(values, qs, options["alpha"])
    return costs, risks


def _tighten_economic_bounds(model, x, n, d, options):
    """Use a feasible incumbent, not the most expensive available generator.

    Costs are nonnegative and each conditional risk dominates its expectation.
    Thus V(root) >= reach_probability(node)*V(node).  Division is exact rational
    arithmetic, so no positive probability or small-probability large loss is
    removed. The harmless outward margin covers incumbent serialization noise.
    Each paid physical action is bounded by this same local cost budget before
    rescaling. In particular, an unused expensive unit cannot erase cheap costs.
    """
    costs, risks = _actual_risks(model, x, n, d, options)
    if any(cost < 0 for cost in costs.values()):
        return None
    objective = risks[d["root"]]
    upper = Fraction(float(objective + 64 * math.ulp(objective)))
    reach, budgets = {}, {}
    for node in d["ordered"]:
        parent = d["nodes"][node]["parent"]
        if parent is None:
            reach[node] = Fraction(1)
        else:
            # Admission permits a 1e-10 sum error. Dividing by max(1,sum q)
            # makes this bound conservative also for such accepted inputs.
            total = sum(
                Fraction(float(d["nodes"][c]["probability"]))
                for c in d["children"][parent]
            )
            reach[node] = (
                reach[parent]
                * Fraction(float(d["nodes"][node]["probability"]))
                / max(Fraction(1), total)
            )
        budget = upper / reach[node]
        original = model.upper[model.keys[node, "risk"]]
        budgets[node] = float(min(Fraction(float(original)), budget))
        for field in ("cost", "risk"):
            i = model.keys[node, field]
            model.upper[i] = min(model.upper[i], budgets[node])
            model.scales[i] = model.upper[i] if model.upper[i] > 0 else 1.0
        for key, coefficient in _local_cost_terms(n, d, options, node).items():
            if coefficient <= 0:
                continue
            i = model.keys[key]
            cap = float(
                min(
                    Fraction(float(model.upper[i])),
                    budget / Fraction(float(coefficient)),
                )
            )
            model.upper[i] = cap
            if not model.integer[i] and cap > 0:
                model.scales[i] = cap
    for node in d["ordered"]:
        if (node, "eta") in model.keys:
            i = model.keys[node, "eta"]
            # lambda*eta is a nonnegative contribution to this parent's V.
            # A rare expensive child must not choose eta's numerical unit.
            model.upper[i] = min(
                model.upper[i],
                budgets[node] / options["risk_weight"],
                max(budgets[c] for c in d["children"][node]),
            )
            model.scales[i] = model.upper[i] if model.upper[i] > 0 else 1.0
            for child in d["children"][node]:
                i = model.keys[node, "excess", child]
                model.upper[i] = min(model.upper[i], budgets[child])
                model.scales[i] = model.upper[i] if model.upper[i] > 0 else 1.0
    return objective


def _electrical_forest(network, domain, node):
    """Fundamental cycles constrain flows; a forest later restores public angles.

    KCL plus these KVL rows is equivalent to the public DC equations. All
    buses connected by a line share the same voltage base, so v_nom**2
    cancels around each cycle. No small susceptance is discarded by design.
    """
    forest = {bus: [] for bus in domain["buses"]}
    leaders = {bus: bus for bus in domain["buses"]}

    def leader(bus):
        while leaders[bus] != bus:
            bus = leaders[bus]
        return bus

    chords = []
    for line in domain["lines"]:
        if not domain["nodes"][node]["line_status"][line]:
            continue
        row = network.lines.loc[line]
        first, second = leader(row.bus0), leader(row.bus1)
        if first == second:
            chords.append(line)
        else:
            leaders[second] = first
            forest[row.bus0].append((row.bus1, line, 1))
            forest[row.bus1].append((row.bus0, line, -1))
    cycles = []
    for chord in chords:
        row = network.lines.loc[chord]
        stack = [(row.bus1, [])]
        seen = {row.bus1}
        while stack:
            bus, path = stack.pop()
            if bus == row.bus0:
                cycles.append([(chord, 1)] + path)
                break
            for neighbor, line, sign in forest[bus]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append((neighbor, path + [(line, sign)]))
    return forest, cycles


def _public_angles(network, domain, forest, flow):
    """Round each exact accumulated tree drop only once into public binary64."""
    angles = {}
    for reference in domain["buses"]:
        if reference in angles:
            continue
        angles[reference] = Fraction(0)
        stack = [reference]
        while stack:
            bus = stack.pop()
            for neighbor, line, sign in forest[bus]:
                if neighbor in angles:
                    continue
                voltage = Fraction(float(network.buses.at[bus, "v_nom"]))
                resistance = Fraction(float(network.lines.at[line, "x"]))
                drop = Fraction(float(flow[line])) * resistance / voltage / voltage
                angles[neighbor] = angles[bus] - sign * drop
                stack.append(neighbor)
    return {bus: float(angle) for bus, angle in angles.items()}


def optimize_policy_tree(network, tree, options):
    started = time.monotonic()
    d = normalize(network, tree, options)
    n, m = network, Model()
    ix = m.keys
    root, order = d["root"], d["ordered"]
    nodes, kids = d["nodes"], d["children"]
    lam, alpha = options["risk_weight"], options["alpha"]
    local_bounds, risk_bounds, quantile_bounds = _cost_bounds(n, d, options)
    electrical = {a: _electrical_forest(n, d, a) for a in order}
    histories = {}
    for a in order:
        parent = nodes[a]["parent"]
        histories[a] = (histories[parent] if parent is not None else []) + [a]
        charging_capacity = math.fsum(
            -float(n.storage_units.at[s, "p_min_pu"])
            * float(n.storage_units.at[s, "p_nom"])
            for s in d["stores"]
        )
        generation_budget = math.fsum(nodes[a]["load"].values()) + charging_capacity
        transfer_budget = generation_budget + math.fsum(
            float(n.storage_units.at[s, "p_max_pu"])
            * float(n.storage_units.at[s, "p_nom"])
            for s in d["stores"]
        )
        for component_name in d["lines"]:
            cap = (
                float(n.lines.at[component_name, "s_nom"])
                if nodes[a]["line_status"][component_name]
                else 0
            )
            cap = min(cap, transfer_budget)
            m.var((a, "flow", component_name), -cap, cap, scale=cap)
        for g in d["gens"]:
            row = n.generators.loc[g]
            cap = float(row.p_nom * row.p_max_pu)
            if g in d["renewable"]:
                cap *= nodes[a]["renewable_pu"][g]
            cap = min(cap, generation_budget)
            m.var((a, "p", g), 0, cap, scale=cap)
        for g in d["thermal"]:
            for field in ("u", "start", "stop"):
                m.var((a, field, g), 0, 1, binary=True)
        for component_name in d["loads"]:
            cap = nodes[a]["load"][component_name]
            m.var((a, "shed", component_name), 0, cap, scale=cap)
        for s in d["stores"]:
            row = n.storage_units.loc[s]
            for field, cap in (
                ("charge", float(-row.p_nom * row.p_min_pu)),
                ("discharge", float(row.p_nom * row.p_max_pu)),
                ("energy", float(row.p_nom * row.max_hours)),
            ):
                low = (
                    options["terminal_soc"][s]
                    if field == "energy" and not kids[a]
                    else 0
                )
                m.var((a, field, s), low, cap, scale=cap)
            m.var((a, "charge_mode", s), 0, 1, binary=True)
        m.var((a, "cost"), high=local_bounds[a], scale=local_bounds[a])
        m.var((a, "risk"), high=risk_bounds[a], scale=risk_bounds[a])
        if kids[a] and lam > 0 and alpha > 0:
            m.var((a, "eta"), high=quantile_bounds[a], scale=quantile_bounds[a])
            for child in kids[a]:
                m.var(
                    (a, "excess", child),
                    high=risk_bounds[child],
                    scale=risk_bounds[child],
                )
    for a in order:
        parent = nodes[a]["parent"]
        for g in d["thermal"]:
            row = n.generators.loc[g]
            h = options["initial"][g]
            u, start, stop, p = [ix[a, k, g] for k in ("u", "start", "stop", "p")]
            P = float(row.p_nom)
            terms = {u: 1, start: -1, stop: 1}
            if parent is None:
                m.constraint(terms, h["status"], h["status"])
                m.constraint({u: 1}, h["root_status"], h["root_status"])
            else:
                terms[ix[parent, "u", g]] = -1
                m.constraint(terms, 0, 0)
            m.constraint({start: 1, stop: 1}, high=1)
            m.constraint({p: 1, u: -P * float(row.p_max_pu)}, high=0)
            m.constraint({p: 1, u: -P * float(row.p_min_pu)}, low=0)
            ru, rd = _rate(row, "ramp_limit_up"), _rate(row, "ramp_limit_down")
            rsu, rsd = (
                _rate(row, "ramp_limit_start_up"),
                _rate(row, "ramp_limit_shut_down"),
            )
            up = {p: 1, u: -P * rsu}
            down = {p: -1, u: -P * (rd - rsd)}
            if parent is None:
                m.constraint(up, high=h["p"] + P * (ru - rsu) * h["status"])
                m.constraint(down, high=-h["p"] + P * rsd * h["status"])
            else:
                up[ix[parent, "p", g]] = -1
                up[ix[parent, "u", g]] = -P * (ru - rsu)
                down[ix[parent, "p", g]] = 1
                down[ix[parent, "u", g]] = -P * rsd
                m.constraint(up, high=0)
                m.constraint(down, high=0)
            for duration, event, sign, bound in (
                (int(row.min_up_time), "start", -1, 0),
                (int(row.min_down_time), "stop", 1, 1),
            ):
                terms = {u: sign}
                if duration > 0:
                    for old in histories[a][-duration:]:
                        _add(terms, ix[old, event, g], 1)
                    m.constraint(terms, high=bound)
            initial_min = int(row.min_up_time if h["status"] else row.min_down_time)
            if nodes[a]["stage"] < max(0, initial_min - h["hours"]):
                m.constraint({u: 1}, h["status"], h["status"])
            if len(kids[a]) > 1:
                first = kids[a][0]
                for child in kids[a][1:]:
                    m.constraint({ix[first, "u", g]: 1, ix[child, "u", g]: -1}, 0, 0)
        for s in d["stores"]:
            row = n.storage_units.loc[s]
            charge, discharge, energy, mode = [
                ix[a, k, s] for k in ("charge", "discharge", "energy", "charge_mode")
            ]
            Pc, Pd = float(-row.p_nom * row.p_min_pu), float(row.p_nom * row.p_max_pu)
            m.constraint({charge: 1, mode: -Pc}, high=0)
            m.constraint({discharge: 1, mode: Pd}, high=Pd)
            terms = {
                energy: 1,
                charge: -float(row.efficiency_store),
                discharge: 1 / float(row.efficiency_dispatch),
            }
            loss = 1 - float(row.standing_loss)
            if parent is None:
                rhs = loss * float(row.state_of_charge_initial)
            else:
                terms[ix[parent, "energy", s]] = -loss
                rhs = 0
            m.constraint(terms, rhs, rhs)
            if not kids[a]:
                m.constraint({energy: 1}, low=options["terminal_soc"][s])
        for cycle in electrical[a][1]:
            max_x = max(Fraction(float(n.lines.at[line, "x"])) for line, _ in cycle)
            terms = {
                ix[a, "flow", line]: sign
                * float(Fraction(float(n.lines.at[line, "x"])) / max_x)
                for line, sign in cycle
            }
            m.constraint(
                terms,
                0,
                0,
                physical_tolerance=2e-7 * min(abs(value) for value in terms.values()),
            )
        for b in d["buses"]:
            terms, demand = {}, 0.0
            for g in d["gens"]:
                if n.generators.at[g, "bus"] == b:
                    _add(terms, ix[a, "p", g], 1)
            for component_name in d["loads"]:
                if n.loads.at[component_name, "bus"] == b:
                    _add(terms, ix[a, "shed", component_name], 1)
                    demand += nodes[a]["load"][component_name]
            for s in d["stores"]:
                if n.storage_units.at[s, "bus"] == b:
                    _add(terms, ix[a, "discharge", s], 1)
                    _add(terms, ix[a, "charge", s], -1)
            for component_name in d["lines"]:
                if n.lines.at[component_name, "bus0"] == b:
                    _add(terms, ix[a, "flow", component_name], -1)
                if n.lines.at[component_name, "bus1"] == b:
                    _add(terms, ix[a, "flow", component_name], 1)
            m.constraint(terms, demand, demand)
        terms = {ix[a, "cost"]: 1}
        for g in d["gens"]:
            _add(terms, ix[a, "p", g], -float(n.generators.at[g, "marginal_cost"]))
            if g in d["thermal"]:
                for field, column in (
                    ("u", "stand_by_cost"),
                    ("start", "start_up_cost"),
                    ("stop", "shut_down_cost"),
                ):
                    _add(terms, ix[a, field, g], -float(n.generators.at[g, column]))
        for component_name in d["loads"]:
            _add(
                terms,
                ix[a, "shed", component_name],
                -options["shed_cost"][component_name],
            )
        for s in d["stores"]:
            fee = options["throughput_cost"][s]
            _add(terms, ix[a, "charge", s], -fee)
            _add(
                terms,
                ix[a, "discharge", s],
                -fee - float(n.storage_units.at[s, "marginal_cost"]),
            )
        m.constraint(terms, 0, 0, economic=True)
        terms = {ix[a, "risk"]: 1, ix[a, "cost"]: -1}
        if kids[a]:
            if lam == 0 or alpha == 0:
                for child in kids[a]:
                    _add(terms, ix[child, "risk"], -nodes[child]["probability"])
            else:
                _add(terms, ix[a, "eta"], -lam)
                for child in kids[a]:
                    q = nodes[child]["probability"]
                    _add(terms, ix[child, "risk"], -(1 - lam) * q)
                    _add(terms, ix[a, "excess", child], -lam * q / (1 - alpha))
                    m.constraint(
                        {
                            ix[a, "excess", child]: 1,
                            ix[child, "risk"]: -1,
                            ix[a, "eta"]: 1,
                        },
                        low=0,
                        economic=True,
                    )
            m.constraint(terms, low=0, economic=True)
        else:
            m.constraint(terms, 0, 0, economic=True)
    result = m.solve(ix[root, "risk"], time_limit=75 - (time.monotonic() - started))
    if not m.physical_residuals_valid(result.x):
        # Economic scaling cannot be used as evidence of physical infeasibility.
        result = m.solve(
            ix[root, "risk"],
            feasibility=True,
            time_limit=75 - (time.monotonic() - started),
        )
        if result.status == 2 and "infeasible" in result.message.lower():
            return {"status": "infeasible"}
        if not m.physical_residuals_valid(result.x):
            return {"status": "limit" if result.status == 1 else "error"}
    proved = False
    for _ in range(4):
        if time.monotonic() - started >= 75:
            return {"status": "limit"}
        incumbent = _tighten_economic_bounds(m, result.x, n, d, options)
        if incumbent is None or incumbent < 0:
            return {"status": "error"}
        result = m.solve(ix[root, "risk"], time_limit=75 - (time.monotonic() - started))
        if result.status != 0 or not m.physical_residuals_valid(result.x):
            return {"status": "limit" if result.status == 1 else "error"}
        _, actual_risks = _actual_risks(m, result.x, n, d, options)
        actual = actual_risks[root]
        dual = getattr(result, "mip_dual_bound", None)
        lower_bound = (
            result.fun if dual is None else float(dual) * m.scales[ix[root, "risk"]]
        )
        tolerance = 2e-6 + 2e-7 * abs(actual)
        if (
            abs(actual - result.fun) <= tolerance
            and abs(actual - lower_bound) <= tolerance
        ):
            proved = True
            break
    if not proved:
        return {"status": "error"}
    x, output, carry = result.x, {}, {}

    def val(a, field, name):
        v = float(x[ix[a, field, name]])
        return v

    for a in order:
        record = {
            "u": {},
            "start": {},
            "stop": {},
            "p": {},
            "charge": {},
            "discharge": {},
            "energy": {},
            "shed": {},
            "flow": {},
            "theta": {},
        }
        for field in ("u", "start", "stop"):
            record[field] = {g: int(round(val(a, field, g))) for g in d["thermal"]}
        for field, members in (
            ("p", d["gens"]),
            ("charge", d["stores"]),
            ("discharge", d["stores"]),
            ("energy", d["stores"]),
            ("shed", d["loads"]),
            ("flow", d["lines"]),
        ):
            record[field] = {key: val(a, field, key) for key in members}
        record["theta"] = _public_angles(n, d, electrical[a][0], record["flow"])
        for line in d["lines"]:
            if nodes[a]["line_status"][line]:
                row = n.lines.loc[line]
                predicted = (
                    float(n.buses.at[row.bus0, "v_nom"]) ** 2
                    / float(row.x)
                    * (record["theta"][row.bus0] - record["theta"][row.bus1])
                )
                if not math.isclose(
                    record["flow"][line], predicted, abs_tol=2e-7, rel_tol=2e-8
                ):
                    return {"status": "error"}
        cost = sum(
            record["p"][g] * float(n.generators.at[g, "marginal_cost"])
            for g in d["gens"]
        )
        for g in d["thermal"]:
            cost += sum(
                record[field][g] * float(n.generators.at[g, column])
                for field, column in (
                    ("u", "stand_by_cost"),
                    ("start", "start_up_cost"),
                    ("stop", "shut_down_cost"),
                )
            )
        cost += sum(
            record["shed"][component_name] * options["shed_cost"][component_name]
            for component_name in d["loads"]
        )
        cost += sum(
            options["throughput_cost"][s]
            * (record["charge"][s] + record["discharge"][s])
            + float(n.storage_units.at[s, "marginal_cost"]) * record["discharge"][s]
            for s in d["stores"]
        )
        record["local_cost"] = float(cost)
        output[a] = record
        if a in d["leaves"]:
            thermal_state = {}
            for g in d["thermal"]:
                h, path = options["initial"][g], histories[a]
                status = record["u"][g]
                count = 0
                for old in reversed(path):
                    if output[old]["u"][g] != status:
                        break
                    count += 1
                if count == len(path) and h["status"] == status:
                    count += h["hours"]
                minimum = int(
                    n.generators.at[g, "min_up_time" if status else "min_down_time"]
                )
                thermal_state[g] = {
                    "status": status,
                    "p": record["p"][g],
                    "hours": count,
                    "remaining": max(0, minimum - count),
                }
            carry[a] = {"generators": thermal_state, "energy": dict(record["energy"])}
    expected = {}
    for a in reversed(order):
        values = [output[c]["risk_to_go"] for c in kids[a]]
        probabilities = [nodes[c]["probability"] for c in kids[a]]
        risk = output[a]["local_cost"]
        if values:
            risk += (1 - lam) * sum(
                q * v for q, v in zip(probabilities, values)
            ) + lam * _tail(values, probabilities, alpha)
        output[a]["risk_to_go"] = float(risk)
        expected[a] = output[a]["local_cost"] + sum(
            nodes[c]["probability"] * expected[c] for c in kids[a]
        )
    return {
        "status": "optimal",
        "root_objective": output[root]["risk_to_go"],
        "expected_cost": float(expected[root]),
        "node_policy": output,
        "leaf_carry_state": carry,
    }
