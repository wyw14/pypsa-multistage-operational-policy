"""Author reference implementation for the public finite policy-tree contract."""
import math
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
        self.rows = []
        self.lo = []
        self.hi = []

    def var(self, key, low=0, high=np.inf, binary=False):
        i = len(self.lower)
        self.keys[key] = i
        self.lower.append(low)
        self.upper.append(high)
        self.integer.append(int(binary))
        return i

    def constraint(self, terms, low=-np.inf, high=np.inf):
        self.rows.append({i: value for i, value in terms.items() if value})
        self.lo.append(low)
        self.hi.append(high)

    def solve(self, root):
        r, c, data = [], [], []
        for j, row in enumerate(self.rows):
            for i, value in row.items():
                r.append(j)
                c.append(i)
                data.append(value)
        mat = coo_matrix((data, (r, c)), shape=(len(self.rows), len(self.lower))).tocsc()
        obj = np.zeros(len(self.lower))
        obj[root] = 1
        return milp(obj, integrality=self.integer, bounds=Bounds(self.lower, self.upper),
                    constraints=LinearConstraint(mat, self.lo, self.hi),
                    options={"time_limit": 75, "mip_rel_gap": 1e-9, "presolve": True})


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
        if left <= 1e-13:
            break
    return total / (1 - alpha)


def optimize_policy_tree(network, tree, options):
    d = normalize(network, tree, options)
    n, m = network, Model()
    ix = m.keys
    root, order = d["root"], d["ordered"]
    nodes, kids = d["nodes"], d["children"]
    lam, alpha = options["risk_weight"], options["alpha"]
    histories = {}
    for a in order:
        parent = nodes[a]["parent"]
        histories[a] = (histories[parent] if parent is not None else []) + [a]
        for b in d["buses"]:
            m.var((a, "theta", b), -np.inf)
        for l in d["lines"]:
            cap = float(n.lines.at[l, "s_nom"]) if nodes[a]["line_status"][l] else 0
            m.var((a, "flow", l), -cap, cap)
        for g in d["gens"]:
            row = n.generators.loc[g]
            cap = float(row.p_nom * row.p_max_pu)
            if g in d["renewable"]:
                cap *= nodes[a]["renewable_pu"][g]
            m.var((a, "p", g), 0, cap)
        for g in d["thermal"]:
            for field in ("u", "start", "stop"):
                m.var((a, field, g), 0, 1, binary=True)
        for l in d["loads"]:
            m.var((a, "shed", l), 0, nodes[a]["load"][l])
        for s in d["stores"]:
            row = n.storage_units.loc[s]
            m.var((a, "charge", s), 0, float(-row.p_nom * row.p_min_pu))
            m.var((a, "discharge", s), 0, float(row.p_nom * row.p_max_pu))
            m.var((a, "energy", s), 0, float(row.p_nom * row.max_hours))
            m.var((a, "charge_mode", s), 0, 1, binary=True)
        m.var((a, "cost"))
        m.var((a, "risk"))
        if kids[a] and lam > 0 and alpha > 0:
            m.var((a, "eta"))
            for child in kids[a]:
                m.var((a, "excess", child))
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
            rsu, rsd = _rate(row, "ramp_limit_start_up"), _rate(row, "ramp_limit_shut_down")
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
            for duration, event, sign, bound in ((int(row.min_up_time), "start", -1, 0), (int(row.min_down_time), "stop", 1, 1)):
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
            charge, discharge, energy, mode = [ix[a, k, s] for k in ("charge", "discharge", "energy", "charge_mode")]
            Pc, Pd = float(-row.p_nom * row.p_min_pu), float(row.p_nom * row.p_max_pu)
            m.constraint({charge: 1, mode: -Pc}, high=0)
            m.constraint({discharge: 1, mode: Pd}, high=Pd)
            terms = {energy: 1, charge: -float(row.efficiency_store), discharge: 1 / float(row.efficiency_dispatch)}
            loss = 1 - float(row.standing_loss)
            if parent is None:
                rhs = loss * float(row.state_of_charge_initial)
            else:
                terms[ix[parent, "energy", s]] = -loss
                rhs = 0
            m.constraint(terms, rhs, rhs)
            if not kids[a]:
                m.constraint({energy: 1}, low=options["terminal_soc"][s])
        adjacency = {b: [] for b in d["buses"]}
        for l in d["lines"]:
            row = n.lines.loc[l]
            if nodes[a]["line_status"][l]:
                b = float(n.buses.at[row.bus0, "v_nom"]) ** 2 / float(row.x)
                m.constraint({ix[a, "flow", l]: 1, ix[a, "theta", row.bus0]: -b, ix[a, "theta", row.bus1]: b}, 0, 0)
                adjacency[row.bus0].append(row.bus1)
                adjacency[row.bus1].append(row.bus0)
        unseen = set(d["buses"])
        while unseen:
            reference = min(unseen)
            stack = [reference]
            unseen.remove(reference)
            while stack:
                for neighbor in adjacency[stack.pop()]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            m.constraint({ix[a, "theta", reference]: 1}, 0, 0)
        for b in d["buses"]:
            terms, demand = {}, 0.0
            for g in d["gens"]:
                if n.generators.at[g, "bus"] == b:
                    _add(terms, ix[a, "p", g], 1)
            for l in d["loads"]:
                if n.loads.at[l, "bus"] == b:
                    _add(terms, ix[a, "shed", l], 1)
                    demand += nodes[a]["load"][l]
            for s in d["stores"]:
                if n.storage_units.at[s, "bus"] == b:
                    _add(terms, ix[a, "discharge", s], 1)
                    _add(terms, ix[a, "charge", s], -1)
            for l in d["lines"]:
                if n.lines.at[l, "bus0"] == b:
                    _add(terms, ix[a, "flow", l], -1)
                if n.lines.at[l, "bus1"] == b:
                    _add(terms, ix[a, "flow", l], 1)
            m.constraint(terms, demand, demand)
        terms = {ix[a, "cost"]: 1}
        for g in d["gens"]:
            _add(terms, ix[a, "p", g], -float(n.generators.at[g, "marginal_cost"]))
            if g in d["thermal"]:
                for field, column in (("u", "stand_by_cost"), ("start", "start_up_cost"), ("stop", "shut_down_cost")):
                    _add(terms, ix[a, field, g], -float(n.generators.at[g, column]))
        for l in d["loads"]:
            _add(terms, ix[a, "shed", l], -options["shed_cost"][l])
        for s in d["stores"]:
            fee = options["throughput_cost"][s]
            _add(terms, ix[a, "charge", s], -fee)
            _add(terms, ix[a, "discharge", s], -fee - float(n.storage_units.at[s, "marginal_cost"]))
        m.constraint(terms, 0, 0)
        terms = {ix[a, "risk"]: 1, ix[a, "cost"]: -1}
        if kids[a]:
            if lam == 0 or alpha == 0:
                for child in kids[a]:
                    _add(terms, ix[child, "risk"], -nodes[child]["probability"])
            else:
                _add(terms, ix[a, "eta"], -lam)
                for child in kids[a]:
                    q = nodes[child]["probability"]
                    _add(terms, ix[child, "risk"], -(1-lam) * q)
                    _add(terms, ix[a, "excess", child], -lam * q / (1-alpha))
                    m.constraint({ix[a, "excess", child]: 1, ix[child, "risk"]: -1, ix[a, "eta"]: 1}, low=0)
            m.constraint(terms, low=0)
        else:
            m.constraint(terms, 0, 0)
    result = m.solve(ix[root, "risk"])
    if result.status == 2:
        return {"status": "infeasible"}
    if result.status != 0:
        return {"status": "limit" if result.status == 1 else "error"}
    x, output, carry = result.x, {}, {}
    def val(a, field, name):
        v = float(x[ix[a, field, name]])
        return 0.0 if abs(v) < 1e-9 else v
    for a in order:
        record = {"u": {}, "start": {}, "stop": {}, "p": {}, "charge": {}, "discharge": {}, "energy": {}, "shed": {}, "flow": {}, "theta": {}}
        for field in ("u", "start", "stop"):
            record[field] = {g: int(round(val(a, field, g))) for g in d["thermal"]}
        for field, members in (("p", d["gens"]), ("charge", d["stores"]), ("discharge", d["stores"]), ("energy", d["stores"]), ("shed", d["loads"]), ("flow", d["lines"]), ("theta", d["buses"])):
            record[field] = {key: val(a, field, key) for key in members}
        cost = sum(record["p"][g] * float(n.generators.at[g, "marginal_cost"]) for g in d["gens"])
        for g in d["thermal"]:
            cost += sum(record[field][g] * float(n.generators.at[g, column]) for field, column in (("u", "stand_by_cost"), ("start", "start_up_cost"), ("stop", "shut_down_cost")))
        cost += sum(record["shed"][l] * options["shed_cost"][l] for l in d["loads"])
        cost += sum(options["throughput_cost"][s] * (record["charge"][s] + record["discharge"][s]) + float(n.storage_units.at[s, "marginal_cost"]) * record["discharge"][s] for s in d["stores"])
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
                minimum = int(n.generators.at[g, "min_up_time" if status else "min_down_time"])
                thermal_state[g] = {"status": status, "p": record["p"][g], "hours": count, "remaining": max(0, minimum-count)}
            carry[a] = {"generators": thermal_state, "energy": dict(record["energy"])}
    expected = {}
    for a in reversed(order):
        values = [output[c]["risk_to_go"] for c in kids[a]]
        probabilities = [nodes[c]["probability"] for c in kids[a]]
        risk = output[a]["local_cost"]
        if values:
            risk += (1-lam) * sum(q*v for q,v in zip(probabilities, values)) + lam * _tail(values, probabilities, alpha)
        output[a]["risk_to_go"] = float(risk)
        expected[a] = output[a]["local_cost"] + sum(nodes[c]["probability"] * expected[c] for c in kids[a])
    return {"status": "optimal", "root_objective": output[root]["risk_to_go"],
            "expected_cost": float(expected[root]), "node_policy": output, "leaf_carry_state": carry}
