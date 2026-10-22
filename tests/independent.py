"""Independent small-case optimum: enumerate physical discrete policies, solve LPs.

This works from JSON records, not PyPSA/reference functions. Conditional risk
uses capped-simplex risk-envelope vertices, not the reference eta/excess form.
Both numerical paths use HiGHS, so solver independence is not claimed.
"""
from itertools import product
import numpy as np
from scipy.optimize import linprog


def risk_weights(probabilities, alpha, lam):
    q = np.array(probabilities, dtype=float)
    if len(q) == 1 or alpha == 0 or lam == 0:
        return [q]
    cap = q / (1-alpha)
    vertices = []
    for pivot in range(len(q)):
        other = [i for i in range(len(q)) if i != pivot]
        for flags in product((0, 1), repeat=len(other)):
            w = np.zeros(len(q))
            for i, flag in zip(other, flags):
                w[i] = cap[i] * flag
            w[pivot] = 1-sum(w)
            if -1e-12 <= w[pivot] <= cap[pivot] + 1e-12:
                vertices.append((1-lam)*q + lam*w)
    return vertices


def exhaustive_optimum(case):
    components = case["network"]
    nodes = {n["id"]: n for n in case["tree"]}
    order = sorted(nodes, key=lambda key: (nodes[key]["stage"], key))
    root = order[0]
    kids = {a: [b for b in order if nodes[b]["parent"] == a] for a in order}
    rows = {key: {r["name"]: r for r in components.get(key, [])} for key in ("buses", "lines", "generators", "loads", "storage_units")}
    buses, lines, gens, loads, stores = [rows[key] for key in ("buses", "lines", "generators", "loads", "storage_units")]
    thermal = [g for g, r in gens.items() if r.get("committable", False)]
    decisions = [(a, g) for a in order if kids[a] for g in thermal]
    hist = {}
    for a in order:
        parent = nodes[a]["parent"]
        hist[a] = (hist[parent] if parent is not None else []) + [a]
    options = case["options"]
    best = np.inf
    feasible_schedules = 0
    for bits in product((0, 1), repeat=len(decisions)):
        commands = dict(zip(decisions, bits))
        u, events, valid = {}, {}, True
        for a in order:
            for g in thermal:
                prior = options["initial"][g]
                parent = nodes[a]["parent"]
                u[a, g] = prior["root_status"] if parent is None else commands[parent, g]
                previous = prior["status"] if parent is None else u[parent, g]
                events[a, g] = (max(0, u[a,g]-previous), max(0, previous-u[a,g]))
                duration = 0
                for old in reversed(hist[a][:-1]):
                    if u[old, g] != previous:
                        break
                    duration += 1
                if duration == len(hist[a])-1 and previous == prior["status"]:
                    duration += prior["hours"]
                minimum = gens[g].get("min_up_time" if previous else "min_down_time", 0)
                if u[a,g] != previous and duration < minimum:
                    valid = False
        if not valid:
            continue
        feasible_schedules += 1
        for modes in product((0, 1, 2), repeat=len(order)*len(stores)):
            storage_mode = dict(zip(product(order, stores), modes))
            names, bounds = {}, []
            def variable(key, lo=0, hi=None):
                names[key] = len(bounds)
                bounds.append((lo, hi))
            for a in order:
                for g,r in gens.items():
                    if g in thermal:
                        lo = r.get("p_min_pu", 0)*r["p_nom"]*u[a,g]
                        hi = r.get("p_max_pu", 1)*r["p_nom"]*u[a,g]
                    else:
                        lo = 0
                        hi = r.get("p_max_pu", 1)*r["p_nom"]*nodes[a]["renewable_pu"][g]
                    variable((a,"p",g),lo,hi)
                for s,r in stores.items():
                    mode = storage_mode[a,s]
                    variable((a,"c",s),0,-r.get("p_min_pu",-1)*r["p_nom"] if mode==1 else 0)
                    variable((a,"d",s),0,r.get("p_max_pu",1)*r["p_nom"] if mode==2 else 0)
                    lower = options["terminal_soc"][s] if not kids[a] else 0
                    variable((a,"e",s),lower,r["p_nom"]*r.get("max_hours",1))
                for l in loads:
                    variable((a,"shed",l),0,nodes[a]["load"][l])
                for l,r in lines.items():
                    cap = r["s_nom"] if nodes[a]["line_status"][l] else 0
                    variable((a,"f",l),-cap,cap)
                for b in buses:
                    variable((a,"angle",b),None,None)
                variable((a,"V"))
            size = len(bounds)
            equations, rhs, limits, ub = [], [], [], []
            def eq(terms,value):
                row = np.zeros(size)
                for key,x in terms.items():
                    row[names[key]] += x
                equations.append(row)
                rhs.append(value)
            def le(terms,value):
                row = np.zeros(size)
                for key,x in terms.items():
                    row[names[key]] += x
                limits.append(row)
                ub.append(value)
            for a in order:
                parent = nodes[a]["parent"]
                for g in thermal:
                    r,h = gens[g],options["initial"][g]
                    P = r["p_nom"]
                    prior_u = h["status"] if parent is None else u[parent,g]
                    up = P*(r.get("ramp_limit_up",1)*prior_u+r.get("ramp_limit_start_up",1)*(u[a,g]-prior_u))
                    down = P*(r.get("ramp_limit_down",1)*u[a,g]+r.get("ramp_limit_shut_down",1)*(prior_u-u[a,g]))
                    terms = {(a,"p",g):1}
                    if parent is None:
                        le(terms,up+h["p"])
                        le({k:-v for k,v in terms.items()},down-h["p"])
                    else:
                        terms[parent,"p",g] = -1
                        le(terms,up)
                        le({k:-v for k,v in terms.items()},down)
                for s,r in stores.items():
                    terms = {(a,"e",s):1,(a,"c",s):-r.get("efficiency_store",1),(a,"d",s):1/r.get("efficiency_dispatch",1)}
                    decay = 1-r.get("standing_loss",0)
                    if parent is None:
                        eq(terms,decay*r.get("state_of_charge_initial",0))
                    else:
                        terms[parent,"e",s] = -decay
                        eq(terms,0)
                # DC flow and one gauge per actual connected component.
                graph = {b:set() for b in buses}
                for l,r in lines.items():
                    if nodes[a]["line_status"][l]:
                        b = buses[r["bus0"]].get("v_nom",1)**2/r["x"]
                        eq({(a,"f",l):1,(a,"angle",r["bus0"]):-b,(a,"angle",r["bus1"]):b},0)
                        graph[r["bus0"]].add(r["bus1"])
                        graph[r["bus1"]].add(r["bus0"])
                unseen = set(buses)
                while unseen:
                    reference = min(unseen)
                    todo = [reference]
                    unseen.remove(reference)
                    while todo:
                        for b in graph[todo.pop()]:
                            if b in unseen:
                                unseen.remove(b)
                                todo.append(b)
                    eq({(a,"angle",reference):1},0)
                for b in buses:
                    terms,demand = {},0
                    for g,r in gens.items():
                        if r["bus"]==b:
                            terms[a,"p",g]=1
                    for l,r in loads.items():
                        if r["bus"]==b:
                            terms[a,"shed",l]=1
                            demand += nodes[a]["load"][l]
                    for s,r in stores.items():
                        if r["bus"]==b:
                            terms[a,"d",s]=1
                            terms[a,"c",s]=-1
                    for l,r in lines.items():
                        if r["bus0"]==b:
                            terms[a,"f",l]=-1
                        if r["bus1"]==b:
                            terms[a,"f",l]=1
                    eq(terms,demand)
                # Exact local cost plus each risk-envelope extreme expectation.
                local,constant = {},0
                for g,r in gens.items():
                    local[a,"p",g] = r.get("marginal_cost",0)
                    if g in thermal:
                        su,sd=events[a,g]
                        constant += r.get("stand_by_cost",0)*u[a,g]+r.get("start_up_cost",0)*su+r.get("shut_down_cost",0)*sd
                for l in loads:
                    local[a,"shed",l]=options["shed_cost"][l]
                for s,r in stores.items():
                    local[a,"c",s]=options["throughput_cost"][s]
                    local[a,"d",s]=options["throughput_cost"][s]+r.get("marginal_cost",0)
                weights = risk_weights([nodes[c]["probability"] for c in kids[a]],options["alpha"],options["risk_weight"]) if kids[a] else [[]]
                for weights_row in weights:
                    terms = dict(local)
                    terms[a,"V"]=-1
                    for c,weight in zip(kids[a],weights_row):
                        terms[c,"V"]=float(weight)
                    le(terms,-constant)
            target = np.zeros(size)
            target[names[root,"V"]]=1
            result = linprog(target,A_ub=np.array(limits),b_ub=np.array(ub),A_eq=np.array(equations),b_eq=np.array(rhs),bounds=bounds,method="highs")
            if result.success:
                best = min(best,result.fun)
    return float(best), feasible_schedules
