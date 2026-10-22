"""Physical policy, causal history, topology and risk checked independently.

All cases are synthetic bounded research networks, not measured grid records.
Optimum fixtures enumerate discrete physical policies and solve risk-envelope
LPs; raw returned policies are checked without importing candidate code.
"""
import copy
import math
import numpy as np
import pytest
from candidate_proxy import candidate
from independent import exhaustive_optimum


def close(actual, expected):
    assert isinstance(actual, (int, float)) and not isinstance(actual, bool)
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=2e-5, rel=2e-6)


def tail(values, probabilities, alpha):
    remaining, weighted = 1-alpha, 0.0
    for i in sorted(range(len(values)), key=lambda j: values[j], reverse=True):
        take = min(remaining, probabilities[i])
        weighted += take*values[i]
        remaining -= take
        if remaining <= 1e-13:
            break
    return weighted/(1-alpha)


def base_case():
    return {"network": {
        "buses": [{"name":"A","v_nom":10},{"name":"B","v_nom":10}],
        "lines": [{"name":"AB","bus0":"A","bus1":"B","x":1.7,"s_nom":7}],
        "generators": [
            {"name":"GT","bus":"A","p_nom":8,"committable":True,"p_min_pu":0.2,"p_max_pu":1,"marginal_cost":4,"stand_by_cost":2,"start_up_cost":15,"shut_down_cost":3,"min_up_time":2,"min_down_time":1,"ramp_limit_up":0.5,"ramp_limit_down":0.6,"ramp_limit_start_up":0.7,"ramp_limit_shut_down":0.7},
            {"name":"PV","bus":"B","p_nom":6,"committable":False,"marginal_cost":0}],
        "loads": [{"name":"LA","bus":"A"},{"name":"LB","bus":"B"}],
        "storage_units": [{"name":"BAT","bus":"B","p_nom":3,"max_hours":2,"efficiency_store":0.9,"efficiency_dispatch":0.8,"standing_loss":0.05,"state_of_charge_initial":1.3,"marginal_cost":0.2}]},
        "tree": [],
        "options": {"risk_weight":0.65,"alpha":0.73,"initial":{"GT":{"status":0,"p":0,"hours":3,"root_status":0}},"terminal_soc":{"BAT":0.4},"shed_cost":{"LA":100,"LB":120},"throughput_cost":{"BAT":0.1}}}


def node(key,parent,stage,prob=1,la=2,lb=3,pv=0,on=True):
    return {"id":key,"parent":parent,"stage":stage,"probability":prob,"load":{"LA":la,"LB":lb},"renewable_pu":{"PV":pv},"line_status":{"AB":on}}


def no_storage(case):
    case["network"]["storage_units"] = []
    case["options"]["terminal_soc"] = {}
    case["options"]["throughput_cost"] = {}


def branched(storage=False):
    case=base_case()
    if not storage:
        no_storage(case)
    case["tree"]=[node("root",None,0,la=1,lb=1,pv=1),
        node("low","root",1,0.7,la=2,lb=1,pv=0.6),
        node("high","root",1,0.3,la=2,lb=4,pv=0,on=False),
        node("low-end","low",2,la=2,lb=4,pv=0.1),
        node("high-end","high",2,la=2,lb=3,pv=0.1,on=True)]
    return case


def validate(case,result,optimum=None):
    if optimum is not None and not math.isfinite(optimum):
        assert result == {"status":"infeasible"}
        return
    assert result["status"] == "optimal"
    tables={kind:{r["name"]:r for r in case["network"].get(kind,[])} for kind in ("buses","lines","generators","loads","storage_units")}
    buses,lines,gens,loads,stores=[tables[k] for k in ("buses","lines","generators","loads","storage_units")]
    thermal=[g for g,r in gens.items() if r.get("committable",False)]
    nodes={r["id"]:r for r in case["tree"]}
    order=sorted(nodes,key=lambda a:(nodes[a]["stage"],a))
    kids={a:[b for b in nodes if nodes[b]["parent"]==a] for a in nodes}
    policies=result["node_policy"]
    assert set(policies)==set(nodes)
    paths={}
    for a in order:
        raw=nodes[a]
        parent=raw["parent"]
        paths[a]=(paths[parent] if parent is not None else [])+[a]
        out=policies[a]
        for field,names in (("u",thermal),("start",thermal),("stop",thermal),("p",gens),("charge",stores),("discharge",stores),("energy",stores),("shed",loads),("flow",lines),("theta",buses)):
            assert set(out[field])==set(names)
            for v in out[field].values():
                close(v,v)
        for g in thermal:
            h=case["options"]["initial"][g]
            r=gens[g]
            u=out["u"][g]
            assert u in (0,1)
            before=h["status"] if parent is None else policies[parent]["u"][g]
            prevp=h["p"] if parent is None else policies[parent]["p"][g]
            close(out["start"][g],max(0,u-before))
            close(out["stop"][g],max(0,before-u))
            if parent is None:
                close(u,h["root_status"])
            for sibling in kids[a]:
                close(policies[sibling]["u"][g],policies[kids[a][0]]["u"][g])
            duration=0
            for old in reversed(paths[a][:-1]):
                if policies[old]["u"][g] != before:
                    break
                duration+=1
            if duration==len(paths[a])-1 and before==h["status"]:
                duration+=h["hours"]
            if u!=before:
                assert duration>=r.get("min_up_time" if before else "min_down_time",0)
            p,P=out["p"][g],r["p_nom"]
            assert p>=P*r.get("p_min_pu",0)*u-2e-5
            assert p<=P*r.get("p_max_pu",1)*u+2e-5
            up=P*(r.get("ramp_limit_up",1)*before+r.get("ramp_limit_start_up",1)*(u-before))
            down=P*(r.get("ramp_limit_down",1)*u+r.get("ramp_limit_shut_down",1)*(before-u))
            assert p-prevp<=up+2e-5
            assert prevp-p<=down+2e-5
        for g,r in gens.items():
            if g not in thermal:
                assert out["p"][g]>=-2e-5
                assert out["p"][g]<=r["p_nom"]*r.get("p_max_pu",1)*raw["renewable_pu"][g]+2e-5
        for s,r in stores.items():
            charge,discharge,e=out["charge"][s],out["discharge"][s],out["energy"][s]
            assert charge>=-2e-5 and discharge>=-2e-5
            assert min(charge,discharge)<=2e-5
            assert charge<=-r.get("p_min_pu",-1)*r["p_nom"]+2e-5
            assert discharge<=r.get("p_max_pu",1)*r["p_nom"]+2e-5
            assert -2e-5<=e<=r["p_nom"]*r.get("max_hours",1)+2e-5
            before=r.get("state_of_charge_initial",0) if parent is None else policies[parent]["energy"][s]
            close(e,(1-r.get("standing_loss",0))*before+r.get("efficiency_store",1)*charge-discharge/r.get("efficiency_dispatch",1))
            if not kids[a]:
                assert e>=case["options"]["terminal_soc"][s]-2e-5
        graph={b:set() for b in buses}
        for l,r in lines.items():
            flow=out["flow"][l]
            assert abs(flow)<=r["s_nom"]+2e-5
            if raw["line_status"][l]:
                close(flow,buses[r["bus0"]].get("v_nom",1)**2/r["x"]*(out["theta"][r["bus0"]]-out["theta"][r["bus1"]]))
                graph[r["bus0"]].add(r["bus1"])
                graph[r["bus1"]].add(r["bus0"])
            else:
                close(flow,0)
        unseen=set(buses)
        while unseen:
            reference=min(unseen)
            stack=[reference]
            unseen.remove(reference)
            while stack:
                for b in graph[stack.pop()]:
                    if b in unseen:
                        unseen.remove(b)
                        stack.append(b)
            close(out["theta"][reference],0)
        for l in loads:
            assert -2e-5<=out["shed"][l]<=raw["load"][l]+2e-5
        for b in buses:
            generation=sum(out["p"][g] for g,r in gens.items() if r["bus"]==b)
            storage=sum(out["discharge"][s]-out["charge"][s] for s,r in stores.items() if r["bus"]==b)
            demand=sum(raw["load"][l]-out["shed"][l] for l,r in loads.items() if r["bus"]==b)
            export=sum(out["flow"][l] for l,r in lines.items() if r["bus0"]==b)-sum(out["flow"][l] for l,r in lines.items() if r["bus1"]==b)
            close(generation+storage,demand+export)
        cost=sum(out["p"][g]*r.get("marginal_cost",0) for g,r in gens.items())
        cost+=sum(out["u"][g]*gens[g].get("stand_by_cost",0)+out["start"][g]*gens[g].get("start_up_cost",0)+out["stop"][g]*gens[g].get("shut_down_cost",0) for g in thermal)
        cost+=sum(out["shed"][l]*case["options"]["shed_cost"][l] for l in loads)
        cost+=sum((out["charge"][s]+out["discharge"][s])*case["options"]["throughput_cost"][s]+out["discharge"][s]*r.get("marginal_cost",0) for s,r in stores.items())
        close(out["local_cost"],cost)
    expected={}
    for a in reversed(order):
        out=policies[a]
        vals=[policies[b]["risk_to_go"] for b in kids[a]]
        qs=[nodes[b]["probability"] for b in kids[a]]
        risk=out["local_cost"]
        if vals:
            lam=case["options"]["risk_weight"]
            risk+=(1-lam)*sum(q*v for q,v in zip(qs,vals))+lam*tail(vals,qs,case["options"]["alpha"])
        close(out["risk_to_go"],risk)
        expected[a]=out["local_cost"]+sum(nodes[b]["probability"]*expected[b] for b in kids[a])
    close(result["root_objective"],policies[order[0]]["risk_to_go"])
    close(result["expected_cost"],expected[order[0]])
    if optimum is not None:
        close(result["root_objective"],optimum)
    leaves={a for a in nodes if not kids[a]}
    assert set(result["leaf_carry_state"])==leaves
    for a in leaves:
        carry=result["leaf_carry_state"][a]
        assert set(carry["generators"])==set(thermal)
        assert set(carry["energy"])==set(stores)
        for s in stores:
            close(carry["energy"][s],policies[a]["energy"][s])
        for g in thermal:
            h=case["options"]["initial"][g]
            status=policies[a]["u"][g]
            hours=0
            for b in reversed(paths[a]):
                if policies[b]["u"][g]!=status:
                    break
                hours+=1
            if hours==len(paths[a]) and status==h["status"]:
                hours+=h["hours"]
            minimum=gens[g].get("min_up_time" if status else "min_down_time",0)
            state=carry["generators"][g]
            close(state["status"],status)
            close(state["p"],policies[a]["p"][g])
            close(state["hours"],hours)
            close(state["remaining"],max(0,minimum-hours))


def checked(case):
    optimum,schedules=exhaustive_optimum(case)
    assert schedules>=0
    answer=candidate.run(case)
    validate(case,answer,optimum)
    return answer


def test_native_deterministic_entry_still_works():
    """Existing n.optimize remains usable; one-bus merit order has exact cost."""
    case={"operation":"deterministic","network":{"buses":[{"name":"b","v_nom":10}],"generators":[{"name":"g","bus":"b","p_nom":10,"marginal_cost":3}],"loads":[{"name":"l","bus":"b"}]},"load":{"l":2}}
    answer=candidate.run(case)
    assert answer["status"]=="ok"
    assert answer["condition"]=="optimal"
    close(answer["objective"],6)
    close(answer["generation"]["g"],2)


def test_uc_initial_history_ramps_and_leaf_obligations():
    """Contract thermal history/ramping/carry: independently enumerated path."""
    case=base_case()
    no_storage(case)
    case["options"]["initial"]["GT"]={"status":1,"p":2,"hours":1,"root_status":1}
    case["network"]["generators"][0]["min_up_time"]=3
    case["tree"]=[node("a",None,0,la=2,lb=1,pv=0.1),node("b","a",1,la=2,lb=5,pv=0),node("c","b",2,la=1,lb=0,pv=1),node("d","c",3,la=2,lb=4,pv=0)]
    checked(case)


@pytest.mark.parametrize("weight,alpha",[(0,0.8),(0.65,0.73),(1,0.5),(0.8,0)])
def test_advance_commitment_and_nested_conditional_risk(weight,alpha):
    """Conditional tree risk and sibling commitments match risk-envelope LPs."""
    case=branched()
    case["options"]["risk_weight"],case["options"]["alpha"]=weight,alpha
    # A second split makes nesting structurally distinct from path CVaR.
    case["tree"][-2]["probability"]=0.8
    case["tree"][-1]["probability"]=0.6
    case["tree"] += [node("low-tail","low",2,0.2,la=2,lb=6,pv=0),node("high-tail","high",2,0.4,la=1,lb=5,pv=0.8)]
    checked(case)


def test_storage_inheritance_islanding_and_reconnection():
    """State, topology, commitment and risk interact; all discrete modes enumerated."""
    checked(branched(storage=True))


def test_storage_losses_efficiencies_reserve_and_exclusive_modes():
    """An independent three-hour LP enumeration verifies both energy boundaries."""
    case=base_case()
    case["options"]["terminal_soc"]["BAT"]=1.7
    case["tree"]=[node("r",None,0,la=0.5,lb=1,pv=1),node("s","r",1,la=2,lb=6,pv=0),node("t","s",2,la=2,lb=1,pv=0.2)]
    checked(case)


def test_node_and_component_order_do_not_change_policy_problem():
    """Native ids and parent history govern results, not JSON record ordering."""
    case=branched(storage=True)
    baseline=candidate.run(case)
    validate(case,baseline)
    shuffled=copy.deepcopy(case)
    shuffled["tree"]=list(reversed(shuffled["tree"]))
    for records in shuffled["network"].values():
        records.reverse()
    result=candidate.run(shuffled)
    validate(shuffled,result,baseline["root_objective"])


def test_mesh_flow_and_independent_island_angle_gauges():
    """DC meshed branches and true outage removal satisfy every bus equation."""
    case=branched()
    case["network"]["buses"].append({"name":"C","v_nom":10})
    case["network"]["loads"].append({"name":"LC","bus":"C"})
    case["options"]["shed_cost"]["LC"]=110
    case["network"]["lines"] += [{"name":"AC","bus0":"A","bus1":"C","x":1,"s_nom":1.5},{"name":"CB","bus0":"C","bus1":"B","x":0.8,"s_nom":2}]
    for raw in case["tree"]:
        raw["load"]["LC"]=1.1
        raw["line_status"]["AC"]=True
        raw["line_status"]["CB"]=raw["id"]!="high"
    checked(case)


@pytest.mark.parametrize("kind",["initial-obligation","excess-minimum","terminal-reserve","storage-no-disposal"])
def test_true_infeasibility_is_reported(kind):
    """No feasible physical schedule must return infeasible, not shed away constraints."""
    case=base_case()
    case["tree"]=[node("r",None,0,la=0,lb=0,pv=0)]
    if kind=="initial-obligation":
        case["network"]["generators"][0]["min_up_time"]=3
        case["options"]["initial"]["GT"]={"status":1,"p":2,"hours":1,"root_status":0}
    elif kind=="excess-minimum":
        no_storage(case)
        case["options"]["initial"]["GT"]={"status":1,"p":2,"hours":3,"root_status":1}
    elif kind=="terminal-reserve":
        case["network"]["storage_units"][0]["state_of_charge_initial"]=0
        case["options"]["terminal_soc"]["BAT"]=2
    else:
        case["options"]["initial"]["GT"]={"status":1,"p":2,"hours":3,"root_status":1}
        case["network"]["storage_units"][0].update(p_nom=10,max_hours=1,state_of_charge_initial=10)
    checked(case)


def test_twenty_five_node_supported_tree():
    """Public upper node bound: exact constant-demand objective on six branches."""
    case=base_case()
    no_storage(case)
    case["network"]["generators"]=[{"name":"PV","bus":"B","p_nom":20,"committable":False,"marginal_cost":2}]
    case["options"]["initial"]={}
    case["tree"]=[node("root",None,0,la=1,lb=2,pv=1)]
    for j in range(6):
        key=f"s{j}"
        case["tree"].append(node(key,"root",1,1/6,la=1,lb=2,pv=1))
        for k in range(3):
            case["tree"].append(node(f"{key}-{k}",key,2,1/3,la=1,lb=2,pv=1))
    validate(case,candidate.run(case),18)


@pytest.mark.parametrize("kind",["probability","missing-parent","unequal-depth","unknown-attribute"])
def test_invalid_contract_inputs_raise_value_error(kind):
    """Published malformed-domain behavior; error text is not constrained."""
    case=branched()
    if kind=="probability":
        case["tree"][1]["probability"]=0.4
    elif kind=="missing-parent":
        case["tree"][1]["parent"]="absent"
    elif kind=="unequal-depth":
        case["tree"].pop()
    else:
        case["network"]["generators"][0]["p_nom_extendable"]=True
    with pytest.raises(ValueError):
        candidate.run(case)
