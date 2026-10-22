"""Observe public native state; parent assertions detect interface side effects.

This is not protection against Python rewriting its own worker bootstrap.
Internal models, caches and private implementation attributes are not observed.
"""

import datetime


def scalar(value):
    """Keep exact floats and missing-value semantics without pickle."""
    import numpy as np
    import pandas as pd

    if value is pd.NA:
        return {"kind": "pandas.NA"}
    if value is pd.NaT:
        return {"kind": "pandas.NaT"}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, (datetime.datetime, datetime.date, datetime.timedelta)):
        return {"kind": type(value).__name__, "value": str(value)}
    if type(value) in (tuple, list):
        return {"kind": type(value).__name__, "items": [scalar(v) for v in value]}
    if type(value) is dict:
        import json

        entries = [[scalar(k), scalar(v)] for k, v in value.items()]
        entries.sort(key=lambda item: json.dumps(item[0], sort_keys=True))
        return {"kind": "dict", "items": entries}
    raise TypeError("Unsupported public Network scalar: " + type(value).__name__)


def dtype_record(dtype):
    result = {"type": type(dtype).__name__, "name": str(dtype)}
    if hasattr(dtype, "categories") and dtype.categories is not None:
        result["categories"] = scalar(dtype.categories.tolist())
        result["ordered"] = bool(dtype.ordered)
    if hasattr(dtype, "storage"):
        result["storage"] = str(dtype.storage)
    return result


def index_record(index):
    result = {
        "type": type(index).__name__,
        "dtype": dtype_record(index.dtype),
        "names": scalar(list(index.names)),
        "values": scalar(index.tolist()),
    }
    if hasattr(index, "levels"):
        result["levels"] = [index_record(level) for level in index.levels]
        result["codes"] = [list(map(int, code)) for code in index.codes]
    if hasattr(index, "freqstr"):
        result["frequency"] = index.freqstr
    return result


def frame_record(frame):
    return {
        "index": index_record(frame.index),
        "columns": index_record(frame.columns),
        "dtypes": [dtype_record(dtype) for dtype in frame.dtypes],
        "values": scalar(frame.to_numpy(dtype=object).tolist()),
        "attrs": scalar(frame.attrs),
    }


def snapshot(network):
    """Record public component tables, time axes/weights, metadata and name."""
    state = {
        "name": scalar(network.name),
        "meta": scalar(network.meta),
        "snapshots": index_record(network.snapshots),
        "snapshot_weightings": frame_record(network.snapshot_weightings),
        "investment_periods": index_record(network.investment_periods),
        "investment_period_weightings": frame_record(
            network.investment_period_weightings
        ),
    }
    for name, component in sorted(network.components.items()):
        state[f"components/{name}/static"] = frame_record(component.static)
        for attribute, frame in sorted(component.dynamic.items()):
            state[f"components/{name}/dynamic/{attribute}"] = frame_record(frame)
    return state


def native_policy_probe(payload):
    """Construct real native objects, then observe the public native call."""
    import pypsa

    network = pypsa.Network()
    for group, kind in (
        ("buses", "Bus"),
        ("lines", "Line"),
        ("generators", "Generator"),
        ("loads", "Load"),
        ("storage_units", "StorageUnit"),
    ):
        for record in payload["network"].get(group, []):
            fields = dict(record)
            name = fields.pop("name")
            network.add(kind, name, **fields)
    before = snapshot(network)
    result = network.optimize.policy_tree(payload["tree"], payload["options"])
    after = snapshot(network)
    return {"result": result, "before": before, "after": after}
