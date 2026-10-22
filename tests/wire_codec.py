"""Data-only RPC. Never deserialize pickle, Python source, or class names."""

import math

try:
    import numpy as np
except ImportError:
    np = None  # Built-in wire values remain usable without the optional array codec.

MAX_ITEMS = 2_000_000


def pack(value):
    """Encode supported data types without importing candidate-defined classes."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"t": "float", "v": repr(value)}
    if np is not None and isinstance(value, np.generic):
        return pack(value.item())
    if type(value) is complex:
        return {"t": "complex", "v": [pack(value.real), pack(value.imag)]}
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or value.size > MAX_ITEMS:
            raise TypeError("Unsupported array dtype or size")
        return {
            "t": "array",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "v": [pack(x) for x in value.ravel().tolist()],
        }
    if type(value) in (list, tuple):
        return {
            "t": "tuple" if type(value) is tuple else "list",
            "v": [pack(x) for x in value],
        }
    if type(value) is dict:
        return {"t": "dict", "v": [[pack(k), pack(v)] for k, v in value.items()]}
    raise TypeError("Unsupported RPC type: " + type(value).__name__)


def unpack(value):
    """Decode bounded data-only values, rejecting unknown tags and array shapes."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if not isinstance(value, dict):
        raise ValueError("Invalid wire value")
    tag = value.get("t")
    data = value.get("v")
    if tag == "float" and data in ("nan", "inf", "-inf"):
        return float(data)
    if tag == "complex" and isinstance(data, list) and len(data) == 2:
        return complex(unpack(data[0]), unpack(data[1]))
    if tag in ("list", "tuple") and isinstance(data, list) and len(data) <= MAX_ITEMS:
        items = [unpack(x) for x in data]
        return tuple(items) if tag == "tuple" else items
    if tag == "dict" and isinstance(data, list) and len(data) <= MAX_ITEMS:
        return {unpack(k): unpack(v) for k, v in data}
    if tag == "array" and np is not None:
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if (
            dtype.kind not in "biufc"
            or dtype.itemsize > 16
            or not isinstance(shape, list)
            or len(shape) > 16
            or any(type(x) is not int or x < 0 for x in shape)
            or math.prod(shape) > MAX_ITEMS
            or len(data) != math.prod(shape)
        ):
            raise ValueError("Invalid wire array")
        return np.asarray([unpack(x) for x in data], dtype=dtype).reshape(shape)
    raise ValueError("Unknown wire value")
