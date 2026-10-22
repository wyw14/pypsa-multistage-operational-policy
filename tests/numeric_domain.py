"""Independent exact public angle-domain check using all-pairs min-plus paths.

Uses JSON records only, no PyPSA, reference, public checker, or optimization.
The binary exponent is found by rational bisection rather than float log2.
"""

from fractions import Fraction


def _power(exponent):
    return Fraction(2) ** exponent


def uniform_rounding_bound(bound):
    if not bound:
        return Fraction(0)
    low, high = -1074, 1024
    if bound < _power(low):
        return min(bound, _power(-1075))
    while low + 1 < high:
        middle = (low + high) // 2
        if _power(middle) <= bound:
            low = middle
        else:
            high = middle
    return min(bound, _power(max(low - 52, -1074) - 1))


def independent_angle_budgets(payload):
    buses = {item["name"]: item for item in payload["network"]["buses"]}
    names = sorted(buses)
    index = {name: i for i, name in enumerate(names)}
    lines = payload["network"].get("lines", [])
    records = []
    for node in payload["tree"]:
        size = len(names)
        distances = [[None for _ in names] for _ in names]
        for i in range(size):
            distances[i][i] = Fraction(0)
        active = []
        for line in lines:
            if not node["line_status"][line["name"]]:
                continue
            i, j = index[line["bus0"]], index[line["bus1"]]
            voltage = Fraction(float(buses[line["bus0"]].get("v_nom", 1)))
            coefficient = voltage * voltage / Fraction(float(line["x"]))
            weight = Fraction(float(line.get("s_nom", 0))) / coefficient
            if distances[i][j] is None or weight < distances[i][j]:
                distances[i][j] = distances[j][i] = weight
            active.append((line["name"], i, j, coefficient))
        for middle in range(size):
            for i in range(size):
                for j in range(size):
                    first, second = distances[i][middle], distances[middle][j]
                    if first is None or second is None:
                        continue
                    proposed = first + second
                    if distances[i][j] is None or proposed < distances[i][j]:
                        distances[i][j] = proposed
        gauge = [
            min(j for j in range(size) if distances[i][j] is not None)
            for i in range(size)
        ]
        bounds = [distances[gauge[i]][i] for i in range(size)]
        for name, i, j, coefficient in active:
            error = coefficient * (
                uniform_rounding_bound(bounds[i]) + uniform_rounding_bound(bounds[j])
            )
            records.append((node["id"], name, error))
    return records
