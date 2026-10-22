"""Public numeric support contract, independent of any optimization algorithm.

All binary64 input values are interpreted exactly when deriving the angle
serialization budget. This is not a HiGHS conditioning or solve-success test.
"""

from fractions import Fraction
import heapq

SCALAR_LIMITS = {
    "v_nom": (1e-3, 1e5),
    "x": (1e-12, 1e6),
    "s_nom": (0.0, 1e6),
    "p_nom": (1e-6, 1e6),
    "max_hours": (1e-6, 1e4),
    "efficiency": (1e-6, 1.0),
    "cost": (0.0, 1e13),
}
ANGLE_FLOW_ROUNDING_BUDGET = Fraction(1, 5_000_000)  # exactly2e-7MW


def rounding_error_bound(bound):
    """Uniform nearest-even binary64 rounding bound on[-bound,bound]."""
    if not bound:
        return Fraction(0)
    exponent = bound.numerator.bit_length() - bound.denominator.bit_length()
    if Fraction(2) ** exponent > bound:
        exponent -= 1
    half_spacing = Fraction(2) ** max(exponent - 52, -1074) / 2
    return min(bound, half_spacing)


def validate_angle_budget(network, buses, lines, nodes):
    """Validate every actual topology by a priori shortest-path angle bounds.

    Along an active line, |theta_u-theta_v|<=s_nom*x/v_nom**2.
    Any path from the component gauge therefore bounds |theta|. Its shortest
    such bound controls the serialization error, without knowing a solution.
    """
    checked = set()
    for node in nodes.values():
        active = tuple(line for line in lines if node["line_status"][line])
        if active in checked:
            continue
        checked.add(active)
        graph = {bus: [] for bus in buses}
        coefficients = {}
        for line in active:
            row = network.lines.loc[line]
            voltage = Fraction(float(network.buses.at[row.bus0, "v_nom"]))
            susceptibility = voltage * voltage / Fraction(float(row.x))
            coefficients[line] = susceptibility
            weight = Fraction(float(row.s_nom)) / susceptibility
            graph[row.bus0].append((row.bus1, weight))
            graph[row.bus1].append((row.bus0, weight))
        distances = {}
        for reference in buses:
            if reference in distances:
                continue
            distances[reference] = Fraction(0)
            queue = [(Fraction(0), reference)]
            while queue:
                distance, bus = heapq.heappop(queue)
                if distance != distances[bus]:
                    continue
                for neighbor, weight in graph[bus]:
                    proposed = distance + weight
                    if neighbor not in distances or proposed < distances[neighbor]:
                        distances[neighbor] = proposed
                        heapq.heappush(queue, (proposed, neighbor))
        for line in active:
            row = network.lines.loc[line]
            error = coefficients[line] * (
                rounding_error_bound(distances[row.bus0])
                + rounding_error_bound(distances[row.bus1])
            )
            if error > ANGLE_FLOW_ROUNDING_BUDGET:
                raise ValueError("Public angle serialization budget exceeded")
