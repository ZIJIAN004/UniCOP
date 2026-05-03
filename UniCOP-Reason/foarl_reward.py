"""
FOARL: Feasibility-and-Optimality-Aware Reinforcement Learning reward.

Reference: "Large Language Models as End-to-end Combinatorial Optimization Solvers"
           (Jiang et al., NeurIPS 2025, arXiv:2509.16865)

R^P = R_f + R_o
  R_f = omega_0 * zeta + sum(omega_i * c_i)    if zeta != 0, else 0
  R_o = alpha / (1 + gap)                      if zeta != 0, else 0
  gap = max(0, (f(x_hat) - f(x*)) / |f(x*)|)
"""

from terminal_reward import compute_terminal_components

_PROBLEM_CACHE = {}


def _get_problem(problem_type: str):
    if problem_type not in _PROBLEM_CACHE:
        from problems import get_problem
        _PROBLEM_CACHE[problem_type] = get_problem(problem_type)
    return _PROBLEM_CACHE[problem_type]


def compute_foarl_reward(
    completion: str,
    instance: dict,
    problem_type: str,
    ref_distance: float,
    alpha: float = 0.5,
    omega_parse: float = 0.2,
    omega_coverage: float = 0.3,
    omega_constraint: float = 0.3,
    omega_format: float = 0.2,
) -> tuple[float, dict]:
    """
    Returns (scalar_reward, components_dict).
    components_dict: parse, coverage, constraint, format, R_f, R_o, gap.
    """
    c = compute_terminal_components(completion, instance, problem_type)

    zeta = c["parse"]
    if zeta == 0.0:
        return 0.0, {**c, "R_f": 0.0, "R_o": 0.0, "gap": None}

    R_f = (omega_parse * zeta
           + omega_coverage * c["coverage"]
           + omega_constraint * c["constraint"]
           + omega_format * c["format"])

    R_o = 0.0
    gap = None
    if ref_distance is not None and abs(ref_distance) > 1e-10:
        dist = _get_problem(problem_type).get_tour_distance(completion, instance)
        if dist is not None:
            gap = max(0.0, (dist - ref_distance) / abs(ref_distance))
            R_o = alpha / (1.0 + gap)

    return R_f + R_o, {**c, "R_f": R_f, "R_o": R_o, "gap": gap}
