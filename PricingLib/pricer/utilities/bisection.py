def bisection(f, lower_bond, upper_bond, tol):
    """
    Finds a root of the function f using the bisection method.

    Parameters:
        f (function): The function to find a root of.
        lower_bond (float): The lower bound of the initial interval.
        upper_bond (float): The upper bound of the initial interval.
        tol (float): The tolerance level for the approximation.

    Returns:
        float: An approximation of a root of the function f.
    """
    if f(lower_bond) * f(upper_bond) >= 0:
        raise ValueError("Function must have opposite signs at a and b.")

    while abs(upper_bond - lower_bond) > tol:
        mid_way = (lower_bond + upper_bond) / 2
        if f(mid_way) == 0:
            return mid_way
        elif f(mid_way) * f(lower_bond) < 0:
            upper_bond = mid_way
        else:
            lower_bond = mid_way

    return (lower_bond + upper_bond) / 2
