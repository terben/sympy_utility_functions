"""
Utilities for symbolic sums with SymPy
======================================

This file contains small, didactically oriented helper functions for
manipulating unevaluated SymPy sums.

The main topics are:

- interchanging sums and integrals,
- pushing factors into sums,
- applying transformations to summands,
- splitting sums over additions,
- differentiating formal power series.

The functions are intentionally conservative. If a requested symbolic
rewrite is not applicable, the original expression is returned unchanged.
This makes the helpers convenient for exploratory calculations in notebooks
and teaching material.
"""

import sympy as sp


def _as_expr(expr, name):
    """
    Convert an object to a SymPy expression.

    Parameters
    ----------
    expr : object
        Object to convert.

    name : str
        Name used in the error message.

    Returns
    -------
    sympy.Expr
        Converted SymPy expression.

    Raises
    ------
    TypeError
        If the object cannot be converted to a SymPy expression.
    """
    try:
        expr = sp.sympify(expr)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a SymPy expression or convertible to one."
        ) from exc

    if not isinstance(expr, sp.Expr):
        raise TypeError(
            f"{name} must be a SymPy expression, got {type(expr).__name__}."
        )

    return expr


def _validate_symbol(symbol, name):
    """
    Validate that an object is a SymPy symbol.

    Parameters
    ----------
    symbol : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    sympy.Symbol
        The validated symbol.

    Raises
    ------
    TypeError
        If ``symbol`` is not a SymPy symbol.
    """
    if not isinstance(symbol, sp.Symbol):
        raise TypeError(
            f"{name} must be a SymPy Symbol, got {type(symbol).__name__}."
        )

    return symbol



def _validate_nonnegative_integer(value, name):
    """
    Validate that an object is a non-negative integer.

    Parameters
    ----------
    value : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    int
        The validated integer as a Python ``int``.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.

    ValueError
        If ``value`` is negative.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a non-negative integer, not bool.")

    if not isinstance(value, (int, sp.Integer)):
        raise TypeError(
            f"{name} must be a non-negative integer, got {type(value).__name__}."
        )

    value = int(value)

    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")

    return value


def _sum_indices(sum_expr):
    """
    Return the summation indices of a SymPy sum.

    Parameters
    ----------
    sum_expr : sympy.Sum
        Sum whose limits are inspected.

    Returns
    -------
    tuple[sympy.Symbol, ...]
        Summation indices occurring in the limits.
    """
    return tuple(limit[0] for limit in sum_expr.limits)


def swap_integral_sum(expr):
    """
    Swap an integral over a sum into a sum over integrals.

    The transformation is

        Integral(Sum(f_i, limits_sum), limits_int)
        -> Sum(Integral(f_i, limits_int), limits_sum)

    and is applied to all matching integrals inside ``expr``.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to transform.

    Returns
    -------
    sympy.Expr
        Expression with matching ``Integral(Sum(...), ...)`` structures
        rewritten. If no matching structure is found, the expression is
        returned unchanged.

    Raises
    ------
    TypeError
        If ``expr`` cannot be converted to a SymPy expression.

    Examples
    --------
    >>> x, i, n = sp.symbols("x i n")
    >>> expr = sp.Integral(sp.Sum(i*x, (i, 1, n)), x)
    >>> swap_integral_sum(expr)
    Sum(Integral(i*x, x), (i, 1, n))
    """
    expr = _as_expr(expr, "expr")

    replacements = {}

    for integral in expr.find(sp.Integral):
        integrand = integral.function
        int_limits = integral.limits

        if isinstance(integrand, sp.Sum):
            summand = integrand.function
            sum_limits = integrand.limits
            replacements[integral] = sp.Sum(
                sp.Integral(summand, *int_limits),
                *sum_limits,
            )

    return expr.xreplace(replacements)


def push_factor_into_sum(expr, factor, summation_index):
    """
    Push a factor symbolically into one matching sum.

    The transformation is

        Sum(f_i, (i, ...)) -> Sum(a*f_i, (i, ...)) / a

    where ``a`` is the given factor. The factor is only pushed into the sum
    if it does not depend on the summation index.

    Only one matching sum is transformed. If no matching sum is found, or if
    ``factor`` depends on ``summation_index``, the expression is returned
    unchanged.

    Parameters
    ----------
    expr : sympy.Expr
        Expression containing the target sum.

    factor : sympy.Expr
        Factor to insert into the summand.

    summation_index : sympy.Symbol
        Summation index of the target sum.

    Returns
    -------
    sympy.Expr
        Transformed expression, or the original expression if the
        transformation is not applicable.

    Raises
    ------
    TypeError
        If ``expr`` or ``factor`` cannot be converted to SymPy expressions,
        or if ``summation_index`` is not a SymPy symbol.

    Examples
    --------
    >>> i, n, a = sp.symbols("i n a")
    >>> expr = sp.Sum(i, (i, 1, n))
    >>> push_factor_into_sum(expr, a, i)
    Sum(a*i, (i, 1, n))/a
    """
    expr = _as_expr(expr, "expr")
    factor = _as_expr(factor, "factor")
    summation_index = _validate_symbol(summation_index, "summation_index")

    if factor.has(summation_index):
        return expr

    target_sum = None

    for sum_expr in expr.find(sp.Sum):
        if summation_index in _sum_indices(sum_expr):
            target_sum = sum_expr
            break

    if target_sum is None:
        return expr

    summand = target_sum.function
    limits = target_sum.limits

    return expr.xreplace({
        target_sum: sp.Sum(factor * summand, *limits) / factor
    })


def apply_to_summand(sum_expr, func, *args, **kwargs):
    """
    Apply a function to the summand of a SymPy sum.

    The summation limits are kept unchanged.

    Parameters
    ----------
    sum_expr : sympy.Sum
        Sum whose summand is transformed.

    func : callable
        Function applied to the summand.

    *args, **kwargs
        Additional arguments passed to ``func``.

    Returns
    -------
    sympy.Sum
        New sum with transformed summand and unchanged limits.

    Raises
    ------
    TypeError
        If ``sum_expr`` is not a SymPy ``Sum`` or if ``func`` is not callable.

    Examples
    --------
    >>> i, n, x = sp.symbols("i n x")
    >>> expr = sp.Sum((x + 1)*(x + 2), (i, 1, n))
    >>> apply_to_summand(expr, sp.expand)
    Sum(x**2 + 3*x + 2, (i, 1, n))
    """
    if not isinstance(sum_expr, sp.Sum):
        raise TypeError(
            f"sum_expr must be a SymPy Sum, got {type(sum_expr).__name__}."
        )

    if not callable(func):
        raise TypeError(
            f"func must be callable, got {type(func).__name__}."
        )

    summand = sum_expr.function
    limits = sum_expr.limits

    return sp.Sum(func(summand, *args, **kwargs), *limits)


def push_prefactors_into_sums(expr):
    """
    Push prefactors of sums into the corresponding summands.

    The transformation is

        a * Sum(f_i, (i, ...)) -> Sum(a*f_i, (i, ...))

    and is applied inside the expression where possible.

    A prefactor is only pushed into the sum if it does not depend on any
    summation index of that sum. Products containing more than one sum are
    deliberately left unchanged.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to transform.

    Returns
    -------
    sympy.Expr
        Expression with applicable prefactors pushed into sums. If no
        applicable structure is found, the expression is returned unchanged.

    Raises
    ------
    TypeError
        If ``expr`` cannot be converted to a SymPy expression.

    Examples
    --------
    >>> i, n, a = sp.symbols("i n a")
    >>> expr = a * sp.Sum(i + 1, (i, 1, n))
    >>> push_prefactors_into_sums(expr)
    Sum(a*(i + 1), (i, 1, n))
    """
    expr = _as_expr(expr, "expr")

    replacements = {}

    for product in expr.find(sp.Mul):
        sum_factors = [arg for arg in product.args if isinstance(arg, sp.Sum)]

        if len(sum_factors) != 1:
            continue

        sum_expr = sum_factors[0]
        other_factors = [arg for arg in product.args if arg != sum_expr]
        prefactor = sp.Mul(*other_factors)

        if any(prefactor.has(index) for index in _sum_indices(sum_expr)):
            continue

        summand = sum_expr.function
        limits = sum_expr.limits
        replacements[product] = sp.Sum(prefactor * summand, *limits)

    return expr.xreplace(replacements)



def split_operator_over_addition(expr, operator):
    """
    Split linear operators over additions.

    The transformation is

        L(f + g) -> L(f) + L(g)

    where ``L`` is a linear operator such as ``sp.Sum`` or
    ``sp.Integral``.
    """
    expr = _as_expr(expr, "expr")

    if not callable(operator):
        raise TypeError(
            "operator must be a callable SymPy operator class."
        )

    replacements = {}

    for op_expr in expr.find(operator):
        interior = op_expr.function
        limits = op_expr.limits

        if isinstance(interior, sp.Add):
            replacements[op_expr] = sp.Add(
                *(operator(term, *limits) for term in interior.args)
            )

    return expr.xreplace(replacements)


def collect_operator_terms(expr, operator, limits):
    """
    Collect operators with identical limits into a single operator.

    The transformation is

        L(f) + L(g) -> L(f + g)

    where ``L`` is a linear operator such as ``sp.Sum`` or
    ``sp.Integral``.
    """
    expr = _as_expr(expr, "expr")

    if not callable(operator):
        raise TypeError(
            "operator must be a callable SymPy operator class."
        )

    targets = [
        op_expr for op_expr in expr.find(operator)
        if op_expr.limits == limits
    ]

    if not targets:
        return expr

    reduced = expr

    for op_expr in targets:
        reduced = reduced.subs(op_expr, 0)

    interior = sp.Add(*(op.function for op in targets)).simplify()

    return reduced + operator(interior, *limits)


def diff_power_series(expr, x, order=1):
    """
    Differentiate a power series and shift the summation index.

    The function is intended for unevaluated SymPy sums representing power
    series of the form

        Sum(a(n) * x**n, (n, 0, ...)).

    After differentiating with respect to ``x``, the summation index is shifted
    so that the result is again written as a power series in ``x**n``.

    Parameters
    ----------
    expr : sympy.Sum
        Unevaluated SymPy sum representing the power series.

    x : sympy.Symbol
        Variable with respect to which the series is differentiated.

    order : int, optional
        Derivative order. The default is 1.

    Returns
    -------
    sympy.Sum
        Differentiated and index-shifted power series.

    Raises
    ------
    TypeError
        If ``expr`` is not a SymPy ``Sum``, if ``x`` is not a SymPy symbol,
        or if ``order`` is not an integer.

    ValueError
        If ``order`` is negative or if ``expr`` is not a single-index sum.

    Notes
    -----
    The function is a structural helper for teaching and symbolic
    experimentation. It assumes that the input is a power series in ``x`` with
    one summation index. It does not attempt to prove convergence or justify
    termwise differentiation.

    Examples
    --------
    >>> i = sp.Symbol("i", integer=True, nonnegative=True)
    >>> x = sp.Symbol("x")
    >>> a = sp.Function("a")
    >>> expr = sp.Sum(a(i) * x**i, (i, 0, sp.oo))
    >>> diff_power_series(expr, x)
    Sum((i + 1)*x**i*a(i + 1), (i, 0, oo))
    """
    if not isinstance(expr, sp.Sum):
        raise TypeError(
            f"expr must be a SymPy Sum, got {type(expr).__name__}."
        )

    x = _validate_symbol(x, "x")
    order = _validate_nonnegative_integer(order, "order")

    if len(expr.limits) != 1:
        raise ValueError("expr must be a single-index Sum.")

    summation_index, lower, upper = expr.limits[0]

    if order == 0:
        return expr

    differentiated = sp.diff(expr, x, order)

    if isinstance(differentiated, sp.Mul):
        sum_factors = [
            factor for factor in differentiated.args
            if isinstance(factor, sp.Sum)
        ]

        if len(sum_factors) == 1:
            sum_factor = sum_factors[0]
            other_factors = [
                factor for factor in differentiated.args
                if factor != sum_factor
            ]
            prefactor = sp.Mul(*other_factors)
            differentiated = sp.Sum(
                prefactor * sum_factor.function,
                *sum_factor.limits,
            )

    if not isinstance(differentiated, sp.Sum):
        raise ValueError(
            "The differentiated expression could not be rewritten as a Sum."
        )

    if len(differentiated.limits) != 1:
        raise ValueError("The differentiated expression must be a single-index Sum.")

    new_index, new_lower, new_upper = differentiated.limits[0]

    shifted_summand = (
        differentiated.function
        .simplify()
        .subs(new_index, summation_index + order)
        .factor()
    )

    shifted_lower = lower
    shifted_upper = new_upper - order if new_upper != sp.oo else sp.oo

    return sp.Sum(shifted_summand, (summation_index, shifted_lower, shifted_upper))


if __name__ == "__main__":
    # Small demonstrations when the file is executed directly.
    #
    # The examples are intentionally short. They show the original expression
    # together with the transformed result.

    x, i, n, a = sp.symbols("x i n a")

    print("\n" + "=" * 72)
    print("swap_integral_sum")
    print("=" * 72)

    expr = sp.Integral(sp.Sum(i * x, (i, 1, n)), x)

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(swap_integral_sum(expr))

    print("\n" + "=" * 72)
    print("push_factor_into_sum")
    print("=" * 72)

    expr = sp.Sum(i, (i, 1, n))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(push_factor_into_sum(expr, a, i))

    print("\n" + "=" * 72)
    print("apply_to_summand")
    print("=" * 72)

    expr = sp.Sum((x + 1) * (x + 2), (i, 1, n))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(apply_to_summand(expr, sp.expand))

    print("\n" + "=" * 72)
    print("push_prefactors_into_sums")
    print("=" * 72)

    expr = a * sp.Sum(i + 1, (i, 1, n))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(push_prefactors_into_sums(expr))

    print("\n" + "=" * 72)
    print("split_operator_over_addition")
    print("=" * 72)

    expr = sp.Sum(i + x, (i, 1, n))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(split_operator_over_addition(expr, sp.Sum))

    print("\n" + "=" * 72)
    print("collect_operator_terms")
    print("=" * 72)

    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(collect_operator_terms(expr, sp.Sum, ((i, 1, n),)))

    print("\n" + "=" * 72)
    print("diff_power_series")
    print("=" * 72)

    a_func = sp.Function("a")
    expr = sp.Sum(a_func(i) * x**i, (i, 0, sp.oo))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(diff_power_series(expr, x))


