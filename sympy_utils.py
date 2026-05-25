"""
SymPy utility functions
=======================

This file contains didactically oriented helper functions for SymPy.

The main topics are:

- formal series expansions,
- trigonometric transformations,
- integration by parts,
- partial fraction decompositions,
- symbolic manipulations.

The functions are intended for:

- courses and lectures,
- Jupyter notebooks,
- student experiments,
- supplementary material for YouTube videos.

The implementations are designed to be:

- mathematically transparent,
- directly usable,
- easy to modify,
- reasonably robust for typical inputs.

Note
----
This file is deliberately not a full Python package. It is a collection
of standalone utility functions intended for direct practical use.
"""

import sympy as sp
from sympy.simplify.fu import TR10


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


def _validate_positive_integer(value, name):
    """
    Validate that an object is a positive integer.

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
        If ``value`` is not positive.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer, not bool.")

    if not isinstance(value, (int, sp.Integer)):
        raise TypeError(
            f"{name} must be a positive integer, got {type(value).__name__}."
        )

    value = int(value)

    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")

    return value


def formal_series(expr, small_expr, x0=0, n=6, *, dummy=None):
    """
    Compute a formal series expansion in a given expression.

    The function treats ``small_expr`` as a single small parameter,
    even if it is a composite expression such as ``x + y`` or ``f(t)``.

    To do this, ``small_expr`` is temporarily replaced by a dummy symbol,
    the series expansion is computed, and the result is substituted back.

    Parameters
    ----------
    expr : sympy.Expr
        Expression to expand.

    small_expr : sympy.Expr
        Expression treated as the expansion variable. For composite
        expressions, it must occur as an exact structural subexpression of
        ``expr``.

    x0 : sympy.Expr, optional
        Expansion point. The default is 0, consistent with ``sympy.series``.

    n : int, optional
        Expansion order. The default is 6, consistent with ``sympy.series``.

    dummy : sympy.Symbol, optional
        Dummy variable used internally for composite ``small_expr`` values.

    Returns
    -------
    sympy.Expr
        Formal series of ``expr`` in ``small_expr``. If ``small_expr`` is a
        symbol, SymPy's ordinary series including the ``O(...)`` term is
        returned. If ``small_expr`` is composite, only the regular part is
        returned because no reliable ``O(...)`` term in a composite expression
        is constructed.

    Raises
    ------
    TypeError
        If inputs cannot be converted to SymPy expressions, if ``dummy`` is
        not a SymPy symbol, or if ``n`` is not an integer.

    ValueError
        If ``n`` is not positive or if a composite ``small_expr`` is not found
        as an exact structural subexpression of ``expr``.

    Notes
    -----
    This is a *formal* series expansion in ``small_expr`` and not a
    Taylor expansion in the variables from which ``small_expr`` is built.

    Examples
    --------
    >>> x, y = sp.symbols("x y")
    >>> formal_series(sp.exp(x + y), x + y, n=4)
    1 + x + y + (x + y)**2/2 + (x + y)**3/6
    """

    expr = _as_expr(expr, "expr")
    small_expr = _as_expr(small_expr, "small_expr")
    x0 = _as_expr(x0, "x0")
    n = _validate_positive_integer(n, "n")

    # --- Fast path: small_expr is a Symbol -------------------------------
    if small_expr.is_Symbol:
        result = expr.series(small_expr, x0, n)
        return result

    # --- General case: composite expression ------------------------------
    z = dummy if dummy is not None else sp.Dummy("formal_series_var")
    z = _validate_symbol(z, "dummy")

    # Exact structural substitution
    substituted = expr.xreplace({small_expr: z})

    # Safety check
    if substituted == expr and expr != small_expr:
        raise ValueError("small_expr was not found as an exact subexpression in expr")

    # Series expansion in dummy variable
    result = substituted.series(z, x0, n)

    # Regular part (without Big-O). We cannot provide the Big-O
    # part for non-symbol small expressions:
    regular_part = result.removeO().xreplace({z: small_expr})

    return regular_part


def expand_to_half_angle(expr, *angles):
    """
    Rewrite selected trigonometric expressions in half-angle form.

    The transformation is based on the identity

        angle = angle/2 + angle/2

    together with SymPy's angle-addition rule ``TR10``.

    Examples
    --------
    >>> x = sp.symbols("x")
    >>> expand_to_half_angle(sp.sin(x), x)
    2*sin(x/2)*cos(x/2)

    >>> expand_to_half_angle(sp.cos(x), x)
    -sin(x/2)**2 + cos(x/2)**2

    Parameters
    ----------
    expr : sympy expression
        The expression to transform.

    *angles : sympy expressions
        The angles whose trigonometric occurrences should be expanded.

    Returns
    -------
    sympy expression
        The transformed expression.
    """
    expr = _as_expr(expr, "expr")

    if not angles:
        return expr

    result = expr

    for angle in angles:
        angle = _as_expr(angle, "angle")

        a1 = sp.Dummy("half_angle_1")
        a2 = sp.Dummy("half_angle_2")

        tmp = result.xreplace({angle: a1 + a2})
        tmp = TR10(tmp)

        result = tmp.xreplace({
            a1: angle / 2,
            a2: angle / 2,
        })

    return result


def integrate_by_parts(integral, u, vp):
    """
    Apply integration by parts to a one-dimensional integral.

    The integral is assumed to be of the form ``u*vp``, where ``vp`` is
    the derivative of a function ``v`` with respect to the integration variable.

    The transformation uses the formula

        integral u*vp dx = u*v - integral diff(u, x)*v dx

    The function preserves whether the input integral is definite or indefinite.

    Examples
    --------
    >>> x = sp.symbols("x")
    >>> integrate_by_parts(sp.Integral(x * sp.exp(x), x), x, sp.exp(x))
    x*exp(x) - Integral(exp(x), x)

    >>> integrate_by_parts(sp.Integral(x * sp.exp(x), (x, 0, 1)), x, sp.exp(x))
    E - Integral(exp(x), (x, 0, 1))

    Parameters
    ----------
    integral : sympy Integral
        One-dimensional integral whose integrand is ``u*vp``.
    u : sympy expression
        Factor to differentiate.
    vp : sympy expression
        Factor to integrate; interpreted as the derivative of ``v``.

    Returns
    -------
    sympy expression
        The expression obtained by applying integration by parts.

    Notes
    -----
    Only one-dimensional indefinite integrals and one-dimensional integrals
    with lower and upper bounds are supported.

    The boundary terms of definite integrals are computed with limits. This
    handles cases where direct substitution would produce indeterminate forms.
    """
    integral = _as_expr(integral, "integral")
    u = _as_expr(u, "u")
    vp = _as_expr(vp, "vp")

    if not isinstance(integral, sp.Integral):
        raise TypeError("integral must be a SymPy Integral")

    if len(integral.limits) != 1:
        raise ValueError("only one-dimensional integrals are supported")

    limit = integral.limits[0]

    if len(limit) == 1:
        x = limit[0]
        bounds = None
    elif len(limit) == 3:
        x, a, b = limit
        bounds = (a, b)
    else:
        raise ValueError("only indefinite integrals and integrals with two bounds are supported")

    if not x.is_Symbol:
        raise TypeError("the integration variable must be a SymPy Symbol")

    if sp.simplify(integral.function - u * vp) != 0:
        raise ValueError("the integrand must be equal to u*vp")

    v = sp.Integral(vp, x).doit()
    new_integrand = sp.diff(u, x) * v

    # If the new integrand has a leading minus sign, put that sign in front
    # of the integral. This keeps the result closer to standard notation.
    sign = -1
    if new_integrand.could_extract_minus_sign():
        new_integrand = -new_integrand
        sign = 1

    if bounds is None:
        return u * v + sign * sp.Integral(new_integrand, x)

    a, b = bounds
    # We use the limit command to take into account infinite integration
    # limits:
    boundary_term = sp.limit(u * v, x, b) - sp.limit(u * v, x, a)

    return boundary_term + sign * sp.Integral(new_integrand, (x, a, b))

def real_apart(expr, x):
    """
    Compute a real partial fraction decomposition of a rational expression.

    SymPy's complete partial fraction decomposition may contain complex
    linear factors when ``full=True`` is used. This function combines
    conjugate complex terms so that the returned decomposition is real-valued
    for real values of ``x``.

    Examples
    --------
    >>> x = sp.symbols("x")
    >>> real_apart(1 / (x**2 + 1), x)
    1/(x**2 + 1)

    >>> real_apart(1 / (x**3 + 1), x)
    -(x - 2)/(3*(x**2 - x + 1)) + 1/(3*(x + 1))

    Parameters
    ----------
    expr : sympy expression
        Rational expression to decompose.
    x : sympy Symbol
        Variable with respect to which the partial fraction decomposition is
        computed. It is treated as real for the purpose of combining complex
        conjugate terms.

    Returns
    -------
    sympy expression
        Real partial fraction decomposition of ``expr`` with respect to ``x``.

    Notes
    -----
    The input must be rational in ``x``. Internally, the expression is first
    decomposed over the complex numbers using ``apart(..., full=True)``.
    Complex conjugate terms are then paired and added. This produces a real
    decomposition whenever the conjugate pairing succeeds.
    """
    expr = _as_expr(expr, "expr")
    x = _validate_symbol(x, "x")

    if not expr.is_rational_function(x):
        raise ValueError("expr must be rational in x")

    # Work with a real dummy variable so that conjugation treats the
    # decomposition variable as real even if the original symbol has no
    # assumptions.
    y = sp.Dummy("real_apart_var", real=True)
    expr_y = expr.xreplace({x: y})

    decomposed = sp.apart(expr_y, y, full=True).doit()
    terms = decomposed.as_ordered_terms()

    real_terms = []
    used = [False] * len(terms)

    for i, term in enumerate(terms):
        if used[i]:
            continue

        conjugate_term = sp.conjugate(term)

        if sp.simplify(term - conjugate_term) == 0:
            real_terms.append(term)
            used[i] = True
            continue

        for j in range(i + 1, len(terms)):
            if used[j]:
                continue
            if sp.simplify(terms[j] - conjugate_term) == 0:
                real_terms.append(term + terms[j])
                used[i] = True
                used[j] = True
                break
        else:
            raise ValueError("complex partial fraction term has no conjugate pair")

    result = sp.Add(*(sp.factor(sp.cancel(term)) for term in real_terms))

    return result.xreplace({y: x})

if __name__ == "__main__":
    # Small demonstrations when the file is executed directly.
    #
    # Die Beispiele sind bewusst kurz gehalten. Sie zeigen jeweils den
    # ursprünglichen Ausdruck und den transformierten Ausdruck.

    x, y = sp.symbols("x y")

    print("\n" + "=" * 72)
    print("formal_series")
    print("=" * 72)

    expr = sp.exp(x + y)

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(formal_series(expr, x + y, n=4))

    print("\n" + "=" * 72)
    print("expand_to_half_angle")
    print("=" * 72)

    expr = sp.sin(x) + sp.cos(x)

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(expand_to_half_angle(expr, x))

    print("\n" + "=" * 72)
    print("integrate_by_parts")
    print("=" * 72)

    expr = sp.Integral(x * sp.exp(x), x)

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(integrate_by_parts(expr, x, sp.exp(x)))

    print("\n" + "=" * 72)
    print("real_apart")
    print("=" * 72)

    expr = 1 / (x**4 - 1)

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(real_apart(expr, x))

