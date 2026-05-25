"""
Tools for power series manipulations with SymPy
===============================================

This file contains helper tools for symbolic manipulations of power series.

This file contains two main tools:

- ``diff_power_series`` differentiates formal power series and shifts the
  summation index back to powers of ``x**n``.
- ``PowerSumNormalizer`` normalizes sums containing powers such as
  ``x**(k + s)`` to a common target exponent such as ``x**n``.

This is useful for:

- power series methods,
- coefficient comparison,
- recurrence relations,
- symbolic solutions of differential equations by series ansatz.
"""

import sympy as sp

from sum_utils import push_prefactors_into_sums, split_operator_over_addition


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


def diff_power_series(expr, x, order=1):
    """
    Differentiate a power series and shift the summation index.

    The function is intended for unevaluated SymPy sums representing power
    series of the form

        Sum(a(n) * x**n, (n, 0, ...)).

    After differentiating with respect to ``x``, the summation index is shifted
    so that the result is again written as a power series in ``x**n``. This is
    a structural helper: it assumes that the differentiated expression remains
    representable as a single SymPy ``Sum``.

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
    one summation index and powers of the form ``x**k``. It does not attempt
    to prove convergence or justify termwise differentiation. If SymPy
    rewrites the derivative into a form that is no longer a single sum, the
    function raises ``ValueError``.

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


class PowerSumNormalizer:
    """
    Normalize power sums to a common target exponent.

    The class transforms sums of the form

        Sum(f(k) * x**(k + s), (k, a, b))

    to sums whose power of ``x`` is written with a common target exponent,
    for example ``x**n``. This is done by shifting the summation index and
    adjusting the summation limits.

    Parameters
    ----------
    x : sympy.Symbol
        Base variable of the power series.

    target_exp : sympy.Expr
        Target exponent, for example ``n`` or ``n + alpha``.

    new_index : sympy.Symbol, optional
        New summation index. If omitted, it is inferred from the free symbol
        in ``target_exp``.

    Examples
    --------
    >>> x, k, n = sp.symbols("x k n")
    >>> a = sp.Function("a")
    >>> normalizer = PowerSumNormalizer(x, n)
    >>> expr = sp.Sum(a(k) * x**(k + 2), (k, 0, sp.oo))
    >>> normalizer.normalize(expr)
    Sum(x**n*a(n - 2), (n, 2, oo))
    """

    def __init__(self, x, target_exp, new_index=None):
        self.x = _validate_symbol(x, "x")
        self.target_exp = _as_expr(target_exp, "target_exp")

        if new_index is None:
            free_symbols = list(self.target_exp.free_symbols)

            if len(free_symbols) != 1:
                raise ValueError(
                    "target_exp must contain exactly one free symbol if "
                    "new_index is not provided."
                )

            new_index = free_symbols[0]

        self.new_index = _validate_symbol(new_index, "new_index")

    def _is_linear_in(self, expr, var):
        """
        Check whether an expression is linear in a variable.
        """
        expr = _as_expr(expr, "expr")
        var = _validate_symbol(var, "var")

        coefficient = expr.coeff(var)
        remainder = sp.simplify(expr - coefficient * var)

        return bool(coefficient.is_number and remainder.is_number)

    def _extract_shift(self, summand, index):
        """
        Extract the exponent shift of ``x`` in a summand.

        If the summand contains ``x**(k + beta)`` and the target exponent is
        ``n + alpha``, then the returned shift is ``beta - alpha``. The old
        index is replaced by ``n - shift``, so that the exponent becomes the
        requested target exponent.

        Examples
        --------
        For target exponent ``n``, the term ``x**(k + 2)`` has shift ``2``
        and therefore uses the replacement ``k -> n - 2``. For target
        exponent ``n + alpha``, the same term has shift ``2 - alpha``.

        Only exponent shifts of the form ``k + constant`` are supported.
        Direct multiplicative factors ``x`` are included in the total exponent,
        so ``x * x**k`` is treated like ``x**(k + 1)``. Exponents such as
        ``2*k`` or nonlinear expressions in ``k`` are not treated as index
        shifts.
        """
        summand = _as_expr(summand, "summand")
        index = _validate_symbol(index, "index")

        if isinstance(summand, sp.Add):
            shifts = {
                sp.simplify(self._extract_shift(term, index))
                for term in summand.args
            }

            if len(shifts) != 1:
                raise ValueError(
                    "Inconsistent exponent shifts within one summand."
                )

            return shifts.pop()

        target_offset = sp.simplify(self.target_exp - self.new_index)

        exponent = sp.Integer(0)
        found_power = False

        for factor in sp.Mul.make_args(summand):
            if factor == self.x:
                exponent += 1
                found_power = True
                continue

            if isinstance(factor, sp.Pow) and factor.base == self.x:
                exponent += factor.exp
                found_power = True

        if not found_power:
            return sp.Integer(0)

        exponent = sp.simplify(exponent)

        if not self._is_linear_in(exponent, index):
            return sp.Integer(0)

        if sp.simplify(exponent.coeff(index) - 1) != 0:
            raise ValueError(
                "The exponent must be a shift of the summation index, "
                "for example k + s with coefficient 1 in k."
            )

        exponent_offset = sp.simplify(exponent - index)
        return sp.simplify(exponent_offset - target_offset)

    def _normalize_single_sum(self, sum_expr):
        """
        Normalize one SymPy sum.

        Parameters
        ----------
        sum_expr : sympy.Sum
            Sum to normalize.

        Returns
        -------
        sympy.Sum
            Sum with shifted index and shifted limits.
        """
        if not isinstance(sum_expr, sp.Sum):
            raise TypeError(
                f"sum_expr must be a SymPy Sum, got {type(sum_expr).__name__}."
            )

        if len(sum_expr.limits) != 1:
            raise ValueError("Only single-index sums can be normalized.")

        old_index, lower, upper = sum_expr.limits[0]
        summand = sp.simplify(sum_expr.function)

        shift = self._extract_shift(summand, old_index)

        old_index_replacement = self.new_index - shift
        new_summand = summand.subs(old_index, old_index_replacement)

        new_lower = sp.simplify(lower + shift)
        new_upper = sp.oo if upper == sp.oo else sp.simplify(upper + shift)

        return sp.Sum(new_summand, (self.new_index, new_lower, new_upper))

    def normalize(self, expr):
        """
        Normalize all matching power sums in an expression.

        The expression is first prepared by pushing prefactors into sums, for
        example ``x * Sum(...) -> Sum(x * ...)`` when this is safe. Summands
        are expanded enough to expose additions, and sums whose summands are
        additions are then split into separate sums. Finally each resulting sum
        is normalized to the configured target exponent.

        Non-sum terms are preserved. The method is structural and assumes
        power factors of the form ``x**(k + s)`` or plain factors ``x``. It
        does not check convergence or mathematical validity of the original
        series.

        Parameters
        ----------
        expr : sympy.Expr
            Expression containing power sums.

        Returns
        -------
        sympy.Expr
            Expression with normalized sums. If no sums are found, the
            original expression is returned unchanged.
        """
        expr = _as_expr(expr, "expr")

        expr = push_prefactors_into_sums(expr)

        expanded_sums = {
            sum_expr: sp.Sum(sp.expand(sum_expr.function), *sum_expr.limits)
            for sum_expr in expr.find(sp.Sum)
        }

        if expanded_sums:
            expr = expr.xreplace(expanded_sums)

        expr = split_operator_over_addition(expr, sp.Sum)

        replacements = {
            sum_expr: self._normalize_single_sum(sum_expr)
            for sum_expr in expr.find(sp.Sum)
        }

        if not replacements:
            return expr

        return expr.xreplace(replacements)


if __name__ == "__main__":
    # Small demonstrations when the file is executed directly.
    #
    # The examples are intentionally short. They show the original expression
    # together with the transformed result.

    x, k, n = sp.symbols("x k n")
    a = sp.Function("a")
    b = sp.Function("b")

    print("\n" + "=" * 72)
    print("diff_power_series")
    print("=" * 72)

    expr = sp.Sum(a(k) * x**k, (k, 0, sp.oo))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(diff_power_series(expr, x))

    normalizer = PowerSumNormalizer(x, n)

    print("\n" + "=" * 72)
    print("single shifted power sum")
    print("=" * 72)

    expr = sp.Sum(a(k) * x**(k + 2), (k, 0, sp.oo))

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(normalizer.normalize(expr))

    print("\n" + "=" * 72)
    print("expression with several sums")
    print("=" * 72)

    expr = (
        sp.Sum(a(k) * x**(k + 1), (k, 0, sp.oo))
        + sp.Sum(b(k) * x**(k + 2), (k, 0, sp.oo))
    )

    print("Original:")
    print(expr)

    print("\nTransformed:")
    print(normalizer.normalize(expr))
