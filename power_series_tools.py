"""
Tools for power series manipulations with SymPy
===============================================

This file contains helper tools for symbolic manipulations of power series.

The main class is ``PowerSumNormalizer``. It normalizes sums containing
powers such as ``x**(k + s)`` to a common target exponent such as ``x**n``.

This is useful for:

- power series methods,
- coefficient comparison,
- recurrence relations,
- symbolic solutions of differential equations by series ansatz.
"""

import sympy as sp

from sum_utils import split_operator_over_addition, push_prefactors_into_sums


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
        index is then replaced by ``n - shift``.
        """
        summand = _as_expr(summand, "summand")
        index = _validate_symbol(index, "index")

        shifts = []

        powers = list(summand.atoms(sp.Pow))

        # SymPy represents a plain factor x not as Pow(x, 1). Treat it as
        # x**1 when it occurs as a direct multiplicative factor.
        direct_x_factors = [
            factor for factor in sp.Mul.make_args(summand)
            if factor == self.x
        ]
        powers.extend(sp.Pow(self.x, 1, evaluate=False) for _ in direct_x_factors)

        target_offset = sp.simplify(self.target_exp - self.new_index)

        for power in powers:
            if power.base != self.x:
                continue

            exponent = power.exp

            if not self._is_linear_in(exponent, index):
                continue

            if sp.simplify(exponent.coeff(index) - 1) != 0:
                raise ValueError(
                    "The exponent must be linear with coefficient 1 in "
                    "the summation index."
                )

            exponent_offset = sp.simplify(exponent - index)
            shifts.append(sp.simplify(exponent_offset - target_offset))

        if not shifts:
            return sp.Integer(0)

        unique_shifts = {sp.simplify(shift) for shift in shifts}

        if len(unique_shifts) != 1:
            raise ValueError(
                "Inconsistent exponent shifts within one summand."
            )

        return unique_shifts.pop()

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

        The expression is first prepared by pushing prefactors into sums and
        splitting sums over additions. Then each sum is normalized separately.

        Non-sum terms are preserved.

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
        expr = split_operator_over_addition(expr, sp.Sum)

        replacements = {
            sum_expr: self._normalize_single_sum(sum_expr)
            for sum_expr in expr.find(sp.Sum)
        }

        if not replacements:
            return expr

        return expr.xreplace(replacements)


if __name__ == "__main__":
    x, k, n = sp.symbols("x k n")
    a = sp.Function("a")
    b = sp.Function("b")

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
