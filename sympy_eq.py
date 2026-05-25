"""
Convenience equations for elementary symbolic transformations
=============================================================

This file defines an ``Eq`` class that extends ``sympy.Eq`` with arithmetic
operations applied to both sides of an equation.

The class is intended for didactic and interactive use, for example in
courses, notebooks, worksheets, and video material. It makes elementary
equation transformations look close to handwritten algebra:

    Eq(x + 1, 2) - 1
    Eq(2*x, 6) / 2
    Eq(x**2, 9).apply("both", sp.sqrt)

Important
---------
This class is a pragmatic convenience wrapper around ``sympy.Eq``. It keeps
the familiar SymPy equation behavior, but also allows direct assignment to
``lhs`` and ``rhs`` for interactive work.

SymPy expressions are normally treated as immutable objects. Directly
changing ``lhs`` or ``rhs`` is therefore useful in teaching contexts, but it
should not be viewed as a general pattern for low-level SymPy programming.

The arithmetic operations are formal symbolic transformations. In particular,
division by an expression does not check whether that expression can be zero;
users are responsible for the corresponding mathematical side conditions.
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


class Eq(sp.Eq):
    """
    SymPy equation with operations applied to both sides.

    Parameters
    ----------
    lhs : sympy.Expr
        Left-hand side of the equation.

    rhs : sympy.Expr
        Right-hand side of the equation.

    Returns
    -------
    Eq
        Equation object with helper methods for elementary transformations.

    Notes
    -----
    Arithmetic operations with ordinary expressions are applied to both sides.
    Arithmetic operations with another SymPy equation combine corresponding
    sides.

    Division operations are purely symbolic. They do not check whether the
    divisor may be zero. This keeps the class simple for interactive work, but
    mathematical side conditions must be considered separately.

    Examples
    --------
    >>> x = sp.Symbol("x")
    >>> Eq(x + 1, 2) - 1
    Eq(x, 1)
    >>> Eq(2*x, 6) / 2
    Eq(x, 3)
    """


    def __new__(cls, lhs, rhs, **options):
        """
        Create a symbolic equation after validating both sides.

        Parameters
        ----------
        lhs : sympy.Expr
            Left-hand side of the equation.

        rhs : sympy.Expr
            Right-hand side of the equation.

        **options
            Additional options passed to ``sympy.Eq``.

        Returns
        -------
        Eq
            New equation object.

        Raises
        ------
        TypeError
            If ``lhs`` or ``rhs`` cannot be converted to a SymPy expression.
        """
        lhs = _as_expr(lhs, "lhs")
        rhs = _as_expr(rhs, "rhs")

        return super().__new__(cls, lhs, rhs, **options)

    @property
    def lhs(self):
        """
        Return the left-hand side of the equation.

        Returns
        -------
        sympy.Expr
            Left-hand side of the equation.
        """
        return super().lhs

    @lhs.setter
    def lhs(self, lhs):
        """
        Set the left-hand side of the equation.

        Parameters
        ----------
        lhs : sympy.Expr
            New left-hand side.

        Notes
        -----
        This setter is intended for interactive and didactic use. It mutates
        the internal arguments of an object that is normally treated as
        immutable in SymPy. This is convenient in notebooks and teaching
        examples, but should be used with care in library-style code.
        """
        self._args = (_as_expr(lhs, "lhs"), self.rhs)

    @property
    def rhs(self):
        """
        Return the right-hand side of the equation.

        Returns
        -------
        sympy.Expr
            Right-hand side of the equation.
        """
        return super().rhs

    @rhs.setter
    def rhs(self, rhs):
        """
        Set the right-hand side of the equation.

        Parameters
        ----------
        rhs : sympy.Expr
            New right-hand side.

        Notes
        -----
        This setter is intended for interactive and didactic use. It mutates
        the internal arguments of an object that is normally treated as
        immutable in SymPy. This is convenient in notebooks and teaching
        examples, but should be used with care in library-style code.
        """
        self._args = (self.lhs, _as_expr(rhs, "rhs"))

    def __add__(self, other):
        """
        Add an expression or equation to both sides.

        Parameters
        ----------
        other : sympy.Expr or sympy.Equality
            Expression added to both sides, or equation whose corresponding
            sides are added.

        Returns
        -------
        Eq
            New equation.
        """
        if isinstance(other, sp.Equality):
            return Eq(self.lhs + other.lhs, self.rhs + other.rhs)

        other = _as_expr(other, "other")
        return Eq(self.lhs + other, self.rhs + other)

    def __radd__(self, other):
        """
        Add both sides to an expression.

        Parameters
        ----------
        other : sympy.Expr
            Expression added to both sides.

        Returns
        -------
        Eq
            New equation.
        """
        other = _as_expr(other, "other")
        return Eq(other + self.lhs, other + self.rhs)

    def __sub__(self, other):
        """
        Subtract an expression or equation from both sides.

        Parameters
        ----------
        other : sympy.Expr or sympy.Equality
            Expression subtracted from both sides, or equation whose
            corresponding sides are subtracted.

        Returns
        -------
        Eq
            New equation.
        """
        if isinstance(other, sp.Equality):
            return Eq(self.lhs - other.lhs, self.rhs - other.rhs)

        other = _as_expr(other, "other")
        return Eq(self.lhs - other, self.rhs - other)

    def __rsub__(self, other):
        """
        Subtract both sides from an expression.

        Parameters
        ----------
        other : sympy.Expr
            Expression from which both sides are subtracted.

        Returns
        -------
        Eq
            New equation.
        """
        other = _as_expr(other, "other")
        return Eq(other - self.lhs, other - self.rhs)

    def __mul__(self, other):
        """
        Multiply both sides by an expression or another equation.

        Parameters
        ----------
        other : sympy.Expr or sympy.Equality
            Expression multiplied with both sides, or equation whose
            corresponding sides are multiplied.

        Returns
        -------
        Eq
            New equation.
        """
        if isinstance(other, sp.Equality):
            return Eq(self.lhs * other.lhs, self.rhs * other.rhs)

        other = _as_expr(other, "other")
        return Eq(self.lhs * other, self.rhs * other)

    def __rmul__(self, other):
        """
        Multiply an expression by both sides.

        Parameters
        ----------
        other : sympy.Expr
            Expression multiplied with both sides.

        Returns
        -------
        Eq
            New equation.
        """
        other = _as_expr(other, "other")
        return Eq(other * self.lhs, other * self.rhs)

    def __truediv__(self, other):
        """
        Divide both sides by an expression or another equation.

        Parameters
        ----------
        other : sympy.Expr or sympy.Equality
            Expression used as denominator for both sides, or equation whose
            corresponding sides are used as denominators.

        Returns
        -------
        Eq
            New equation.

        Notes
        -----
        This operation does not check whether the divisor can be zero.
        """
        if isinstance(other, sp.Equality):
            return Eq(self.lhs / other.lhs, self.rhs / other.rhs)

        other = _as_expr(other, "other")
        return Eq(self.lhs / other, self.rhs / other)

    def __rtruediv__(self, other):
        """
        Divide an expression by both sides.

        Parameters
        ----------
        other : sympy.Expr
            Numerator divided by both sides.

        Returns
        -------
        Eq
            New equation.

        Notes
        -----
        This operation does not check whether either side can be zero.
        """
        other = _as_expr(other, "other")
        return Eq(other / self.lhs, other / self.rhs)

    def __pow__(self, power):
        """
        Raise both sides to the same power.

        Parameters
        ----------
        power : sympy.Expr
            Exponent applied to both sides.

        Returns
        -------
        Eq
            New equation.
        """
        power = _as_expr(power, "power")
        return Eq(self.lhs**power, self.rhs**power)

    def apply(self, side, func, *args, **kwargs):
        """
        Apply a function to the left-hand side, right-hand side, or both.

        Parameters
        ----------
        side : {"lhs", "rhs", "both"}
            Side to which the function is applied.

        func : callable
            Function applied to the selected side or sides.

        *args, **kwargs
            Additional arguments passed to ``func``.

        Returns
        -------
        Eq
            New equation with transformed side or sides.

        Raises
        ------
        TypeError
            If ``func`` is not callable.

        ValueError
            If ``side`` is not ``"lhs"``, ``"rhs"``, or ``"both"``.
        """
        if not callable(func):
            raise TypeError(
                f"func must be callable, got {type(func).__name__}."
            )

        if side == "lhs":
            return Eq(func(self.lhs, *args, **kwargs), self.rhs)

        if side == "rhs":
            return Eq(self.lhs, func(self.rhs, *args, **kwargs))

        if side == "both":
            return Eq(
                func(self.lhs, *args, **kwargs),
                func(self.rhs, *args, **kwargs),
            )

        raise ValueError(
            "side must be 'lhs', 'rhs', or 'both'."
        )

    def doit(self, **kwargs):
        """
        Evaluate both sides with SymPy's ``doit`` method.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``doit``.

        Returns
        -------
        Eq
            New equation with evaluated sides.
        """
        return Eq(
            sp.sympify(self.lhs).doit(**kwargs),
            sp.sympify(self.rhs).doit(**kwargs),
        )

    def solve(self, symbol):
        """
        Solve the equation for one symbol if the solution is unique.

        This method is intentionally small and didactic. It calls
        ``sympy.solve`` on ``lhs - rhs`` and only accepts the result if
        exactly one solution is returned. Equations with no solution, several
        solutions, or solution sets that SymPy does not return as a single
        expression raise ``ValueError``.

        Parameters
        ----------
        symbol : sympy.Symbol
            Symbol to solve for.

        Returns
        -------
        Eq
            Equation of the form ``Eq(symbol, expression)``.

        Raises
        ------
        TypeError
            If ``symbol`` is not a SymPy symbol.

        ValueError
            If the solution is not unique.
        """
        symbol = _validate_symbol(symbol, "symbol")

        solutions = sp.solve(self.lhs - self.rhs, symbol)

        if len(solutions) != 1:
            raise ValueError(
                f"Cannot solve uniquely for {symbol}: "
                f"found {len(solutions)} solutions."
            )

        return Eq(symbol, solutions[0])


def main():
    """Run a small command-line demonstration."""
    # Small demonstrations when the file is executed directly.
    #
    # The examples are intentionally short. They show the original equation
    # together with the transformed result.

    x, y = sp.symbols("x y")

    print("\n" + "=" * 72)
    print("basic arithmetic on both sides")
    print("=" * 72)

    eq = Eq(x + 1, 2)

    print("Original:")
    print(eq)

    print("\nTransformed:")
    print(eq - 1)

    print("\n" + "=" * 72)
    print("combining two equations")
    print("=" * 72)

    eq1 = Eq(x + 1, 2)
    eq2 = sp.Eq(y, 5)

    print("Original:")
    print(eq1)
    print(eq2)

    print("\nTransformed:")
    print(eq1 + eq2)

    print("\n" + "=" * 72)
    print("applying functions")
    print("=" * 72)

    eq = Eq(x**2, 9)

    print("Original:")
    print(eq)

    print("\nTransformed:")
    print(eq.apply("both", sp.sqrt))

    print("\n" + "=" * 72)
    print("using doit")
    print("=" * 72)

    eq = Eq(sp.Integral(x, x), sp.Integral(2 * x, x))

    print("Original:")
    print(eq)

    print("\nTransformed:")
    print(eq.doit())

    print("\n" + "=" * 72)
    print("solving")
    print("=" * 72)

    eq = Eq(x + y, 3)

    print("Original:")
    print(eq)

    print("\nTransformed:")
    print(eq.solve(x))

    print("\n" + "=" * 72)
    print("directly setting lhs and rhs")
    print("=" * 72)

    eq = Eq(x, 2)

    print("Original:")
    print(eq)

    eq.lhs = y + 1
    eq.rhs = 7

    print("\nTransformed:")
    print(eq)



if __name__ == "__main__":
    main()
