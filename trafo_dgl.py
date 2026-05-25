"""
Utilities for transforming ordinary differential equations with SymPy
=====================================================================

This file contains didactically oriented helper functions for symbolic
transformations of ordinary differential equations.

The main topics are:

- repeated application of differential operators,
- explicit changes of the independent variable,
- dynamical reparametrizations by a state variable,
- autonomous and non-autonomous second-order systems.

The central idea is that transformations of derivatives must be handled
carefully. In particular,

    (A(x) d/dx)^n

is an operator power, not a naive algebraic substitution. If ``A`` depends
on ``x``, each repeated application produces product-rule terms.

The functions are intended for:

- courses and lectures,
- Jupyter notebooks,
- student experiments,
- supplementary material for videos.

Note
----
This file is deliberately not a full Python package. It is a collection of
standalone utility functions intended for direct practical use.
"""

import sympy as sp


def _validate_symbol(obj, name):
    """
    Validate that an object is a SymPy symbol.

    Parameters
    ----------
    obj : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    sympy.Symbol
        The validated symbol.
    """
    if not isinstance(obj, sp.Symbol):
        raise TypeError(
            f"{name} must be a SymPy Symbol, got {type(obj).__name__}."
        )

    return obj


def _validate_function_class(func, name):
    """
    Validate that an object can be used as a SymPy function class.

    Parameters
    ----------
    func : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    object
        The validated function-like object.
    """
    test_symbol = sp.Symbol("_test_symbol")

    try:
        test_value = func(test_symbol)
    except Exception as exc:
        raise TypeError(
            f"{name} must be a SymPy function class or compatible callable, "
            f"for example sp.Function('U')."
        ) from exc

    if not isinstance(test_value, sp.Expr):
        raise TypeError(
            f"{name} must return a SymPy expression when called with a Symbol."
        )

    return func


def _validate_nonnegative_integer(n, name):
    """
    Validate that an object is a non-negative integer.

    Parameters
    ----------
    n : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    int
        The validated integer as a Python ``int``.
    """
    if isinstance(n, bool):
        raise TypeError(f"{name} must be a non-negative integer, not bool.")

    if not isinstance(n, (int, sp.Integer)):
        raise TypeError(
            f"{name} must be a non-negative integer, got {type(n).__name__}."
        )

    n = int(n)

    if n < 0:
        raise ValueError(f"{name} must be non-negative, got {n}.")

    return n


def _validate_boolean(value, name):
    """
    Validate that an object is a Boolean value.

    Parameters
    ----------
    value : object
        Object to validate.

    name : str
        Name used in the error message.

    Returns
    -------
    bool
        The validated Boolean value.
    """
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool, got {type(value).__name__}.")

    return value


def _as_zero_expression(eq):
    """
    Convert an equation or expression to expression form ``F = 0``.

    Parameters
    ----------
    eq : sympy.Eq or sympy.Expr
        Equation or expression. If an expression is passed, it is interpreted
        as ``eq = 0``.

    Returns
    -------
    sympy.Expr
        The expression ``lhs - rhs`` for equations, otherwise ``sympify(eq)``.

    Raises
    ------
    TypeError
        If ``eq`` cannot be converted to a SymPy expression.
    """
    if isinstance(eq, sp.Equality):
        return eq.lhs - eq.rhs

    try:
        expr = sp.sympify(eq)
    except Exception as exc:
        raise TypeError(
            "eq must be a SymPy equation or an object that can be converted "
            "to a SymPy expression."
        ) from exc

    if not isinstance(expr, sp.Expr):
        raise TypeError(
            f"eq must be a SymPy equation or expression, got {type(eq).__name__}."
        )

    return expr


def _derivative_order_in_var(derivative, var):
    """
    Count how many times a derivative differentiates with respect to ``var``.

    Parameters
    ----------
    derivative : sympy.Derivative
        Derivative object to inspect.

    var : sympy.Symbol
        Variable with respect to which the differentiation order is counted.

    Returns
    -------
    int
        Total differentiation order with respect to ``var``.
    """
    return sum(
        v[1] if isinstance(v, tuple) and v[0] == var
        else (1 if v == var else 0)
        for v in derivative.variables
    )


def _max_derivative_order_of_function(func, var, expr):
    """
    Find the maximum derivative order of ``func(var)`` appearing in ``expr``.

    Parameters
    ----------
    func : sympy.Function
        Function symbol, for example ``U``.

    var : sympy.Symbol
        Independent variable, for example ``theta``.

    expr : sympy.Expr
        Expression to inspect.

    Returns
    -------
    int
        Maximum derivative order. Returns 0 if no such derivative occurs.
    """
    max_order = 0

    for derivative in expr.atoms(sp.Derivative):
        if derivative.expr == func(var):
            order = _derivative_order_in_var(derivative, var)
            max_order = max(max_order, order)

    return max_order


def apply_operator_forward(E, A, x, n):
    """
    Apply the differential operator ``(A(x) d/dx)^n`` to an expression.

    Mathematical setting
    --------------------
    The function implements the operator identity

        d/dtheta = A(x) d/dx,

    where typically

        A(x) = dx/dtheta.

    Consequently,

        (d/dtheta)^n E = (A(x) d/dx)^n E.

    Important
    ---------
    This is an operator identity, not a naive substitution. If ``A`` depends
    on ``x``, repeated applications generate product-rule terms.

    Parameters
    ----------
    E : sympy.Expr
        Expression to which the operator is applied.

    A : sympy.Expr
        Coefficient of the differential operator.

    x : sympy.Symbol
        Differentiation variable.

    n : int
        Number of repeated applications. For ``n = 0`` the expression is
        returned unchanged.

    Returns
    -------
    sympy.Expr
        The expression obtained by applying ``A(x) d/dx`` exactly ``n`` times.

    Raises
    ------
    TypeError
        If ``x`` is not a SymPy symbol or ``n`` is not an integer.

    ValueError
        If ``n`` is negative.

    Examples
    --------
    >>> x = sp.Symbol("x")
    >>> A = x
    >>> apply_operator_forward(sp.Function("f")(x), A, x, 2)
    x**2*Derivative(f(x), (x, 2)) + x*Derivative(f(x), x)
    """

    x = _validate_symbol(x, "x")
    n = _validate_nonnegative_integer(n, "n")
    E = sp.sympify(E)
    A = sp.sympify(A)

    if n == 0:
        return E

    result = E

    for _ in range(n):
        result = A * sp.diff(result, x)
        result = sp.expand(result)

    return result


def transform_ode(eq, old_var, new_sym, f_theta, g_x, old_func, new_func):
    """
    Transform a scalar ODE under an explicit change of independent variable.

    Mathematical setting
    --------------------
    The independent variable is changed explicitly by

        x = f(theta),     theta = g(x).

    Thus the new variable is known as a concrete expression in the old
    variable, and the inverse relation is also supplied explicitly. The
    dependent function is transformed as

        U(theta) = V(x).

    The function distinguishes two different operations:

    1. Derivatives of the dependent function are transformed by the chain
       rule.

    2. Remaining derivatives with respect to ``theta`` are pushed forward by
       the operator identity

           d/dtheta = (dx/dtheta) d/dx.

    Important
    ---------
    These two steps should not be confused. Derivatives of the dependent
    function require the chain rule, while derivatives of general expressions
    can be transformed using the operator pushforward.

    Parameters
    ----------
    eq : sympy.Eq or sympy.Expr
        Original ODE in the variable ``old_var``. If an expression is passed,
        it is interpreted as ``eq = 0``.

    old_var : sympy.Symbol
        Original independent variable, for example ``theta``.

    new_sym : sympy.Symbol
        New independent variable, for example ``x``.

    f_theta : sympy.Expr
        Forward transformation ``x = f(theta)``.

    g_x : sympy.Expr
        Inverse transformation ``theta = g(x)``.

    old_func : sympy.Function
        Dependent function in the original equation, for example ``U``.

    new_func : sympy.Function
        Dependent function in the transformed equation, for example ``V``.

    Returns
    -------
    sympy.Eq
        Transformed equation in the form ``F(x, V, V', ...) = 0``.

    Raises
    ------
    TypeError
        If variables are not SymPy symbols or function arguments are not
        SymPy function classes.

    Examples
    --------
    >>> theta, x = sp.symbols("theta x")
    >>> U = sp.Function("U")
    >>> V = sp.Function("V")
    >>> eq = sp.Eq(sp.diff(U(theta), theta, 2) + U(theta), 0)
    >>> transform_ode(eq, theta, x, sp.exp(theta), sp.log(x), U, V)
    Eq(x**2*Derivative(V(x), (x, 2)) + x*Derivative(V(x), x) + V(x), 0)
    """

    old_var = _validate_symbol(old_var, "old_var")
    new_sym = _validate_symbol(new_sym, "new_sym")
    old_func = _validate_function_class(old_func, "old_func")
    new_func = _validate_function_class(new_func, "new_func")
    f_theta = sp.sympify(f_theta)
    g_x = sp.sympify(g_x)

    expr = _as_zero_expression(eq)

    # ------------------------------------------------------------
    # 1) Transform derivatives of the dependent function only.
    # ------------------------------------------------------------
    # These derivatives must be handled via the chain rule, not via
    # the operator identity.
    # ------------------------------------------------------------

    max_order = _max_derivative_order_of_function(old_func, old_var, expr)

    if max_order > 0:
        # Introduce a dummy variable u(theta) to avoid illegal derivatives
        # such as d/d(cos(theta)).
        u = sp.Function("u")(old_var)

        # Precompute derivatives of x = f(theta).
        f_derivatives = {
            k: sp.diff(f_theta, (old_var, k))
            for k in range(1, max_order + 1)
        }

        # Build chain-rule expressions recursively.
        chain = {}
        chain[1] = sp.diff(new_func(u), u) * f_derivatives[1]

        for n in range(2, max_order + 1):
            chain[n] = sp.diff(chain[n - 1], old_var)

        # Replace u-derivatives and u(theta) consistently.
        replacement_map = {
            sp.Derivative(new_func(u), u): sp.Derivative(new_func(new_sym), new_sym)
        }

        for k in range(2, max_order + 1):
            replacement_map[
                sp.Derivative(new_func(u), (u, k))
            ] = sp.Derivative(new_func(new_sym), (new_sym, k))

        for k in range(1, max_order + 1):
            replacement_map[
                sp.Derivative(u, (old_var, k))
            ] = f_derivatives[k]

        chain = {n: chain[n].xreplace(replacement_map) for n in chain}

        # Apply substitutions for derivatives of the dependent function.
        for n in range(1, max_order + 1):
            expr = expr.subs(
                sp.Derivative(old_func(old_var), (old_var, n)),
                chain[n],
            )

    # Always transform the dependent function itself, also for zero-order ODEs.
    expr = expr.subs(old_func(old_var), new_func(new_sym))

    # ------------------------------------------------------------
    # 2) Transform remaining theta-derivatives via operator pushforward.
    # ------------------------------------------------------------
    # At this point, all derivatives of new_func(x) are already correct.
    # ------------------------------------------------------------

    A = sp.diff(f_theta, old_var)  # dx/dtheta

    derivatives_to_transform = [
        derivative for derivative in expr.atoms(sp.Derivative)
        if _derivative_order_in_var(derivative, old_var) > 0
    ]

    for derivative in derivatives_to_transform:
        n = _derivative_order_in_var(derivative, old_var)

        E = derivative.expr
        E_x = E.subs(old_var, g_x)
        A_x = A.subs(old_var, g_x)

        replacement = apply_operator_forward(E_x, A_x, new_sym, n)

        # Structural replacement avoids accidental rewriting inside unrelated
        # subexpressions.
        expr = expr.xreplace({derivative: replacement})

    # ------------------------------------------------------------
    # 3) Final substitution theta -> g(x).
    # ------------------------------------------------------------

    expr = expr.subs(old_var, g_x)

    return sp.Eq(sp.simplify(sp.trigsimp(expr)), 0, evaluate=False)


def reparametrize_by_state(eq, old_var, state_func, velocity_func, new_sym, new_velocity_func):
    """
    Reparametrize an autonomous second-order system by the state variable.

    Mathematical setting
    --------------------
    This function implements the standard reduction of an autonomous system

        dX/dt = v,
        dv/dt = F(X, v),

    by using the state variable itself as the new independent variable.
    Introduce

        v(t) = V(x),     where x = X(t).

    Along a solution curve,

        d/dt = (dX/dt) d/dx = v d/dx = V(x) d/dx.

    Therefore,

        dv/dt       -> V(x) dV/dx,
        d^2v/dt^2   -> (V(x) d/dx)^2 V(x),
        and so on.

    Important
    ---------
    This is not an explicit coordinate transformation ``x = f(t)``. Instead,
    the new independent variable is the state variable along a solution curve.

    The function is intended for autonomous equations. If explicit occurrences
    of ``old_var`` remain after the transformation, the result is deliberately
    left as a non-closed equation for ``V(x)``. Use
    ``reparametrize_nonautonomous_by_state`` when the old time variable should
    be replaced by an additional unknown function ``T(x)``.

    Parameters
    ----------
    eq : sympy.Eq or sympy.Expr
        Equation or expression involving ``X(t)``, ``v(t)``, and time
        derivatives. If an expression is passed, it is interpreted as
        ``eq = 0``.

    old_var : sympy.Symbol
        Original independent variable, usually ``t``.

    state_func : sympy.Function
        State function ``X``, used as ``X(t)``.

    velocity_func : sympy.Function
        Velocity function ``v``, used as ``v(t)``.

    new_sym : sympy.Symbol
        New independent variable, usually ``x``.

    new_velocity_func : sympy.Function
        New velocity function ``V``, used as ``V(x)``.

    Returns
    -------
    sympy.Eq
        Transformed equation in the form ``F(x, V, V', ...) = 0``.

    Raises
    ------
    TypeError
        If variables are not SymPy symbols or function arguments are not
        SymPy function classes.

    Examples
    --------
    >>> t, x, omega = sp.symbols("t x omega")
    >>> X = sp.Function("X")
    >>> v = sp.Function("v")
    >>> V = sp.Function("V")
    >>> eq = sp.Eq(sp.diff(v(t), t), -omega**2 * X(t))
    >>> reparametrize_by_state(eq, t, X, v, x, V)
    Eq(omega**2*x + V(x)*Derivative(V(x), x), 0)
    """

    old_var = _validate_symbol(old_var, "old_var")
    new_sym = _validate_symbol(new_sym, "new_sym")
    state_func = _validate_function_class(state_func, "state_func")
    velocity_func = _validate_function_class(velocity_func, "velocity_func")
    new_velocity_func = _validate_function_class(
        new_velocity_func,
        "new_velocity_func",
    )

    expr = _as_zero_expression(eq)

    X_t = state_func(old_var)
    v_t = velocity_func(old_var)

    V_x = new_velocity_func(new_sym)

    # The dynamical pushforward operator is d/dt = V(x) d/dx.
    A = V_x

    derivatives_to_transform = [
        derivative for derivative in expr.atoms(sp.Derivative)
        if _derivative_order_in_var(derivative, old_var) > 0
    ]

    # Sort higher-order derivatives first. This avoids replacing a lower-order
    # derivative inside a higher-order one too early.
    derivatives_to_transform = sorted(
        derivatives_to_transform,
        key=lambda derivative: _derivative_order_in_var(derivative, old_var),
        reverse=True,
    )

    replacement_map = {}

    for derivative in derivatives_to_transform:
        n = _derivative_order_in_var(derivative, old_var)

        if derivative.expr == X_t:
            if n == 1:
                # dX/dt = v -> V(x)
                replacement_map[derivative] = V_x
            else:
                # d^n X/dt^n = d^(n-1) v/dt^(n-1)
                replacement_map[derivative] = apply_operator_forward(V_x, A, new_sym, n - 1)

        elif derivative.expr == v_t:
            # d^n v/dt^n = (V(x) d/dx)^n V(x)
            replacement_map[derivative] = apply_operator_forward(V_x, A, new_sym, n)

    expr = expr.xreplace(replacement_map)

    expr = expr.subs({
        X_t: new_sym,
        v_t: V_x,
    })

    # If old_var still occurs here, the equation was not autonomous. We
    # deliberately leave it untouched instead of inventing an additional
    # function T(x). That belongs to the non-autonomous transformation.
    return sp.Eq(sp.simplify(sp.trigsimp(expr)), 0, evaluate=False)


def reparametrize_nonautonomous_by_state(
    eq,
    old_var,
    state_func,
    velocity_func,
    new_sym,
    new_time_func,
    new_velocity_func,
    return_auxiliary=True,
):
    """
    Reparametrize a non-autonomous system by the state variable.

    Mathematical setting
    --------------------
    This function implements the non-autonomous extension of the autonomous
    transformation

        dX/dt = v,
        dv/dt = F(t, X, v).

    Since the old independent variable ``t`` appears explicitly, it cannot
    simply disappear. The transformed problem is usually a coupled system, not
    a single closed equation. Introduce

        t = T(x),     v(t) = V(x),     x = X(t).

    Along a solution curve,

        d/dt = (dX/dt) d/dx = v d/dx = V(x) d/dx.

    Therefore,

        dv/dt       -> V(x) dV/dx,
        d^2v/dt^2   -> (V(x) d/dx)^2 V(x),
        and so on.

    The explicit old time variable becomes

        t -> T(x).

    The missing relation for ``T(x)`` follows from ``dX/dt = v``:

        dt/dx = 1/v = 1/V(x),

    equivalently,

        V(x) T'(x) - 1 = 0.

    Important
    ---------
    A non-autonomous equation usually does not transform into a single closed
    ODE for ``V(x)``. It transforms into a coupled system for ``V(x)`` and
    ``T(x)``.

    Parameters
    ----------
    eq : sympy.Eq or sympy.Expr
        Equation or expression involving ``t``, ``X(t)``, ``v(t)``, and time
        derivatives. If an expression is passed, it is interpreted as
        ``eq = 0``.

    old_var : sympy.Symbol
        Original independent variable, usually ``t``.

    state_func : sympy.Function
        State function ``X``, used as ``X(t)``.

    velocity_func : sympy.Function
        Velocity function ``v``, used as ``v(t)``.

    new_sym : sympy.Symbol
        New independent variable, usually ``x``.

    new_time_func : sympy.Function
        New time function ``T``, used as ``T(x)``.

    new_velocity_func : sympy.Function
        New velocity function ``V``, used as ``V(x)``.

    return_auxiliary : bool, optional
        If True, return a tuple ``(main_equation, auxiliary_equation)``.
        The auxiliary equation is ``V(x)*T'(x) - 1 = 0``. If False, return
        only the transformed main equation.

    Returns
    -------
    sympy.Eq or tuple[sympy.Eq, sympy.Eq]
        Transformed main equation, optionally together with the auxiliary
        equation for ``T(x)``.

    Raises
    ------
    TypeError
        If variables are not SymPy symbols, function arguments are not SymPy
        function classes, or ``return_auxiliary`` is not a Boolean value.

    Examples
    --------
    >>> t, x = sp.symbols("t x")
    >>> X = sp.Function("X")
    >>> v = sp.Function("v")
    >>> T = sp.Function("T")
    >>> V = sp.Function("V")
    >>> eq = sp.Eq(sp.diff(v(t), t), t * X(t))
    >>> reparametrize_nonautonomous_by_state(eq, t, X, v, x, T, V)
    (Eq(-x*T(x) + V(x)*Derivative(V(x), x), 0), Eq(V(x)*Derivative(T(x), x) - 1, 0))
    """

    old_var = _validate_symbol(old_var, "old_var")
    new_sym = _validate_symbol(new_sym, "new_sym")
    state_func = _validate_function_class(state_func, "state_func")
    velocity_func = _validate_function_class(velocity_func, "velocity_func")
    new_time_func = _validate_function_class(new_time_func, "new_time_func")
    new_velocity_func = _validate_function_class(
        new_velocity_func,
        "new_velocity_func",
    )
    return_auxiliary = _validate_boolean(return_auxiliary, "return_auxiliary")

    expr = _as_zero_expression(eq)

    X_t = state_func(old_var)
    v_t = velocity_func(old_var)

    T_x = new_time_func(new_sym)
    V_x = new_velocity_func(new_sym)

    # The dynamical pushforward operator is d/dt = V(x) d/dx.
    A = V_x

    derivatives_to_transform = [
        derivative for derivative in expr.atoms(sp.Derivative)
        if _derivative_order_in_var(derivative, old_var) > 0
    ]

    derivatives_to_transform = sorted(
        derivatives_to_transform,
        key=lambda derivative: _derivative_order_in_var(derivative, old_var),
        reverse=True,
    )

    replacement_map = {}

    for derivative in derivatives_to_transform:
        n = _derivative_order_in_var(derivative, old_var)

        if derivative.expr == X_t:
            if n == 1:
                replacement_map[derivative] = V_x
            else:
                replacement_map[derivative] = apply_operator_forward(V_x, A, new_sym, n - 1)

        elif derivative.expr == v_t:
            replacement_map[derivative] = apply_operator_forward(V_x, A, new_sym, n)

    expr = expr.xreplace(replacement_map)

    expr = expr.subs({
        X_t: new_sym,
        v_t: V_x,
        old_var: T_x,
    })

    main_eq = sp.Eq(sp.simplify(sp.trigsimp(expr)), 0, evaluate=False)

    auxiliary_expr = V_x * sp.diff(T_x, new_sym) - 1
    auxiliary_eq = sp.Eq(sp.simplify(auxiliary_expr), 0, evaluate=False)

    if return_auxiliary:
        return main_eq, auxiliary_eq

    return main_eq


def main():
    """Run small demonstrations for direct execution of this file."""
    # The examples are intentionally short. They are meant to show the input
    # equation or expression together with the transformed result.

    print("\n" + "=" * 72)
    print("apply_operator_forward")
    print("=" * 72)

    x = sp.Symbol("x")
    f = sp.Function("f")
    A = x

    expr = f(x)

    print("Original:")
    print(expr)

    print("\nOperator:")
    print("(x d/dx)^2")

    print("\nTransformed:")
    print(apply_operator_forward(expr, A, x, 2))

    print("\n" + "=" * 72)
    print("transform_ode")
    print("=" * 72)

    theta = sp.Symbol("theta")
    U = sp.Function("U")
    V = sp.Function("V")

    eq = sp.Eq(sp.diff(U(theta), theta, 2) + U(theta), 0)

    print("Original:")
    print(eq)

    print("\nTransformation:")
    print("x = exp(theta), theta = log(x)")

    print("\nTransformed:")
    print(transform_ode(eq, theta, x, sp.exp(theta), sp.log(x), U, V))

    print("\n" + "=" * 72)
    print("reparametrize_by_state")
    print("=" * 72)

    t = sp.Symbol("t")
    omega = sp.Symbol("omega")
    X = sp.Function("X")
    v = sp.Function("v")

    eq = sp.Eq(sp.diff(v(t), t), -omega**2 * X(t))

    print("Original:")
    print(eq)

    print("\nTransformed:")
    print(reparametrize_by_state(eq, t, X, v, x, V))

    print("\n" + "=" * 72)
    print("reparametrize_nonautonomous_by_state")
    print("=" * 72)

    T = sp.Function("T")

    eq = sp.Eq(sp.diff(v(t), t), t * X(t))

    print("Original:")
    print(eq)

    print("\nTransformed:")
    main_eq, auxiliary_eq = reparametrize_nonautonomous_by_state(eq, t, X, v, x, T, V)
    print(main_eq)
    print(auxiliary_eq)


if __name__ == "__main__":
    main()
