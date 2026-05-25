"""
test_trafo_dgl.py
=================

Pytest test suite for trafo_dgl.py.

Run from the directory containing trafo_dgl.py with

    pytest -q

The tests focus on mathematical equivalence rather than exact printed form,
because SymPy may choose different but equivalent representations.
"""

import sympy as sp

from trafo_dgl import (
    apply_operator_forward,
    transform_ode,
    reparametrize_by_state,
)


def assert_expr_equal(actual, expected):
    """
    Assert symbolic equality of two SymPy expressions.

    We use simplify/trigsimp because many transformations produce equivalent
    expressions with different syntactic structure.
    """
    diff = sp.trigsimp(sp.simplify(actual - expected))
    assert diff == 0, f"\nactual:   {actual}\nexpected: {expected}\ndiff:     {diff}"


def assert_eq_lhs_equal(eq, expected_lhs):
    """
    Assert that an equation has the form expected_lhs = 0 up to simplification.
    """
    assert isinstance(eq, sp.Equality)
    assert eq.rhs == 0
    assert_expr_equal(eq.lhs, expected_lhs)


# ---------------------------------------------------------------------------
# Tests for apply_operator_forward
# ---------------------------------------------------------------------------


def test_apply_operator_forward_order_zero_is_identity():
    x = sp.Symbol("x")
    E = sp.sin(x)
    A = x**2

    result = apply_operator_forward(E, A, x, 0)

    assert result == E


def test_apply_operator_forward_first_order():
    x = sp.Symbol("x")
    E = sp.sin(x)
    A = x**2

    result = apply_operator_forward(E, A, x, 1)
    expected = x**2 * sp.cos(x)

    assert_expr_equal(result, expected)


def test_apply_operator_forward_second_order_includes_product_rule_term():
    x = sp.Symbol("x")
    E = sp.Function("E")
    A = sp.Function("A")

    result = apply_operator_forward(E(x), A(x), x, 2)
    expected = A(x) * sp.diff(A(x), x) * sp.diff(E(x), x) + A(x)**2 * sp.diff(E(x), x, 2)

    assert_expr_equal(result, expected)


def test_apply_operator_forward_third_order_for_A_equal_x():
    x = sp.Symbol("x")
    E = sp.Function("E")

    result = apply_operator_forward(E(x), x, x, 3)
    expected = x * sp.diff(E(x), x) + 3*x**2 * sp.diff(E(x), x, 2) + x**3 * sp.diff(E(x), x, 3)

    assert_expr_equal(result, expected)


# ---------------------------------------------------------------------------
# Tests for transform_ode: explicit transformations
# ---------------------------------------------------------------------------


def test_transform_ode_first_order_euler_equation():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    eq = sp.Eq(x * sp.diff(Y(x), x) + Y(x) - sp.sin(sp.log(x)), 0)

    transformed = transform_ode(
        eq=eq,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = sp.diff(U(t), t) + U(t) - sp.sin(t)
    assert_eq_lhs_equal(transformed, expected)


def test_transform_ode_zero_order_equation_transforms_dependent_function():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    eq = sp.Eq(Y(x) + sp.log(x), 0)

    transformed = transform_ode(
        eq=eq,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = U(t) + t
    assert_eq_lhs_equal(transformed, expected)


def test_transform_ode_second_derivative_under_log_transform():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    # x^2 Y'' + x Y' is (x D_x)^2 Y and should become U''(t).
    eq = sp.Eq(x**2 * sp.diff(Y(x), x, 2) + x * sp.diff(Y(x), x), 0)

    transformed = transform_ode(
        eq=eq,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = sp.diff(U(t), t, 2)
    assert_eq_lhs_equal(transformed, expected)


def test_transform_ode_third_order_euler_operator():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    eq = sp.Eq(
        x**3 * sp.diff(Y(x), x, 3)
        + 3*x**2 * sp.diff(Y(x), x, 2)
        + x * sp.diff(Y(x), x)
        - sp.sin(sp.log(x)),
        0,
    )

    transformed = transform_ode(
        eq=eq,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = sp.diff(U(t), t, 3) - sp.sin(t)
    assert_eq_lhs_equal(transformed, expected)


def test_transform_ode_fourth_order_euler_operator():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    eq = sp.Eq(
        x**4 * sp.diff(Y(x), x, 4)
        + 6*x**3 * sp.diff(Y(x), x, 3)
        + 7*x**2 * sp.diff(Y(x), x, 2)
        + x * sp.diff(Y(x), x)
        - sp.sin(sp.log(x)),
        0,
    )

    transformed = transform_ode(
        eq=eq,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = sp.diff(U(t), t, 4) - sp.sin(t)
    assert_eq_lhs_equal(transformed, expected)


def test_transform_ode_expr_input_is_interpreted_as_zero_equation():
    x, t = sp.symbols("x t", real=True)
    Y = sp.Function("Y")
    U = sp.Function("U")

    expr = x * sp.diff(Y(x), x) + Y(x)

    transformed = transform_ode(
        eq=expr,
        old_var=x,
        new_sym=t,
        f_theta=sp.log(x),
        g_x=sp.exp(t),
        old_func=Y,
        new_func=U,
    )

    expected = sp.diff(U(t), t) + U(t)
    assert_eq_lhs_equal(transformed, expected)


# ---------------------------------------------------------------------------
# Tests for reparametrize_by_state: dynamic reparametrization
# ---------------------------------------------------------------------------


def test_reparametrize_by_state_harmonic_oscillator():
    t, x = sp.symbols("t x", real=True)
    omega = sp.Symbol("omega", positive=True)

    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    eq = sp.Eq(sp.diff(v(t), t), -omega**2 * X(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = V(x) * sp.diff(V(x), x) + omega**2 * x
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_autonomous_force_depends_on_x_and_v():
    t, x = sp.symbols("t x", real=True)
    gamma, k = sp.symbols("gamma k")

    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    # dv/dt = -k X - gamma v
    eq = sp.Eq(sp.diff(v(t), t), -k * X(t) - gamma * v(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = V(x) * sp.diff(V(x), x) + k*x + gamma*V(x)
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_expression_input():
    t, x = sp.symbols("t x", real=True)
    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    # Expression form: dv/dt + X**2 = 0
    expr = sp.diff(v(t), t) + X(t)**2

    transformed = reparametrize_by_state(
        eq=expr,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = V(x) * sp.diff(V(x), x) + x**2
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_state_acceleration_becomes_velocity_acceleration():
    t, x = sp.symbols("t x", real=True)
    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    # X''(t) = -X(t) should become V V' + x = 0.
    eq = sp.Eq(sp.diff(X(t), t, 2), -X(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = V(x) * sp.diff(V(x), x) + x
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_second_time_derivative_of_velocity():
    t, x = sp.symbols("t x", real=True)
    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    # v''(t) should become (V D_x)^2 V.
    eq = sp.Eq(sp.diff(v(t), t, 2), X(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = (
        V(x) * sp.diff(V(x), x)**2
        + V(x)**2 * sp.diff(V(x), x, 2)
        - x
    )
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_first_state_derivative_is_velocity():
    t, x = sp.symbols("t x", real=True)
    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    eq = sp.Eq(sp.diff(X(t), t), v(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    # The kinematic identity dX/dt = v becomes V(x) - V(x) = 0.
    expected = 0
    assert_eq_lhs_equal(transformed, expected)


def test_reparametrize_by_state_leaves_explicit_time_visible_for_nonautonomous_equation():
    t, x = sp.symbols("t x", real=True)
    X = sp.Function("X")
    v = sp.Function("v")
    V = sp.Function("V")

    # Non-autonomous example. The function should not hide the remaining t.
    eq = sp.Eq(sp.diff(v(t), t), t * X(t))

    transformed = reparametrize_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_velocity_func=V,
    )

    expected = V(x) * sp.diff(V(x), x) - t*x
    assert_eq_lhs_equal(transformed, expected)
    assert t in transformed.free_symbols


# ---------------------------------------------------------------------------
# Tests for reparametrize_nonautonomous_by_state: dynamic reparametrization
# with explicit time dependence
# ---------------------------------------------------------------------------


def test_reparametrize_nonautonomous_by_state_time_dependent_force():
    t, x = sp.symbols("t x", real=True)

    X = sp.Function("X")
    v = sp.Function("v")
    T = sp.Function("T")
    V = sp.Function("V")

    from trafo_dgl import reparametrize_nonautonomous_by_state

    # dv/dt = t*X(t)
    eq = sp.Eq(sp.diff(v(t), t), t * X(t))

    main_eq, aux_eq = reparametrize_nonautonomous_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_time_func=T,
        new_velocity_func=V,
    )

    expected_main = V(x) * sp.diff(V(x), x) - x*T(x)
    expected_aux = V(x) * sp.diff(T(x), x) - 1

    assert_eq_lhs_equal(main_eq, expected_main)
    assert_eq_lhs_equal(aux_eq, expected_aux)


def test_reparametrize_nonautonomous_by_state_time_and_velocity_dependent_force():
    t, x = sp.symbols("t x", real=True)
    alpha, beta = sp.symbols("alpha beta")

    X = sp.Function("X")
    v = sp.Function("v")
    T = sp.Function("T")
    V = sp.Function("V")

    from trafo_dgl import reparametrize_nonautonomous_by_state

    # dv/dt = alpha*t*v(t) - beta*X(t)
    eq = sp.Eq(sp.diff(v(t), t), alpha*t*v(t) - beta*X(t))

    main_eq, aux_eq = reparametrize_nonautonomous_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_time_func=T,
        new_velocity_func=V,
    )

    expected_main = V(x) * sp.diff(V(x), x) - alpha*T(x)*V(x) + beta*x
    expected_aux = V(x) * sp.diff(T(x), x) - 1

    assert_eq_lhs_equal(main_eq, expected_main)
    assert_eq_lhs_equal(aux_eq, expected_aux)


def test_reparametrize_nonautonomous_by_state_can_return_main_equation_only():
    t, x = sp.symbols("t x", real=True)

    X = sp.Function("X")
    v = sp.Function("v")
    T = sp.Function("T")
    V = sp.Function("V")

    from trafo_dgl import reparametrize_nonautonomous_by_state

    eq = sp.Eq(sp.diff(v(t), t), sp.sin(t))

    main_eq = reparametrize_nonautonomous_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_time_func=T,
        new_velocity_func=V,
        return_auxiliary=False,
    )

    expected_main = V(x) * sp.diff(V(x), x) - sp.sin(T(x))

    assert_eq_lhs_equal(main_eq, expected_main)


def test_reparametrize_nonautonomous_by_state_transforms_state_acceleration():
    t, x = sp.symbols("t x", real=True)

    X = sp.Function("X")
    v = sp.Function("v")
    T = sp.Function("T")
    V = sp.Function("V")

    from trafo_dgl import reparametrize_nonautonomous_by_state

    # X''(t) = t + X(t)
    eq = sp.Eq(sp.diff(X(t), t, 2), t + X(t))

    main_eq, aux_eq = reparametrize_nonautonomous_by_state(
        eq=eq,
        old_var=t,
        state_func=X,
        velocity_func=v,
        new_sym=x,
        new_time_func=T,
        new_velocity_func=V,
    )

    expected_main = V(x) * sp.diff(V(x), x) - T(x) - x
    expected_aux = V(x) * sp.diff(T(x), x) - 1

    assert_eq_lhs_equal(main_eq, expected_main)
    assert_eq_lhs_equal(aux_eq, expected_aux)


# ---------------------------------------------------------------------------
# Validation and API error handling
# ---------------------------------------------------------------------------


def test_apply_operator_forward_rejects_invalid_variable():
    x = sp.Symbol("x")

    try:
        apply_operator_forward(x**2, x, "x", 1)
    except TypeError as exc:
        assert "x must be a SymPy Symbol" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_apply_operator_forward_rejects_non_integer_order():
    x = sp.Symbol("x")

    try:
        apply_operator_forward(x**2, x, x, 1.5)
    except TypeError as exc:
        assert "n must be a non-negative integer" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_apply_operator_forward_rejects_bool_order():
    x = sp.Symbol("x")

    try:
        apply_operator_forward(x**2, x, x, True)
    except TypeError as exc:
        assert "not bool" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_apply_operator_forward_rejects_negative_order():
    x = sp.Symbol("x")

    try:
        apply_operator_forward(x**2, x, x, -1)
    except ValueError as exc:
        assert "n must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_transform_ode_rejects_invalid_old_variable():
    x, t = sp.symbols("x t")
    Y = sp.Function("Y")
    U = sp.Function("U")
    eq = sp.Eq(Y(x), 0)

    try:
        transform_ode(eq, "x", t, sp.log(x), sp.exp(t), Y, U)
    except TypeError as exc:
        assert "old_var must be a SymPy Symbol" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_transform_ode_rejects_invalid_function_class():
    x, t = sp.symbols("x t")
    Y = sp.Function("Y")
    eq = sp.Eq(Y(x), 0)

    try:
        transform_ode(eq, x, t, sp.log(x), sp.exp(t), Y, "U")
    except TypeError as exc:
        assert "new_func must be a SymPy function class" in str(exc)
    else:
        raise AssertionError("Expected TypeError")


def test_reparametrize_nonautonomous_by_state_rejects_invalid_return_auxiliary():
    t, x = sp.symbols("t x")
    X = sp.Function("X")
    v = sp.Function("v")
    T = sp.Function("T")
    V = sp.Function("V")
    eq = sp.Eq(sp.diff(v(t), t), t * X(t))

    from trafo_dgl import reparametrize_nonautonomous_by_state

    try:
        reparametrize_nonautonomous_by_state(
            eq=eq,
            old_var=t,
            state_func=X,
            velocity_func=v,
            new_sym=x,
            new_time_func=T,
            new_velocity_func=V,
            return_auxiliary="yes",
        )
    except TypeError as exc:
        assert "return_auxiliary must be bool" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
