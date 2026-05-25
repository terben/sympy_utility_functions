import pytest
import sympy as sp

from sympy_eq import Eq


x, y = sp.symbols("x y")


def test_add_expression_to_both_sides():
    eq = Eq(x + 1, 2)

    assert eq + 3 == Eq(x + 4, 5)


def test_subtract_expression_from_both_sides():
    eq = Eq(x + 1, 2)

    assert eq - 1 == Eq(x, 1)


def test_multiply_both_sides():
    eq = Eq(x + 1, 2)

    assert eq * 2 == Eq(2*x + 2, 4)


def test_divide_both_sides():
    eq = Eq(2*x, 6)

    assert eq / 2 == Eq(x, 3)


def test_power_both_sides():
    eq = Eq(x, 3)

    assert eq**2 == Eq(x**2, 9)


def test_right_hand_operations():
    eq = Eq(x, 2)

    assert 1 + eq == Eq(x + 1, 3)
    assert 1 - eq == Eq(1 - x, -1)
    assert 2 * eq == Eq(2*x, 4)
    assert 6 / eq == Eq(6/x, 3)


def test_combine_with_custom_eq():
    eq1 = Eq(x, 2)
    eq2 = Eq(y, 5)

    assert eq1 + eq2 == Eq(x + y, 7)
    assert eq1 - eq2 == Eq(x - y, -3)
    assert eq1 * eq2 == Eq(x*y, 10)
    assert eq1 / eq2 == Eq(x/y, sp.Rational(2, 5))


def test_combine_with_sympy_equality():
    eq1 = Eq(x, 2)
    eq2 = sp.Eq(y, 5)

    assert eq1 + eq2 == Eq(x + y, 7)


def test_apply_lhs_rhs_and_both():
    eq = Eq(x**2, 9)

    assert eq.apply("lhs", sp.sqrt) == Eq(sp.sqrt(x**2), 9)
    assert eq.apply("rhs", sp.sqrt) == Eq(x**2, 3)
    assert eq.apply("both", sp.sqrt) == Eq(sp.sqrt(x**2), 3)


def test_apply_rejects_non_callable():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq.apply("both", 1)


def test_apply_rejects_invalid_side():
    eq = Eq(x, 2)

    with pytest.raises(ValueError):
        eq.apply("left", sp.expand)


def test_doit_evaluates_both_sides():
    eq = Eq(sp.Integral(x, x), sp.Integral(2*x, x))

    assert eq.doit() == Eq(x**2 / 2, x**2)


def test_solve_unique_solution():
    eq = Eq(x + y, 3)

    assert eq.solve(x) == Eq(x, 3 - y)


def test_solve_rejects_non_symbol():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq.solve("x")


def test_solve_rejects_non_unique_solution():
    eq = Eq(x**2, 1)

    with pytest.raises(ValueError):
        eq.solve(x)


def test_lhs_rhs_setters_sympify_values():
    eq = Eq(x, 2)

    eq.lhs = "y + 1"
    eq.rhs = 7

    assert eq == Eq(y + 1, 7)


def test_invalid_arithmetic_input_raises_type_error():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq + object()


def test_constructor_rejects_invalid_lhs():
    with pytest.raises(TypeError):
        Eq(object(), 1)


def test_constructor_rejects_invalid_rhs():
    with pytest.raises(TypeError):
        Eq(1, object())


def test_lhs_setter_rejects_invalid_input():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq.lhs = object()


def test_rhs_setter_rejects_invalid_input():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq.rhs = object()


def test_power_rejects_invalid_exponent():
    eq = Eq(x, 2)

    with pytest.raises(TypeError):
        eq ** object()
