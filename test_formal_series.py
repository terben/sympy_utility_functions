import pytest
import sympy as sp

from sympy_utils import formal_series


def test_symbol_fast_path_with_order_term():
    x = sp.symbols("x")

    result = formal_series(sp.sin(x), x, n=6)

    expected = x - x**3 / 6 + x**5 / 120 + sp.O(x**6)
    assert result == expected



def test_composite_expression_x_plus_y():
    x, y = sp.symbols("x y")

    result = formal_series(sp.sin(x + y), x + y, n=6)

    expected = (
        x + y
        - (x + y)**3 / 6
        + (x + y)**5 / 120
    )

    assert result == expected


def test_undefined_function_argument():
    t = sp.symbols("t")
    f = sp.Function("f")

    result = formal_series(sp.sin(f(t)), f(t), n=6)

    expected = (
        f(t)
        - f(t)**3 / 6
        + f(t)**5 / 120
    )

    assert result == expected


def test_expansion_point_nonzero():
    u = sp.symbols("u")

    result = formal_series(sp.exp(u), u, x0=1, n=3)

    expected = (
        sp.E
        + sp.E * (u - 1)
        + sp.E * (u - 1)**2 / 2
        + sp.O((u - 1)**3, (u, 1))
    )

    assert result == expected


def test_small_expr_not_found_raises():
    x, y = sp.symbols("x y")

    with pytest.raises(ValueError, match="small_expr was not found"):
        formal_series(sp.sin(x), x + y, n=6)


def test_invalid_order_raises():
    x = sp.symbols("x")

    with pytest.raises(ValueError, match="n must be a positive integer"):
        formal_series(sp.sin(x), x, n=0)


def test_invalid_dummy_raises():
    x, y = sp.symbols("x y")

    with pytest.raises(TypeError, match="dummy must be a SymPy Symbol"):
        formal_series(sp.sin(x + y), x + y, dummy=x + y)


def test_expression_equal_to_small_expr():
    x, y = sp.symbols("x y")

    result = formal_series(x + y, x + y, n=4)

    assert result == x + y
