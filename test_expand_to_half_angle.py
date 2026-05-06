import sympy as sp

from sympy_utils import expand_to_half_angle


def test_expand_sin_to_half_angle():
    x = sp.symbols("x")

    result = expand_to_half_angle(sp.sin(x), x)

    assert result == 2 * sp.sin(x / 2) * sp.cos(x / 2)


def test_expand_cos_to_half_angle():
    x = sp.symbols("x")

    result = expand_to_half_angle(sp.cos(x), x)

    assert result == sp.cos(x / 2)**2 - sp.sin(x / 2)**2


def test_expand_multiple_occurrences():
    x = sp.symbols("x")

    expr = sp.sin(x) + sp.cos(x)
    result = expand_to_half_angle(expr, x)

    expected = (
        2 * sp.sin(x / 2) * sp.cos(x / 2)
        + sp.cos(x / 2)**2
        - sp.sin(x / 2)**2
    )

    assert result == expected


def test_expression_without_matching_angle_is_unchanged():
    x, y = sp.symbols("x y")

    expr = sp.sin(y)
    result = expand_to_half_angle(expr, x)

    assert result == expr


def test_no_angles_returns_original_expression():
    x = sp.symbols("x")

    expr = sp.sin(x)
    result = expand_to_half_angle(expr)

    assert result == expr


def test_expand_selected_angle_only():
    x, y = sp.symbols("x y")

    expr = sp.sin(x) + sp.sin(y)
    result = expand_to_half_angle(expr, x)

    expected = 2 * sp.sin(x / 2) * sp.cos(x / 2) + sp.sin(y)

    assert result == expected


def test_expand_two_angles():
    x, y = sp.symbols("x y")

    expr = sp.sin(x) + sp.cos(y)
    result = expand_to_half_angle(expr, x, y)

    expected = (
        2 * sp.sin(x / 2) * sp.cos(x / 2)
        + sp.cos(y / 2)**2
        - sp.sin(y / 2)**2
    )

    assert result == expected


def test_result_is_mathematically_equivalent():
    x = sp.symbols("x")

    expr = sp.sin(x) + sp.cos(x)
    result = expand_to_half_angle(expr, x)

    assert sp.trigsimp(result - expr) == 0
