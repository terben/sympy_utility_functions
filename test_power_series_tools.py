import pytest
import sympy as sp

from power_series_tools import PowerSumNormalizer, diff_power_series


x, k, n, m, alpha = sp.symbols("x k n m alpha")
a = sp.Function("a")
b = sp.Function("b")


def test_normalize_single_shifted_power_sum():
    normalizer = PowerSumNormalizer(x, n)
    expr = sp.Sum(a(k) * x**(k + 2), (k, 0, sp.oo))
    expected = sp.Sum(x**n * a(n - 2), (n, 2, sp.oo))

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_preserves_non_sum_terms():
    normalizer = PowerSumNormalizer(x, n)
    expr = 1 + sp.Sum(a(k) * x**(k + 1), (k, 0, sp.oo))
    expected = 1 + sp.Sum(x**n * a(n - 1), (n, 1, sp.oo))

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_multiple_sums():
    normalizer = PowerSumNormalizer(x, n)
    expr = (
        sp.Sum(a(k) * x**(k + 1), (k, 0, sp.oo))
        + sp.Sum(b(k) * x**(k + 2), (k, 0, sp.oo))
    )
    expected = (
        sp.Sum(x**n * a(n - 1), (n, 1, sp.oo))
        + sp.Sum(x**n * b(n - 2), (n, 2, sp.oo))
    )

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_pushes_prefactor_into_sum():
    normalizer = PowerSumNormalizer(x, n)
    expr = x * sp.Sum(a(k) * x**k, (k, 0, sp.oo))
    expected = sp.Sum(x**n * a(n - 1), (n, 1, sp.oo))

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_splits_sum_over_addition():
    normalizer = PowerSumNormalizer(x, n)
    expr = sp.Sum(a(k) * x**(k + 1) + b(k) * x**(k + 2), (k, 0, sp.oo))
    expected = (
        sp.Sum(x**n * a(n - 1), (n, 1, sp.oo))
        + sp.Sum(x**n * b(n - 2), (n, 2, sp.oo))
    )

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_with_shifted_target_exponent():
    normalizer = PowerSumNormalizer(x, n + alpha, new_index=n)
    expr = sp.Sum(a(k) * x**(k + 2), (k, 0, sp.oo))
    expected = sp.Sum(x**(alpha + n) * a(alpha + n - 2), (n, 2 - alpha, sp.oo))

    result = normalizer.normalize(expr)

    assert result == expected


def test_normalize_expression_without_sums_is_unchanged():
    normalizer = PowerSumNormalizer(x, n)
    expr = x**2 + 1

    result = normalizer.normalize(expr)

    assert result == expr


def test_constructor_infers_new_index_from_target_exponent():
    normalizer = PowerSumNormalizer(x, n + 1)

    assert normalizer.new_index == n


def test_constructor_accepts_explicit_new_index():
    normalizer = PowerSumNormalizer(x, n + 1, new_index=m)

    assert normalizer.new_index == m


def test_constructor_rejects_invalid_x():
    with pytest.raises(TypeError):
        PowerSumNormalizer("x", n)


def test_constructor_rejects_ambiguous_target_exp_without_new_index():
    with pytest.raises(ValueError):
        PowerSumNormalizer(x, n + m)


def test_constructor_rejects_invalid_new_index():
    with pytest.raises(TypeError):
        PowerSumNormalizer(x, n, new_index="n")


def test_normalize_rejects_multi_index_sum():
    normalizer = PowerSumNormalizer(x, n)
    expr = sp.Sum(a(k) * x**k, (k, 0, sp.oo), (m, 0, sp.oo))

    with pytest.raises(ValueError):
        normalizer.normalize(expr)


def test_normalize_rejects_exponent_with_wrong_index_coefficient():
    normalizer = PowerSumNormalizer(x, n)
    expr = sp.Sum(a(k) * x**(2 * k), (k, 0, sp.oo))

    with pytest.raises(ValueError):
        normalizer.normalize(expr)


def test_normalize_rejects_inconsistent_shifts():
    normalizer = PowerSumNormalizer(x, n)
    expr = sp.Sum(a(k) * x**k + b(k) * x**(k + 1), (k, 0, sp.oo))

    with pytest.raises(ValueError):
        normalizer._normalize_single_sum(expr)


def test_diff_power_series_first_derivative_infinite_series():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))
    expected = sp.Sum((k + 1) * coeff(k + 1) * x**k, (k, 0, sp.oo))

    result = diff_power_series(expr, x)

    assert result == expected


def test_diff_power_series_second_derivative_infinite_series():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))
    expected = sp.Sum((k + 1) * (k + 2) * coeff(k + 2) * x**k, (k, 0, sp.oo))

    result = diff_power_series(expr, x, order=2)

    assert result == expected


def test_diff_power_series_finite_upper_bound():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, n))
    expected = sp.Sum((k + 1) * coeff(k + 1) * x**k, (k, 0, n - 1))

    result = diff_power_series(expr, x)

    assert result == expected


def test_diff_power_series_order_zero_returns_original_sum():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))

    result = diff_power_series(expr, x, order=0)

    assert result == expr


def test_diff_power_series_raises_type_error_for_non_sum():
    with pytest.raises(TypeError):
        diff_power_series(x**2, x)


def test_diff_power_series_raises_type_error_for_invalid_variable():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))

    with pytest.raises(TypeError):
        diff_power_series(expr, "x")


def test_diff_power_series_raises_type_error_for_invalid_order_type():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))

    with pytest.raises(TypeError):
        diff_power_series(expr, x, order=1.5)


def test_diff_power_series_raises_value_error_for_negative_order():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, sp.oo))

    with pytest.raises(ValueError):
        diff_power_series(expr, x, order=-1)


def test_diff_power_series_raises_value_error_for_multi_index_sum():
    coeff = sp.Function("c")
    expr = sp.Sum(coeff(k) * x**k, (k, 0, n), (m, 0, sp.oo))

    with pytest.raises(ValueError):
        diff_power_series(expr, x)
