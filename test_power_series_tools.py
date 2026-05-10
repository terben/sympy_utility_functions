import pytest
import sympy as sp

from power_series_tools import PowerSumNormalizer


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
