import pytest
import sympy as sp

from sum_utils import (
    swap_integral_sum,
    push_factor_into_sum,
    apply_to_summand,
    push_prefactors_into_sums,
    split_operator_over_addition,
    collect_operator_terms,
)


x, i, j, n, m, a = sp.symbols("x i j n m a")


def test_swap_integral_sum_basic():
    # Integral over a sum should become a sum of integrals.
    expr = sp.Integral(sp.Sum(i * x, (i, 1, n)), x)
    expected = sp.Sum(sp.Integral(i * x, x), (i, 1, n))

    result = swap_integral_sum(expr)

    assert result == expected


def test_swap_integral_sum_leaves_other_integrals_unchanged():
    # If the integrand is not a sum, nothing should change.
    expr = sp.Integral(x**2, x)

    result = swap_integral_sum(expr)

    assert result == expr


def test_swap_integral_sum_inside_larger_expression():
    # The transformation should also work inside a larger expression.
    expr = 1 + sp.Integral(sp.Sum(i * x, (i, 1, n)), x)
    expected = 1 + sp.Sum(sp.Integral(i * x, x), (i, 1, n))

    result = swap_integral_sum(expr)

    assert result == expected


def test_swap_integral_sum_converts_plain_input():
    # Inputs convertible by sympify should be accepted.
    result = swap_integral_sum(1)

    assert result == sp.Integer(1)


def test_swap_integral_sum_raises_type_error_for_invalid_input():
    # Objects that cannot be sympified should raise TypeError.
    with pytest.raises(TypeError):
        swap_integral_sum(object())


def test_push_factor_into_sum_basic():
    # Push the factor into a matching sum.
    expr = sp.Sum(i, (i, 1, n))
    expected = sp.Sum(a * i, (i, 1, n)) / a

    result = push_factor_into_sum(expr, a, i)

    assert result == expected


def test_push_factor_into_sum_changes_one_matching_sum():
    # If several sums match, exactly one of them should be changed.
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(i**2, (i, 1, n))

    result = push_factor_into_sum(expr, a, i)

    expected1 = sp.Sum(a * i, (i, 1, n)) / a + sp.Sum(i**2, (i, 1, n))
    expected2 = sp.Sum(i, (i, 1, n)) + sp.Sum(a * i**2, (i, 1, n)) / a

    assert result == expected1 or result == expected2


def test_push_factor_into_sum_leaves_expression_unchanged_if_no_match():
    # If no matching index is found, the expression stays unchanged.
    expr = sp.Sum(j, (j, 1, m))

    result = push_factor_into_sum(expr, a, i)

    assert result == expr


def test_push_factor_into_sum_leaves_expression_unchanged_if_factor_depends_on_index():
    # A factor containing the summation index should not be pushed into the sum.
    expr = sp.Sum(i + 1, (i, 1, n))

    result = push_factor_into_sum(expr, a * i, i)

    assert result == expr


def test_push_factor_into_sum_accepts_sympifiable_factor():
    # Factors convertible by sympify should be accepted.
    expr = sp.Sum(i, (i, 1, n))
    expected = sp.Sum(2 * i, (i, 1, n)) / 2

    result = push_factor_into_sum(expr, 2, i)

    assert result == expected


def test_push_factor_into_sum_raises_type_error_for_invalid_index():
    # The summation index must be a SymPy symbol.
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(TypeError):
        push_factor_into_sum(expr, a, "i")


def test_push_factor_into_sum_raises_type_error_for_invalid_factor():
    # The factor must be sympifiable.
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(TypeError):
        push_factor_into_sum(expr, object(), i)


def test_push_factor_into_sum_multi_limit_sum():
    # The index check should also work for sums with several limits.
    expr = sp.Sum(i + j, (i, 1, n), (j, 1, m))
    expected = sp.Sum(a * (i + j), (i, 1, n), (j, 1, m)) / a

    result = push_factor_into_sum(expr, a, j)

    assert result == expected


def test_apply_to_summand_with_expand():
    # Apply a SymPy function to the summand.
    expr = sp.Sum((x + 1) * (x + 2), (i, 1, n))
    expected = sp.Sum(sp.expand((x + 1) * (x + 2)), (i, 1, n))

    result = apply_to_summand(expr, sp.expand)

    assert result == expected


def test_apply_to_summand_with_custom_function():
    # Apply a user-defined function to the summand.
    expr = sp.Sum(i + 1, (i, 1, n))
    expected = sp.Sum((i + 1)**2, (i, 1, n))

    result = apply_to_summand(expr, lambda s: s**2)

    assert result == expected


def test_apply_to_summand_raises_type_error_for_non_sum():
    # The function expects a SymPy Sum as first argument.
    with pytest.raises(TypeError):
        apply_to_summand(i + 1, sp.expand)


def test_apply_to_summand_raises_type_error_for_non_callable():
    # The transformation must be callable.
    expr = sp.Sum(i + 1, (i, 1, n))

    with pytest.raises(TypeError):
        apply_to_summand(expr, 1)


def test_push_prefactors_into_sums_basic():
    # A prefactor should be moved into the summand.
    expr = a * sp.Sum(i + 1, (i, 1, n))
    expected = sp.Sum(a * (i + 1), (i, 1, n))

    result = push_prefactors_into_sums(expr)

    assert result == expected


def test_push_prefactors_into_sums_inside_larger_expression():
    # The transformation should also work inside a larger expression.
    expr = 1 + a * sp.Sum(i, (i, 1, n))
    expected = 1 + sp.Sum(a * i, (i, 1, n))

    result = push_prefactors_into_sums(expr)

    assert result == expected


def test_push_prefactors_into_sums_leaves_expression_unchanged_if_prefactor_depends_on_index():
    # A prefactor containing the summation index should not be pushed into the sum.
    expr = i * sp.Sum(i + 1, (i, 1, n))

    result = push_prefactors_into_sums(expr)

    assert result == expr


def test_push_prefactors_into_sums_leaves_plain_sum_unchanged():
    # A sum without an outside prefactor should stay unchanged.
    expr = sp.Sum(i + 1, (i, 1, n))

    result = push_prefactors_into_sums(expr)

    assert result == expr


def test_push_prefactors_into_sums_leaves_products_with_two_sums_unchanged():
    # Products with more than one sum are deliberately not transformed.
    expr = a * sp.Sum(i, (i, 1, n)) * sp.Sum(j, (j, 1, m))

    result = push_prefactors_into_sums(expr)

    assert result == expr


def test_push_prefactors_into_sums_accepts_plain_input():
    # Inputs convertible by sympify should be accepted.
    result = push_prefactors_into_sums(1)

    assert result == sp.Integer(1)


def test_push_prefactors_into_sums_raises_type_error_for_invalid_input():
    # Objects that cannot be sympified should raise TypeError.
    with pytest.raises(TypeError):
        push_prefactors_into_sums(object())


def test_split_operator_over_addition_sum_basic():
    # A sum over an addition should become an addition of sums.
    expr = sp.Sum(i + x, (i, 1, n))
    expected = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))

    result = split_operator_over_addition(expr, sp.Sum)

    assert result == expected


def test_split_operator_over_addition_integral_basic():
    # An integral over an addition should become an addition of integrals.
    expr = sp.Integral(x + 1, x)
    expected = sp.Integral(x, x) + sp.Integral(1, x)

    result = split_operator_over_addition(expr, sp.Integral)

    assert result == expected


def test_split_operator_over_addition_inside_larger_expression():
    # The transformation should also work inside a larger expression.
    expr = 1 + sp.Sum(i + x, (i, 1, n))
    expected = 1 + sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))

    result = split_operator_over_addition(expr, sp.Sum)

    assert result == expected


def test_split_operator_over_addition_leaves_non_additive_interior_unchanged():
    # If the interior is not an addition, nothing should change.
    expr = sp.Sum(i * x, (i, 1, n))

    result = split_operator_over_addition(expr, sp.Sum)

    assert result == expr


def test_split_operator_over_addition_multi_limit_sum():
    # Splitting should preserve all summation limits.
    expr = sp.Sum(i + j + x, (i, 1, n), (j, 1, m))
    expected = (
        sp.Sum(i, (i, 1, n), (j, 1, m))
        + sp.Sum(j, (i, 1, n), (j, 1, m))
        + sp.Sum(x, (i, 1, n), (j, 1, m))
    )

    result = split_operator_over_addition(expr, sp.Sum)

    assert result == expected


def test_split_operator_over_addition_raises_type_error_for_invalid_input():
    # Objects that cannot be sympified should raise TypeError.
    with pytest.raises(TypeError):
        split_operator_over_addition(object(), sp.Sum)


def test_split_operator_over_addition_raises_type_error_for_invalid_operator():
    # The operator must be callable.
    expr = sp.Sum(i + x, (i, 1, n))

    with pytest.raises(TypeError):
        split_operator_over_addition(expr, 1)


def test_collect_operator_terms_sum_basic():
    # Sums with identical limits should be collected.
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = sp.Sum(i + x, (i, 1, n))

    result = collect_operator_terms(
        expr,
        sp.Sum,
        ((i, 1, n),),
    )

    assert result == expected


def test_collect_operator_terms_integral_basic():
    # Integrals with identical limits should be collected.
    expr = sp.Integral(x, x) + sp.Integral(x**2, x)

    result = collect_operator_terms(
        expr,
        sp.Integral,
        ((x,),),
    )

    assert isinstance(result, sp.Integral)
    assert result.limits == ((x,),)
    assert sp.simplify(result.function - (x**2 + x)) == 0


def test_collect_operator_terms_inside_larger_expression():
    # The collection should also work inside a larger expression.
    expr = 1 + sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = 1 + sp.Sum(i + x, (i, 1, n))

    result = collect_operator_terms(
        expr,
        sp.Sum,
        ((i, 1, n),),
    )

    assert result == expected


def test_collect_operator_terms_leaves_expression_unchanged_if_no_match():
    # Expressions without matching operators should stay unchanged.
    expr = sp.Sum(i, (i, 1, n))

    result = collect_operator_terms(
        expr,
        sp.Sum,
        ((j, 1, m),),
    )

    assert result == expr


def test_collect_operator_terms_only_collects_matching_limits():
    # Operators with different limits should not be collected.
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(j, (j, 1, m))
    expected = expr

    result = collect_operator_terms(
        expr,
        sp.Sum,
        ((i, 1, m),),
    )

    assert result == expected


def test_collect_operator_terms_raises_type_error_for_invalid_input():
    # Objects that cannot be sympified should raise TypeError.
    with pytest.raises(TypeError):
        collect_operator_terms(object(), sp.Sum, ((i, 1, n),))


def test_collect_operator_terms_raises_type_error_for_invalid_operator():
    # The operator must be callable.
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(TypeError):
        collect_operator_terms(expr, 1, ((i, 1, n),))
