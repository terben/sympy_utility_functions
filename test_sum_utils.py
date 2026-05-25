import pytest
import sympy as sp

from sum_utils import (
    swap_integral_sum,
    push_factor_into_sum,
    apply_to_summand,
    push_prefactors_into_sums,
    split_operator_over_addition,
    collect_operator_terms,
    rewrite_sum_lower_limit,
)


x, i, j, n, m, a = sp.symbols("x i j n m a")


def test_swap_integral_sum_basic():
    expr = sp.Integral(sp.Sum(i * x, (i, 1, n)), x)
    expected = sp.Sum(sp.Integral(i * x, x), (i, 1, n))
    assert swap_integral_sum(expr) == expected


def test_swap_integral_sum_leaves_other_integrals_unchanged():
    expr = sp.Integral(x**2, x)
    assert swap_integral_sum(expr) == expr


def test_swap_integral_sum_inside_larger_expression():
    expr = 1 + sp.Integral(sp.Sum(i * x, (i, 1, n)), x)
    expected = 1 + sp.Sum(sp.Integral(i * x, x), (i, 1, n))
    assert swap_integral_sum(expr) == expected


def test_swap_integral_sum_converts_plain_input():
    assert swap_integral_sum(1) == sp.Integer(1)


def test_swap_integral_sum_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        swap_integral_sum(object())


def test_push_factor_into_sum_basic():
    expr = sp.Sum(i, (i, 1, n))
    expected = sp.Sum(a * i, (i, 1, n)) / a
    assert push_factor_into_sum(expr, a, i) == expected


def test_push_factor_into_sum_changes_one_matching_sum():
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(i**2, (i, 1, n))
    result = push_factor_into_sum(expr, a, i)
    expected1 = sp.Sum(a * i, (i, 1, n)) / a + sp.Sum(i**2, (i, 1, n))
    expected2 = sp.Sum(i, (i, 1, n)) + sp.Sum(a * i**2, (i, 1, n)) / a
    assert result == expected1 or result == expected2


def test_push_factor_into_sum_leaves_expression_unchanged_if_no_match():
    expr = sp.Sum(j, (j, 1, m))
    assert push_factor_into_sum(expr, a, i) == expr


def test_push_factor_into_sum_leaves_expression_unchanged_if_factor_depends_on_index():
    expr = sp.Sum(i + 1, (i, 1, n))
    assert push_factor_into_sum(expr, a * i, i) == expr


def test_push_factor_into_sum_accepts_sympifiable_factor():
    expr = sp.Sum(i, (i, 1, n))
    expected = sp.Sum(2 * i, (i, 1, n)) / 2
    assert push_factor_into_sum(expr, 2, i) == expected


def test_push_factor_into_sum_raises_type_error_for_invalid_index():
    expr = sp.Sum(i, (i, 1, n))
    with pytest.raises(TypeError):
        push_factor_into_sum(expr, a, "i")


def test_push_factor_into_sum_raises_type_error_for_invalid_factor():
    expr = sp.Sum(i, (i, 1, n))
    with pytest.raises(TypeError):
        push_factor_into_sum(expr, object(), i)


def test_push_factor_into_sum_multi_limit_sum():
    expr = sp.Sum(i + j, (i, 1, n), (j, 1, m))
    expected = sp.Sum(a * (i + j), (i, 1, n), (j, 1, m)) / a
    assert push_factor_into_sum(expr, a, j) == expected


def test_apply_to_summand_with_expand():
    expr = sp.Sum((x + 1) * (x + 2), (i, 1, n))
    expected = sp.Sum(sp.expand((x + 1) * (x + 2)), (i, 1, n))
    assert apply_to_summand(expr, sp.expand) == expected


def test_apply_to_summand_with_custom_function():
    expr = sp.Sum(i + 1, (i, 1, n))
    expected = sp.Sum((i + 1)**2, (i, 1, n))
    assert apply_to_summand(expr, lambda s: s**2) == expected


def test_apply_to_summand_raises_type_error_for_non_sum():
    with pytest.raises(TypeError):
        apply_to_summand(i + 1, sp.expand)


def test_apply_to_summand_raises_type_error_for_non_callable():
    expr = sp.Sum(i + 1, (i, 1, n))
    with pytest.raises(TypeError):
        apply_to_summand(expr, 1)


def test_push_prefactors_into_sums_basic():
    expr = a * sp.Sum(i + 1, (i, 1, n))
    expected = sp.Sum(a * (i + 1), (i, 1, n))
    assert push_prefactors_into_sums(expr) == expected


def test_push_prefactors_into_sums_inside_larger_expression():
    expr = 1 + a * sp.Sum(i, (i, 1, n))
    expected = 1 + sp.Sum(a * i, (i, 1, n))
    assert push_prefactors_into_sums(expr) == expected


def test_push_prefactors_into_sums_leaves_expression_unchanged_if_prefactor_depends_on_index():
    expr = i * sp.Sum(i + 1, (i, 1, n))
    assert push_prefactors_into_sums(expr) == expr


def test_push_prefactors_into_sums_leaves_plain_sum_unchanged():
    expr = sp.Sum(i + 1, (i, 1, n))
    assert push_prefactors_into_sums(expr) == expr


def test_push_prefactors_into_sums_leaves_products_with_two_sums_unchanged():
    expr = a * sp.Sum(i, (i, 1, n)) * sp.Sum(j, (j, 1, m))
    assert push_prefactors_into_sums(expr) == expr


def test_push_prefactors_into_sums_accepts_plain_input():
    assert push_prefactors_into_sums(1) == sp.Integer(1)


def test_push_prefactors_into_sums_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        push_prefactors_into_sums(object())


def test_split_operator_over_addition_sum_basic():
    expr = sp.Sum(i + x, (i, 1, n))
    expected = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    assert split_operator_over_addition(expr, sp.Sum) == expected


def test_split_operator_over_addition_integral_basic():
    expr = sp.Integral(x + 1, x)
    expected = sp.Integral(x, x) + sp.Integral(1, x)
    assert split_operator_over_addition(expr, sp.Integral) == expected


def test_split_operator_over_addition_inside_larger_expression():
    expr = 1 + sp.Sum(i + x, (i, 1, n))
    expected = 1 + sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    assert split_operator_over_addition(expr, sp.Sum) == expected


def test_split_operator_over_addition_leaves_non_additive_interior_unchanged():
    expr = sp.Sum(i * x, (i, 1, n))
    assert split_operator_over_addition(expr, sp.Sum) == expr


def test_split_operator_over_addition_multi_limit_sum():
    expr = sp.Sum(i + j + x, (i, 1, n), (j, 1, m))
    expected = (
        sp.Sum(i, (i, 1, n), (j, 1, m))
        + sp.Sum(j, (i, 1, n), (j, 1, m))
        + sp.Sum(x, (i, 1, n), (j, 1, m))
    )
    assert split_operator_over_addition(expr, sp.Sum) == expected


def test_split_operator_over_addition_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        split_operator_over_addition(object(), sp.Sum)


def test_split_operator_over_addition_raises_type_error_for_invalid_operator():
    expr = sp.Sum(i + x, (i, 1, n))
    with pytest.raises(TypeError):
        split_operator_over_addition(expr, 1)


def test_collect_operator_terms_sum_basic():
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = sp.Sum(i + x, (i, 1, n))
    assert collect_operator_terms(expr, sp.Sum, ((i, 1, n),)) == expected


def test_collect_operator_terms_integral_basic():
    expr = sp.Integral(x, x) + sp.Integral(x**2, x)
    result = collect_operator_terms(expr, sp.Integral, ((x,),))
    assert isinstance(result, sp.Integral)
    assert result.limits == ((x,),)
    assert sp.simplify(result.function - (x**2 + x)) == 0


def test_collect_operator_terms_inside_larger_expression():
    expr = 1 + sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = 1 + sp.Sum(i + x, (i, 1, n))
    assert collect_operator_terms(expr, sp.Sum, ((i, 1, n),)) == expected


def test_collect_operator_terms_leaves_expression_unchanged_if_no_match():
    expr = sp.Sum(i, (i, 1, n))
    assert collect_operator_terms(expr, sp.Sum, ((j, 1, m),)) == expr


def test_collect_operator_terms_only_collects_matching_limits():
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(j, (j, 1, m))
    assert collect_operator_terms(expr, sp.Sum, ((i, 1, m),)) == expr


def test_collect_operator_terms_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        collect_operator_terms(object(), sp.Sum, ((i, 1, n),))


def test_collect_operator_terms_raises_type_error_for_invalid_operator():
    expr = sp.Sum(i, (i, 1, n))
    with pytest.raises(TypeError):
        collect_operator_terms(expr, 1, ((i, 1, n),))


def test_rewrite_sum_lower_limit_moves_lower_limit_up():
    expr = sp.Sum(x**i, (i, 0, 10))
    expected = 1 + x + sp.Sum(x**i, (i, 2, 10))

    result = rewrite_sum_lower_limit(expr, 2, index=i)

    assert result == expected


def test_rewrite_sum_lower_limit_moves_lower_limit_down():
    expr = sp.Sum(x**i, (i, 2, 10))
    expected = sp.Sum(x**i, (i, 0, 10)) - 1 - x

    result = rewrite_sum_lower_limit(expr, 0, index=i)

    assert result == expected


def test_rewrite_sum_lower_limit_inside_larger_expression():
    expr = 7 + sp.Sum(x**i, (i, 0, 10))
    expected = 8 + x + sp.Sum(x**i, (i, 2, 10))

    result = rewrite_sum_lower_limit(expr, 2, index=i)

    assert result == expected


def test_rewrite_sum_lower_limit_only_rewrites_matching_index():
    expr = sp.Sum(x**i, (i, 0, 10)) + sp.Sum(x**j, (j, 0, 10))
    expected = 1 + x + sp.Sum(x**i, (i, 2, 10)) + sp.Sum(x**j, (j, 0, 10))

    result = rewrite_sum_lower_limit(expr, 2, index=i)

    assert result == expected


def test_rewrite_sum_lower_limit_rewrites_all_single_index_sums_without_index_argument():
    expr = sp.Sum(x**i, (i, 0, 10)) + sp.Sum(x**j, (j, 0, 10))
    expected = (
        1 + x + sp.Sum(x**i, (i, 2, 10))
        + 1 + x + sp.Sum(x**j, (j, 2, 10))
    )

    result = rewrite_sum_lower_limit(expr, 2)

    assert result == expected


def test_rewrite_sum_lower_limit_no_change_if_lower_limit_already_matches():
    expr = sp.Sum(x**i, (i, 2, 10))

    result = rewrite_sum_lower_limit(expr, 2, index=i)

    assert result == expr


def test_rewrite_sum_lower_limit_works_with_symbolic_upper_bound():
    expr = sp.Sum(x**i, (i, 0, n))
    expected = 1 + x + sp.Sum(x**i, (i, 2, n))

    result = rewrite_sum_lower_limit(expr, 2, index=i)

    assert result == expected


def test_rewrite_sum_lower_limit_raises_type_error_for_invalid_input():
    with pytest.raises(TypeError):
        rewrite_sum_lower_limit(object(), 0, index=i)


def test_rewrite_sum_lower_limit_raises_type_error_for_invalid_index():
    expr = sp.Sum(x**i, (i, 0, 10))

    with pytest.raises(TypeError):
        rewrite_sum_lower_limit(expr, 2, index="i")


def test_rewrite_sum_lower_limit_raises_value_error_for_non_integer_shift():
    expr = sp.Sum(x**i, (i, 0, 10))

    with pytest.raises(ValueError):
        rewrite_sum_lower_limit(expr, sp.Rational(1, 2), index=i)


def test_rewrite_sum_lower_limit_raises_value_error_for_symbolic_shift():
    expr = sp.Sum(x**i, (i, 0, 10))

    with pytest.raises(ValueError):
        rewrite_sum_lower_limit(expr, n, index=i)


def test_rewrite_sum_lower_limit_raises_value_error_for_matching_multi_index_sum():
    expr = sp.Sum(x**i * j, (i, 0, 10), (j, 0, 5))

    with pytest.raises(ValueError):
        rewrite_sum_lower_limit(expr, 2, index=i)


def test_collect_operator_terms_accepts_list_limits():
    expr = sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = sp.Sum(i + x, (i, 1, n))

    result = collect_operator_terms(expr, sp.Sum, [(i, 1, n)])

    assert result == expected


def test_collect_operator_terms_preserves_non_operator_terms():
    expr = 7 + sp.Sum(i, (i, 1, n)) + sp.Sum(x, (i, 1, n))
    expected = 7 + sp.Sum(i + x, (i, 1, n))

    result = collect_operator_terms(expr, sp.Sum, ((i, 1, n),))

    assert result == expected


def test_collect_operator_terms_rejects_non_iterable_limits():
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(TypeError):
        collect_operator_terms(expr, sp.Sum, 1)


def test_collect_operator_terms_rejects_string_limits():
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(TypeError):
        collect_operator_terms(expr, sp.Sum, "limits")


def test_collect_operator_terms_rejects_empty_limits():
    expr = sp.Sum(i, (i, 1, n))

    with pytest.raises(ValueError):
        collect_operator_terms(expr, sp.Sum, ())
