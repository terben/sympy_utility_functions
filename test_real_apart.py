"""Unit tests for real_apart in sympy_utils."""

import unittest

import sympy as sp

from sympy_utils import real_apart


class TestRealApart(unittest.TestCase):
    """Tests for real partial fraction decompositions."""

    def setUp(self):
        self.x = sp.symbols("x")

    def assert_equivalent(self, actual, expected):
        """Assert symbolic equivalence of two rational expressions."""
        self.assertEqual(sp.simplify(actual - expected), 0)

    def assert_real_decomposition(self, expr):
        """Assert that real_apart preserves expr and removes explicit I."""
        result = real_apart(expr, self.x)
        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))
        return result

    def test_irreducible_quadratic_remains_real(self):
        expr = 1 / (self.x**2 + 1)

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))
        self.assertEqual(result, expr)

    def test_cubic_with_one_real_and_two_complex_roots(self):
        expr = 1 / (self.x**3 + 1)
        expected = -(self.x - 2) / (3 * (self.x**2 - self.x + 1)) + 1 / (3 * (self.x + 1))

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expected)
        self.assertFalse(result.has(sp.I))

    def test_mixed_real_and_irreducible_quadratic_factors(self):
        expr = (2 * self.x + 3) / ((self.x - 1) * (self.x**2 + 1))
        expected = -(5 * self.x + 1) / (2 * (self.x**2 + 1)) + 5 / (2 * (self.x - 1))

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expected)
        self.assertFalse(result.has(sp.I))

    def test_expression_with_polynomial_part(self):
        expr = self.x + 1 / (self.x**2 + 1)

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))

    def test_repeated_irreducible_quadratic_factor(self):
        expr = 1 / (self.x**2 + 1) ** 2

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))

    def test_shifted_irreducible_quadratic_factor(self):
        expr = 1 / (self.x**2 + 2 * self.x + 2)

        result = real_apart(expr, self.x)

        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))

    def test_sympifies_string_expression(self):
        result = real_apart("1/(x**2 + 1)", self.x)

        self.assert_equivalent(result, 1 / (self.x**2 + 1))
        self.assertFalse(result.has(sp.I))

    def test_rejects_non_symbol_variable(self):
        with self.assertRaisesRegex(TypeError, "x must be a SymPy Symbol"):
            real_apart(1 / (self.x + 1), self.x + 1)

    def test_rejects_non_rational_expression(self):
        with self.assertRaisesRegex(ValueError, "expr must be rational in x"):
            real_apart(sp.sin(self.x), self.x)

    def test_symbol_without_real_assumption_is_treated_as_real_variable(self):
        z = sp.symbols("z")
        expr = 1 / (z**2 + 1)

        result = real_apart(expr, z)

        self.assert_equivalent(result, expr)
        self.assertFalse(result.has(sp.I))


if __name__ == "__main__":
    unittest.main()
