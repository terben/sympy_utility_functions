"""Unit tests for integrate_by_parts in sympy_utils."""

import unittest

import sympy as sp

from sympy_utils import integrate_by_parts


class TestIntegrateByParts(unittest.TestCase):
    """Tests for one-dimensional integration by parts."""

    def setUp(self):
        self.x = sp.symbols("x")

    def assert_sympy_equal(self, actual, expected):
        """Assert symbolic equality after simplification."""
        self.assertEqual(sp.simplify(actual - expected), 0)

    def test_indefinite_integral(self):
        x = self.x

        result = integrate_by_parts(
            sp.Integral(x * sp.exp(x), x),
            x,
            sp.exp(x),
        )

        expected = x * sp.exp(x) - sp.Integral(sp.exp(x), x)
        self.assertEqual(result, expected)

    def test_definite_integral(self):
        x = self.x

        result = integrate_by_parts(
            sp.Integral(x * sp.exp(x), (x, 0, 1)),
            x,
            sp.exp(x),
        )

        expected = sp.E - sp.Integral(sp.exp(x), (x, 0, 1))
        self.assertEqual(result, expected)

    def test_definite_integral_uses_limits_for_boundary_term(self):
        x = self.x

        result = integrate_by_parts(
            sp.Integral(x * sp.exp(-x), (x, 0, sp.oo)),
            x,
            sp.exp(-x),
        )

        expected = sp.Integral(sp.exp(-x), (x, 0, sp.oo))
        self.assertEqual(result, expected)

    def test_leading_minus_sign_is_extracted(self):
        x = self.x

        result = integrate_by_parts(
            sp.Integral(sp.exp(-x), x),
            sp.Integer(1),
            sp.exp(-x),
        )

        expected = -sp.exp(-x)
        self.assertEqual(result.doit(), expected)

    def test_symbolically_equal_integrand_is_accepted(self):
        x = self.x

        result = integrate_by_parts(
            sp.Integral(2 * x, x),
            x,
            sp.Integer(2),
        )

        expected = 2 * x**2 - sp.Integral(2 * x, x)
        self.assertEqual(result, expected)

    def test_non_integral_input_raises_type_error(self):
        x = self.x

        with self.assertRaisesRegex(TypeError, "integral must be a SymPy Integral"):
            integrate_by_parts(x**2, x, x)

    def test_multiple_integral_raises_value_error(self):
        x, y = sp.symbols("x y")

        with self.assertRaisesRegex(ValueError, "only one-dimensional integrals"):
            integrate_by_parts(sp.Integral(x * y, x, y), x, y)

    def test_invalid_integrand_factorization_raises_value_error(self):
        x = self.x

        with self.assertRaisesRegex(ValueError, "integrand must be equal to u\\*vp"):
            integrate_by_parts(sp.Integral(x**2, x), x, sp.exp(x))

    def test_non_symbol_integration_variable_raises_type_error(self):
        x = self.x
        f = sp.Function("f")

        with self.assertRaisesRegex(TypeError, "integration variable must be a SymPy Symbol"):
            integrate_by_parts(sp.Integral(f(x), f(x)), f(x), sp.Integer(1))


if __name__ == "__main__":
    unittest.main()
