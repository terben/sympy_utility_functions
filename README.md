# SymPy Utility Functions

Small standalone utility files for symbolic computations with SymPy.

The files are intended for:

- university teaching,
- Jupyter notebooks,
- exploratory symbolic calculations,
- supplementary material for videos and exercises.

The utilities are deliberately distributed as individual files instead of a
full Python package. This keeps them easy to inspect, modify, and reuse,
especially for students who are still learning Python and SymPy.

---

## Getting started

Each file can be used in two simple ways.

### 1. Run a file directly

Most files contain a small demo block at the end.

Run a file with:

```bash
python sympy_utils.py
python sum_utils.py
python trafo_dgl.py
python sympy_eq.py
```

This prints short examples showing the original expression together with the transformed result.

---

### 2. Import functions into your own script or notebook

Import selected functions:

```python
from sympy_utils import formal_series, real_apart
from sum_utils import split_sum_of_addition
from sympy_eq import Eq
```

or import a complete file as a namespace:

```python
import sympy_utils as su
import sum_utils as sums
```

The utilities are intended for direct interactive use in:

- Python scripts,
- Jupyter notebooks,
- worksheets,
- teaching material.

---

## Requirements

The files currently depend only on:

- Python 3
- SymPy

Typical setup:

```bash
pip install sympy
```

---

## File overview

### `sympy_utils.py`

General-purpose symbolic helper functions.

Main topics:

- formal series expansions,
- trigonometric transformations,
- integration by parts,
- partial fraction decompositions,
- symbolic manipulations.

Example use cases:

- symbolic calculus,
- quick notebook calculations,
- teaching examples,
- algebraic experimentation.

---

### `trafo_dgl.py`

Utilities for transforming ordinary differential equations.

Main topics:

- operator identities,
- repeated differential operators,
- explicit coordinate transformations,
- dynamical reparametrizations,
- autonomous and non-autonomous systems.

The file focuses on mathematically transparent implementations of transformations such as

\[
\frac{d}{d\theta}
=
A(x)\frac{d}{dx}.
\]

Particular care is taken to distinguish:
- chain-rule transformations,
- operator pushforwards,
- dynamical reparametrizations.

---

### `sum_utils.py`

Helper functions for symbolic manipulations of unevaluated SymPy sums.

Main topics:

- interchanging sums and integrals,
- pushing factors into sums,
- transformations of summands,
- splitting sums over additions.

The functions are intentionally conservative:
if a transformation is not applicable, the original expression is usually returned unchanged.

---

### `sympy_eq.py`

Convenience equation class for elementary symbolic transformations.

The custom `Eq` class extends `sympy.Eq` with operations applied to both sides:

```python
Eq(2*x + 1, 5) - 1
```

produces

```python
Eq(2*x, 4)
```

Main features:

- arithmetic on both sides,
- applying functions to one or both sides,
- direct solving for variables,
- interactive symbolic manipulations.

The class is primarily intended for teaching and interactive work.

---

## Design goals

The utilities are designed to be:

- mathematically transparent,
- easy to read,
- easy to modify,
- directly usable in notebooks,
- suitable for teaching.

The implementations intentionally avoid excessive abstraction in order to remain accessible to students learning SymPy and symbolic computation.

---

## Mathematical note

These utilities are intended for symbolic experimentation and teaching.

Users should still verify:
- assumptions,
- domains,
- convergence conditions,
- differentiability requirements,
- validity of symbolic transformations.

This is especially important for:
- differential equation transformations,
- operator identities,
- symbolic summation manipulations.

---

## Running tests

Tests use `pytest`.

Run all tests with:

```bash
pytest
```

or run a single test file:

```bash
pytest test_sum_utils.py
```

---

## Example

### Symbolic equation manipulation

```python
from sympy import symbols
from sympy_eq import Eq

x = symbols("x")

eq = Eq(2*x + 1, 5)

print(eq - 1)
print(eq / 2)
```

Output:

```python
Eq(2*x, 4)
Eq(x, 2)
```

---

### Formal series expansion

```python
import sympy as sp
from sympy_utils import formal_series

x = sp.Symbol("x")

expr = sp.exp(sp.sin(x))

print(formal_series(expr, sp.sin(x), n=5))
```

---

### Sum transformations

```python
import sympy as sp
from sum_utils import split_sum_of_addition

i, n, x = sp.symbols("i n x")

expr = sp.Sum(i + x, (i, 1, n))

print(split_sum_of_addition(expr))
```

Output:

```python
Sum(i, (i, 1, n)) + Sum(x, (i, 1, n))
```

## Development note

Parts of the documentation, code cleanup, test extensions, and API
consistency improvements were developed with the assistance of ChatGPT
(OpenAI).

The mathematical concepts, algorithms, and overall project design were
developed and curated by the project author.
