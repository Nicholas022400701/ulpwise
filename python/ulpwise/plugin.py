"""pytest plugin.

* A test that takes an ``edge_f64``, ``edge_f32``, ``edge_f16`` or ``edge_bf16`` argument is
  parametrized over the named edge values of that dtype (ids are the names, so failures read
  ``test_x[square_overflows]``).
* The ``assert_max_ulp`` fixture hands out :func:`ulpwise.assert_max_ulp`.
"""

import pytest

import ulpwise


def pytest_generate_tests(metafunc):
    for arg, dtype in (("edge_f32", "f32"), ("edge_f64", "f64"), ("edge_f16", "f16"), ("edge_bf16", "bf16")):
        if arg in metafunc.fixturenames:
            pairs = ulpwise.special(dtype)
            metafunc.parametrize(arg, [v for _, v in pairs], ids=[n for n, _ in pairs])


@pytest.fixture
def assert_max_ulp():
    return ulpwise.assert_max_ulp
