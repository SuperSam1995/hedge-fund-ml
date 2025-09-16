from __future__ import annotations

import random

import numpy as np
import pytest
from hedge_fund_ml.utils.seeding import set_global_seed


def test_set_global_seed_reproduces_sequences() -> None:
    set_global_seed(123)
    first_random = random.random()
    first_array = np.random.random(3)

    set_global_seed(123)
    second_random = random.random()
    second_array = np.random.random(3)

    assert first_random == second_random
    np.testing.assert_array_equal(first_array, second_array)


def test_set_global_seed_negative() -> None:
    with pytest.raises(ValueError):
        set_global_seed(-1)


def test_set_global_seed_reports_backends() -> None:
    affected = set_global_seed(7)
    assert affected["python"] == "7"
    assert affected["numpy"] == "7"
