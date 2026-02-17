import pytest
import turbo_ml


def test_sum_as_string():
    assert turbo_ml.sum_as_string(1, 1) == "2"
