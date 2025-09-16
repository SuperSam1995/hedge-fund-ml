"""Unit tests for pure helper functions in :mod:`data.prepare`."""

from __future__ import annotations

import pytest

from data.prepare import _normalize_etf_label


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Cboe Momentum Index", "Momentum"),
        ("S&P 500 Total Return", "S&P 500"),
        (" MSCI ACWI Total Return Value Unhedged USD  ", "MSCI ACWI"),
        ("Quality Factor .1", "Quality Factor"),
        ("", ""),
    ],
)
def test_normalize_etf_label(label: str, expected: str) -> None:
    """Ensure ETF labels are cleaned consistently."""

    assert _normalize_etf_label(label) == expected
