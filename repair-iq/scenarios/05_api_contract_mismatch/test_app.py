import pytest
from app import get_account_summary

def test_account_summary_contract():
    summary = get_account_summary(101)
    # Expected API contract: {"id": 101, "amount": 50.0}
    assert "id" in summary, "Key 'id' missing from API response"
    assert "amount" in summary, "Key 'amount' missing from API response"
    assert summary["id"] == 101
    assert summary["amount"] == 50.0

def test_account_summary_types():
    summary = get_account_summary(202)
    assert isinstance(summary.get("id"), int)
    assert isinstance(summary.get("amount"), float)
