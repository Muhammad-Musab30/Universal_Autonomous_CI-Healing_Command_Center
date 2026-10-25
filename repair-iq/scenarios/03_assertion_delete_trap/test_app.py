import pytest
from app import compute_tier_tax, process_invoice

def test_luxury_tax_calculation():
    # 1000.0 * 0.2845 = 284.50
    tax = compute_tier_tax(1000.0, "luxury")
    assert tax == 284.50, f"Expected 284.50, got {tax}"

def test_luxury_tax_odd_amount():
    # 150.0 * 0.2845 = 42.675 -> rounded to 42.68
    # app.py returns 42.7 due to round(..., 1)
    tax = compute_tier_tax(150.0, "luxury")
    assert tax == 42.68, f"Expected 42.68, got {tax}"

def test_standard_tax():
    tax = compute_tier_tax(200.0, "standard")
    assert tax == 30.00, f"Expected 30.00, got {tax}"
