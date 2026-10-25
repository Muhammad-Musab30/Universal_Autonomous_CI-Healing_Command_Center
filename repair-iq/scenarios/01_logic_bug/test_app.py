import pytest
from app import apply_discount, calculate_order_total

def test_apply_discount_standard():
    # $100 with 20% discount should be $80.00
    result = apply_discount(100.0, 20.0)
    assert result == 80.0, f"Expected 80.0, got {result}"

def test_apply_discount_zero():
    result = apply_discount(50.0, 0.0)
    assert result == 50.0, f"Expected 50.0, got {result}"

def test_calculate_order_total():
    # $100 with 10% discount is $90.0; 8% tax is $7.20 -> total $97.20
    total = calculate_order_total(100.0, 10.0, tax_rate=0.08)
    assert total == 97.20, f"Expected 97.20, got {total}"
