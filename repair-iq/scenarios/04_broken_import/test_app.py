import pytest
from app import generate_bill

def test_generate_bill_standard():
    bill = generate_bill(100.0)
    assert bill["subtotal"] == 100.0
    assert bill["tax"] == 5.0
    assert bill["total"] == 105.0

def test_generate_bill_zero():
    bill = generate_bill(0.0)
    assert bill["tax"] == 0.0
    assert bill["total"] == 0.0
