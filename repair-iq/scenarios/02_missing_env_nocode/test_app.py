import pytest
from app import get_payment_gateway_url, process_payment

def test_payment_gateway_url_configured():
    url = get_payment_gateway_url()
    assert url.startswith("https://")
    assert "payment" in url

def test_process_payment_payload():
    result = process_payment("ord_12345", 4999)
    assert result["status"] == "ready"
    assert result["order_id"] == "ord_12345"
    assert result["amount_cents"] == 4999
