import pytest
from app import PaymentGatewayClient, process_charge

def test_charge_success_with_transient_delay():
    # Client fails 2 times before succeeding on 3rd attempt
    client = PaymentGatewayClient(failure_count=2)
    response = process_charge(client, 150.0)
    assert response["status"] == "success"
    assert response["transaction_id"] == "tx_flaky_88192"
    assert response["amount"] == 150.0

def test_charge_exceeding_max_retries_raises():
    # If failures exceed 3, TimeoutError MUST be explicitly re-raised (per reviewer convention)
    client = PaymentGatewayClient(failure_count=5)
    with pytest.raises(TimeoutError) as exc_info:
        process_charge(client, 200.0)
    assert "Payment gateway did not respond within 100ms" in str(exc_info.value)
