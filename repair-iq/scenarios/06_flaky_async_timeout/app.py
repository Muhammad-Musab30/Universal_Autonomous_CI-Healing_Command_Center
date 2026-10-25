"""Payment Processing Service with Transient Latency."""

class PaymentGatewayClient:
    def __init__(self, failure_count: int = 2):
        self.attempts = 0
        self.failure_count = failure_count

    def send_payment(self, payload: dict) -> dict:
        self.attempts += 1
        if self.attempts <= self.failure_count:
            raise TimeoutError("TimeoutError: Payment gateway did not respond within 100ms")
        return {
            "status": "success",
            "transaction_id": "tx_flaky_88192",
            "amount": payload.get("amount", 0)
        }

def process_charge(client: PaymentGatewayClient, amount: float) -> dict:
    return client.send_payment({"amount": amount})
