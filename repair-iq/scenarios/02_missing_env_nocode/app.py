"""Payment Processing Gateway Client."""
import os

def get_payment_gateway_url() -> str:
    """
    Fetches the payment gateway URL from environment variables.
    Requires PAYMENT_GATEWAY_URL to be set in the execution environment.
    """
    return os.environ["PAYMENT_GATEWAY_URL"]

def process_payment(order_id: str, amount_cents: int) -> dict:
    url = get_payment_gateway_url()
    return {
        "status": "ready",
        "gateway": url,
        "order_id": order_id,
        "amount_cents": amount_cents
    }
