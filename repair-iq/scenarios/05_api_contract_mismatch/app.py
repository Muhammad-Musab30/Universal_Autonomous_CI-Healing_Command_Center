"""User Account Billing Summary Service."""

def get_account_summary(user_id: int) -> dict:
    return {
        "user_id": user_id,
        "total_cents": 5000
    }
