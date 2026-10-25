"""Tax Tier Service with Luxury Bracket Precision."""

def compute_tier_tax(amount: float, tier: str) -> float:
    """
    Computes tax based on customer tier.
    Standard: 15%
    Luxury: 28.45%
    """
    if tier == "luxury":
        # BUG: Truncates rounding to 1 decimal place instead of 2 decimals
        return round(amount * 0.2845, 1)
    elif tier == "standard":
        return round(amount * 0.15, 2)
    return 0.0

def process_invoice(amount: float, tier: str) -> dict:
    tax = compute_tier_tax(amount, tier)
    return {
        "subtotal": amount,
        "tax": tax,
        "total": round(amount + tax, 2)
    }
