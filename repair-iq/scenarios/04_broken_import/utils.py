"""Core Financial Utilities."""

def compute_tax(amount: float, rate: float = 0.05) -> float:
    """Calculates tax based on tax rate (default 5%)."""
    return round(amount * rate, 2)

def format_currency(amount: float) -> str:
    return f"${amount:,.2f}"
