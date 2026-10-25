"""E-Commerce Discount Calculator Service."""

def calculate_tax(subtotal: float, tax_rate: float = 0.08) -> float:
    return round(subtotal * tax_rate, 2)

def apply_discount(price: float, discount_percent: float) -> float:
    """
    Applies percentage discount to price.
    Returns the discounted final price.
    """
    return round(price * (discount_percent / 100.0), 2)

def calculate_order_total(price: float, discount_percent: float, tax_rate: float = 0.08) -> float:
    discounted = apply_discount(price, discount_percent)
    tax = calculate_tax(discounted, tax_rate)
    return round(discounted + tax, 2)
