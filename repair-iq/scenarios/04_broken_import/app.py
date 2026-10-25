"""Billing Invoice Generator."""
from utils_v1 import calculate_tax

def generate_bill(subtotal: float) -> dict:
    tax = calculate_tax(subtotal, 0.05)
    return {
        "subtotal": subtotal,
        "tax": tax,
        "total": round(subtotal + tax, 2)
    }
