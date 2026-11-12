"""Business catalog: channels, costs and offers the agents may use."""

CHANNEL_COST_INR = {"email": 0.25, "push": 0.10, "sms": 0.30, "whatsapp": 0.90}
SMS_MAX_CHARS = 160
PUSH_MAX_CHARS = 120

OFFERS = {
    "none": "Content only (product news, recommendations) - no incentive.",
    "percent_discount": "Percentage discount on next order (set discount_pct).",
    "free_shipping": "Free shipping on next order.",
    "loyalty_points": "Bonus loyalty points on next order.",
    "early_access": "Early access to a new collection or sale.",
    "bundle": "Bundle deal on complementary products.",
}

OBJECTIVES = {
    "retain": "Keep valuable active customers engaged.",
    "upsell": "Increase order value of active customers.",
    "nurture": "Turn new customers into repeat buyers.",
    "win_back": "Bring back previously valuable customers who are lapsing.",
    "reactivate": "Re-engage long-dormant customers.",
}

# Segments whose names start with these are treated as VIP for margin protection.
VIP_SEGMENT_PREFIXES = ("Champions", "Loyal")

BANNED_PHRASES = [
    "guaranteed", "risk-free", "risk free", "no strings attached", "you have won",
    "winner", "100% free", "act now or lose", "last chance ever", "miracle",
]
