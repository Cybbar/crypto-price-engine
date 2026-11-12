
from dataclasses import dataclass
from math import isfinite

# Initial paper-trading risk settings
MAX_LEVERAGE = 3.0
MAX_RISK_PER_TRADE = 0.005  # 0.5%

# Experimental transaction-cost assumptions
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
ROUND_TRIP_COST = 2 * (FEE_RATE + SLIPPAGE_RATE)


@dataclass
class PositionSize:
    direction: str
    quantity_btc: float
    notional_usdt: float
    isolated_margin_usdt: float
    risk_budget_usdt: float
    estimated_loss_usdt: float


def calculate_position_size(
    account_balance,
    entry_price,
    stop_price,
    direction,
):
    values = [account_balance, entry_price, stop_price]

    if not all(isfinite(x) and x > 0 for x in values):
        raise ValueError("Balance and prices must be positive and finite.")

    if direction not in ("long", "short"):
        raise ValueError("Direction must be long or short.")

    if direction == "long" and stop_price >= entry_price:
        raise ValueError("Long stop must be below entry.")

    if direction == "short" and stop_price <= entry_price:
        raise ValueError("Short stop must be above entry.")

    # Maximum amount of account equity we plan to risk
    risk_budget = account_balance * MAX_RISK_PER_TRADE

    # Price movement against our position
    stop_distance_pct = abs(entry_price - stop_price) / entry_price

    # Include estimated trading costs in the risk budget
    total_risk_pct = stop_distance_pct + ROUND_TRIP_COST

    max_notional_by_risk = risk_budget / total_risk_pct

    # Maximum notional exposure at our leverage ceiling
    max_notional_by_leverage = account_balance * MAX_LEVERAGE

    notional = min(
        max_notional_by_risk,
        max_notional_by_leverage,
    )

    quantity = notional / entry_price
    margin = notional / MAX_LEVERAGE
    estimated_loss = notional * total_risk_pct

    return PositionSize(
        direction=direction,
        quantity_btc=quantity,
        notional_usdt=notional,
        isolated_margin_usdt=margin,
        risk_budget_usdt=risk_budget,
        estimated_loss_usdt=estimated_loss,
    )


if __name__ == "__main__":
    # Example paper trade — not a live trading signal
    position = calculate_position_size(
        account_balance=1000,
        entry_price=100000,
        stop_price=98000,
        direction="long",
    )

    print("PAPER TRADING RISK TEST")
    print("-" * 35)
    print(f"Direction: {position.direction}")
    print(f"BTC quantity: {position.quantity_btc:.6f}")
    print(f"Position value: ${position.notional_usdt:.2f}")
    print(f"Isolated margin: ${position.isolated_margin_usdt:.2f}")
    print(f"Risk budget: ${position.risk_budget_usdt:.2f}")
    print(f"Estimated stop loss: ${position.estimated_loss_usdt:.2f}")
