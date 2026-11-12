
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

FEE_RATE = Decimal("0.001")
SLIPPAGE_RATE = Decimal("0.0005")
PRECISION = Decimal("0.00000001")


def positive_decimal(value, name):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if not number.is_finite() or number <= 0:
        raise ValueError(f"{name} must be positive and finite.")

    return number


def calculate_trade_close(position, exit_price):
    direction = position["direction"]

    if direction not in ("long", "short"):
        raise ValueError("Invalid trade direction.")

    entry = positive_decimal(
        position["entry_price"], "entry_price"
    )
    exit_price = positive_decimal(exit_price, "exit_price")
    quantity = positive_decimal(
        position["quantity_btc"], "quantity_btc"
    )

    # Apply adverse slippage to the simulated exit.
    if direction == "long":
        filled_exit = exit_price * (1 - SLIPPAGE_RATE)
        realized_pnl = (filled_exit - entry) * quantity
    else:
        filled_exit = exit_price * (1 + SLIPPAGE_RATE)
        realized_pnl = (entry - filled_exit) * quantity

    closing_fee = filled_exit * quantity * FEE_RATE
    net_pnl = realized_pnl - closing_fee

    return {
        "position_id": position["id"],
        "direction": direction,
        "filled_exit_price": str(filled_exit),
        "realized_pnl_usdt": str(
            realized_pnl.quantize(
                PRECISION, rounding=ROUND_HALF_UP
            )
        ),
        "closing_fee_usdt": str(
            closing_fee.quantize(
                PRECISION, rounding=ROUND_HALF_UP
            )
        ),
        "net_pnl_usdt": str(
            net_pnl.quantize(
                PRECISION, rounding=ROUND_HALF_UP
            )
        ),
    }


def main():
    print("PAPER TRADE CLOSING TEST")
    print("-" * 40)

    positions = [
        {
            "id": "long001",
            "direction": "long",
            "entry_price": "100000",
            "quantity_btc": "0.002",
        },
        {
            "id": "short001",
            "direction": "short",
            "entry_price": "100000",
            "quantity_btc": "0.002",
        },
    ]

    for position, exit_price in zip(
        positions, ["101000", "99000"]
    ):
        result = calculate_trade_close(
            position, exit_price
        )

        print(f"\nPosition: {result['position_id']}")
        print(f"Direction: {result['direction']}")
        print(f"Realized P&L: ${result['realized_pnl_usdt']}")
        print(f"Closing fee: ${result['closing_fee_usdt']}")
        print(f"Net closing P&L: ${result['net_pnl_usdt']}")


if __name__ == "__main__":
    main()
