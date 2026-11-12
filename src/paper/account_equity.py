
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from .account_ledger import PaperAccountLedger
from .daily_risk_guard import DailyRiskGuard


def positive_decimal(value, name):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc

    if not number.is_finite() or number <= 0:
        raise ValueError(f"{name} must be positive and finite.")

    return number


def calculate_account_equity(ledger, open_positions):
    state = ledger.load()

    cash_balance = Decimal(state["cash_balance"])
    unrealized_pnl = Decimal("0")

    for position in open_positions:
        direction = position["direction"]

        if direction not in ("long", "short"):
            raise ValueError("Invalid position direction.")

        entry = positive_decimal(
            position["entry_price"], "entry_price"
        )
        mark = positive_decimal(
            position["mark_price"], "mark_price"
        )
        quantity = positive_decimal(
            position["quantity_btc"], "quantity_btc"
        )

        if direction == "long":
            pnl = (mark - entry) * quantity
        else:
            pnl = (entry - mark) * quantity

        unrealized_pnl += pnl

    equity = cash_balance + unrealized_pnl

    return {
        "cash_balance": cash_balance,
        "unrealized_pnl": unrealized_pnl,
        "equity": equity,
    }


def main():
    print("PAPER ACCOUNT EQUITY TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        account_path = Path(directory) / "account.json"
        risk_path = Path(directory) / "daily_risk.json"

        ledger = PaperAccountLedger(account_path)
        ledger.initialize(1000)

        guard = DailyRiskGuard(state_path=risk_path)

        # Establish the starting equity for the test day.
        guard.can_open_trade(1000)

        # Simulate a trading fee.
        ledger.record_transaction(
            "fee001", "trading_fee", "-1.00"
        )

        # Simulate one open BTC long position.
        positions = [{
            "direction": "long",
            "entry_price": "100000",
            "mark_price": "99000",
            "quantity_btc": "0.002",
        }]

        account = calculate_account_equity(
            ledger, positions
        )

        print(f"Cash balance: ${account['cash_balance']}")
        print(f"Unrealized P&L: ${account['unrealized_pnl']}")
        print(f"Account equity: ${account['equity']}")

        allowed = guard.can_open_trade(
            float(account["equity"])
        )

        print(
            "Daily risk status:",
            "ALLOWED" if allowed else "BLOCKED"
        )

        # Simulate a larger adverse BTC price movement.
        positions[0]["mark_price"] = "90000"

        account = calculate_account_equity(
            ledger, positions
        )

        print(f"\nEquity after price drop: ${account['equity']}")

        allowed = guard.can_open_trade(
            float(account["equity"])
        )

        print(
            "Daily risk status:",
            "ALLOWED" if allowed else "BLOCKED"
        )

        # Simulate a restart after the risk lock.
        restarted = DailyRiskGuard(state_path=risk_path)

        print(
            "After restart:",
            "BLOCKED" if restarted.trading_locked else "ALLOWED"
        )


if __name__ == "__main__":
    main()
