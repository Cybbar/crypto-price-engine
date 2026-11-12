
from pathlib import Path
from tempfile import TemporaryDirectory

from .account_ledger import PaperAccountLedger
from .position_store import PaperPositionStore
from .account_equity import (
    calculate_account_equity,
    positive_decimal,
)
from .daily_risk_guard import DailyRiskGuard


def get_account_snapshot(ledger, position_store, mark_price=None):
    # Restore positions from persistent storage.
    state = position_store.load()
    positions = state["positions"]

    if positions:
        if mark_price is None:
            raise ValueError(
                "A mark price is required for an open position."
            )

        mark = positive_decimal(mark_price, "mark_price")

        # Add the current market price to each restored position.
        open_positions = [
            {**position, "mark_price": str(mark)}
            for position in positions
        ]
    else:
        open_positions = []

    return calculate_account_equity(
        ledger,
        open_positions,
    )


def main():
    print("PERSISTENT ACCOUNT EQUITY TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        directory = Path(directory)

        account_path = directory / "account.json"
        position_path = directory / "positions.json"
        risk_path = directory / "daily_risk.json"

        ledger = PaperAccountLedger(account_path)
        ledger.initialize(1000)

        positions = PaperPositionStore(position_path)
        positions.initialize()

        guard = DailyRiskGuard(state_path=risk_path)
        guard.can_open_trade(1000)

        # Simulated fee and position for testing only.
        ledger.record_transaction(
            "fee001", "trading_fee", "-1"
        )

        positions.record_open_position(
            position_id="position001",
            direction="long",
            entry_price="100000",
            stop_price="98000",
            quantity_btc="0.002",
        )

        # Simulate restarting the application.
        restarted_ledger = PaperAccountLedger(account_path)
        restarted_positions = PaperPositionStore(position_path)
        restarted_guard = DailyRiskGuard(state_path=risk_path)

        account = get_account_snapshot(
            restarted_ledger,
            restarted_positions,
            mark_price="99000",
        )

        print(f"Restored cash: ${account['cash_balance']}")
        print(f"Restored equity: ${account['equity']}")

        allowed = restarted_guard.can_open_trade(
            float(account["equity"])
        )

        print(
            "Risk status:",
            "ALLOWED" if allowed else "BLOCKED"
        )

        # Simulate a further price decline.
        account = get_account_snapshot(
            restarted_ledger,
            restarted_positions,
            mark_price="90000",
        )

        print(f"Equity after decline: ${account['equity']}")

        allowed = restarted_guard.can_open_trade(
            float(account["equity"])
        )

        print(
            "Risk status:",
            "ALLOWED" if allowed else "BLOCKED"
        )

        final_guard = DailyRiskGuard(state_path=risk_path)

        print(
            "After another restart:",
            "BLOCKED" if final_guard.trading_locked else "ALLOWED"
        )


if __name__ == "__main__":
    main()
