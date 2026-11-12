from pathlib import Path
from tempfile import TemporaryDirectory
from .daily_risk_guard import DailyRiskGuard, MAX_DAILY_LOSS
from .risk_engine import calculate_position_size


def check_paper_trade(
    guard,
    account_equity,
    entry_price,
    stop_price,
    direction,
    timestamp=None,
):
    # Check daily circuit breaker first.
    if not guard.can_open_trade(account_equity, timestamp):
        return False, "Daily risk guard is locked", None

    position = calculate_position_size(
        account_balance=account_equity,
        entry_price=entry_price,
        stop_price=stop_price,
        direction=direction,
    )

    # Do not approve a trade whose estimated loss would
    # exceed the remaining daily loss allowance.
    daily_equity_floor = (
        guard.starting_equity * (1 - MAX_DAILY_LOSS)
    )
    remaining_daily_budget = account_equity - daily_equity_floor

    if position.estimated_loss_usdt >= remaining_daily_budget:
        return False, "Insufficient daily risk allowance", None

    return True, "Paper trade approved", position


def main():
    print("PERSISTENT RISK MANAGER TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        state_file = Path(directory) / "daily_risk.json"

        guard = DailyRiskGuard(state_path=state_file)

        for equity in [1000, 989, 985]:
            approved, message, position = check_paper_trade(
                guard=guard,
                account_equity=equity,
                entry_price=100000,
                stop_price=98000,
                direction="long",
            )

            print(
                f"Equity: ${equity:.2f} | "
                f"{'APPROVED' if approved else 'BLOCKED'} | "
                f"{message}"
            )

        # Simulate restarting the paper-trading process.
        restarted_guard = DailyRiskGuard(
            state_path=state_file
        )

        approved, message, position = check_paper_trade(
            guard=restarted_guard,
            account_equity=990,
            entry_price=100000,
            stop_price=98000,
            direction="long",
        )

        print(
            f"After restart: "
            f"{'APPROVED' if approved else 'BLOCKED'} | "
            f"{message}"
        )


if __name__ == "__main__":
    main()