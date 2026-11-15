
"""Exit-only recovery for a paper position carried across UTC days.

No new entries, daily rollover, or heartbeat creation.
Must be called while the exclusive monitor lock is held.
"""

from decimal import Decimal, InvalidOperation
from pathlib import Path

from .paper_startup import inspect_paper_startup
from .live_equity_check import fetch_validated_mark_price
from .paper_stop_loss import evaluate_stop_exit
from .persistent_market_monitor import revoke_heartbeat


MAX_MARK_AGE_SECONDS = Decimal("10")


def check_exit_only_position(database_path):
    db = Path(database_path)

    # Confirm that an existing position requires previous-day recovery.
    account = inspect_paper_startup(db)

    if (
        account["status"] != "OPEN_POSITION_REQUIRES_RECONCILIATION"
        or account["saved_day"] >= account["current_utc_day"]
    ):
        raise RuntimeError(
            "Exit-only recovery requires an open position "
            "with a stale UTC risk date."
        )

    # Never authorize new entries during this recovery mode.
    revoke_heartbeat(db)

    # Fetch a fresh, validated Binance BTC perpetual mark price.
    mark_price, mark_age = fetch_validated_mark_price()

    try:
        mark = Decimal(str(mark_price))
        age = Decimal(str(mark_age))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise RuntimeError("Invalid recovery market data.") from exc

    if (
        not mark.is_finite()
        or mark <= 0
        or not age.is_finite()
        or age < 0
        or age > MAX_MARK_AGE_SECONDS
    ):
        raise RuntimeError(
            "Recovery mark price is invalid or stale."
        )

    # Only evaluate the saved position's exit condition.
    # Do not run the current-day equity/heartbeat authorization path.
    status, balance = evaluate_stop_exit(db, mark)

    if status not in ("HOLD", "SETTLED"):
        raise RuntimeError(
            "Unexpected position state during exit-only recovery."
        )

    return {
        "mark_price": str(mark),
        "mark_age_seconds": str(age),
        "stop_status": status,
        "settlement_balance_usdt": balance,
        "new_entries_allowed": False,
    }
