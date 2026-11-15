from .monitor_lock import exclusive_monitor_lock
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import time
from .exit_only_recovery import check_exit_only_position

from .paper_startup import inspect_paper_startup
from .paper_database import initialize_paper_database
from .sqlite_daily_risk import initialize_daily_risk_day
from .persistent_market_monitor import (
    run_monitor,
    revoke_heartbeat,
)
from .sqlite_monitor_heartbeat import (
    check_monitor_heartbeat_for_open,
)


def validate_open_position_recovery(
    db, account, allow_previous_day=False
):
    """Validate an existing paper position before resuming monitoring."""

    if account["status"] != "OPEN_POSITION_REQUIRES_RECONCILIATION":
        raise RuntimeError("No position recovery required.")

    # Do not silently reset the daily risk state.
    if account["saved_day"] > account["current_utc_day"]:
        raise RuntimeError(
            "Future-dated position risk state; recovery blocked."
        )

    if (
            account["saved_day"] < account["current_utc_day"]
            and not allow_previous_day
    ):
        raise RuntimeError(
            "Previous-day position requires exit-only recovery."
        )

    try:
        with sqlite3.connect(
            db.resolve().as_uri() + "?mode=ro",
            uri=True,
        ) as conn:
            positions = conn.execute(
                "SELECT id, direction, entry_price, stop_price, "
                "quantity_btc, exit_price "
                "FROM positions WHERE status='open'"
            ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise RuntimeError(
            "Cannot inspect saved position; recovery blocked."
        ) from exc

    if len(positions) != 1 or account["open_positions"] != 1:
        raise RuntimeError(
            "Unexpected open position count; recovery blocked."
        )

    position_id, direction, entry, stop, quantity, exit_price = positions[0]

    if not isinstance(position_id, str) or not position_id.strip():
        raise RuntimeError("Invalid position ID; recovery blocked.")

    if direction not in ("long", "short") or exit_price is not None:
        raise RuntimeError("Invalid saved position; recovery blocked.")

    try:
        entry = Decimal(entry)
        stop = Decimal(stop)
        quantity = Decimal(quantity)
    except Exception as exc:
        raise RuntimeError(
            "Invalid position numbers; recovery blocked."
        ) from exc

    if any(
        not value.is_finite() or value <= 0
        for value in (entry, stop, quantity)
    ):
        raise RuntimeError(
            "Invalid position values; recovery blocked."
        )

    if direction == "long" and stop >= entry:
        raise RuntimeError("Invalid long stop-loss; recovery blocked.")

    if direction == "short" and stop <= entry:
        raise RuntimeError("Invalid short stop-loss; recovery blocked.")

    return {
        "position_id": position_id,
        "direction": direction,
        "entry_price": str(entry),
        "stop_price": str(stop),
        "quantity_btc": str(quantity),
    }


def run_exit_only_monitor(db_path, cycles=3, interval_seconds=5):
    """Monitor an overnight paper position without allowing new entries.

    Must run while the exclusive monitor lock is held.
    """

    if not isinstance(cycles, int) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")

    if not 0 <= interval_seconds <= 3600:
        raise ValueError("Invalid monitoring interval")

    print("EXIT-ONLY RECOVERY MODE: ACTIVE", flush=True)
    print("New position entries: DISABLED", flush=True)
    print("Automatic daily rollover: DISABLED", flush=True)

    count = 0

    while cycles == 0 or count < cycles:
        # This function checks a fresh validated market price,
        # evaluates the stop, and never creates a heartbeat.
        result = check_exit_only_position(db_path)

        count += 1

        print(
            f"Recovery check {count}: "
            f"mark={result['mark_price']} USDT | "
            f"stop={result['stop_status']}",
            flush=True,
        )

        if result["stop_status"] == "SETTLED":
            print(
                "OVERNIGHT PAPER POSITION CLOSED | "
                f"Balance: {result['settlement_balance_usdt']} USDT",
                flush=True,
            )
            print(
                "Daily reconciliation required before new entries.",
                flush=True,
            )
            break

        if cycles == 0 or count < cycles:
            time.sleep(interval_seconds)

    return count


def start_safe_monitor(db_path, cycles=3, interval_seconds=5):
    db = Path(db_path)

    if not db.is_file():
        raise RuntimeError("Paper database missing. Startup blocked.")

    # Only one cooperating monitor can control this account.
    # Acquire the lock BEFORE changing heartbeat state.
    with exclusive_monitor_lock(db):

        # Remove any previous monitoring authorization.
        revoke_heartbeat(db)

        try:
            account = inspect_paper_startup(db)


            if account["status"] == "READY_FOR_MONITOR":
                print("Normal startup: APPROVED")


            elif account["status"] == "OPEN_POSITION_REQUIRES_RECONCILIATION":

                if account["saved_day"] < account["current_utc_day"]:
                    recovered = validate_open_position_recovery(
                        db,
                        account,
                        allow_previous_day=True,
                    )

                    print("Previous-day open position detected.")
                    print("Saved position: VALIDATED")
                    print("Position ID:", recovered["position_id"])
                    print("Direction:", recovered["direction"])
                    print("Stop-loss:", recovered["stop_price"])
                    print("Starting exit-only recovery.")
                    return run_exit_only_monitor(

                        db_path=db,

                        cycles=cycles,

                        interval_seconds=interval_seconds,

                    )

                recovered = validate_open_position_recovery(db, account)
                print("Existing paper position: VALIDATED")
                print("Position ID:", recovered["position_id"])
                print("Direction:", recovered["direction"])
                print("Entry price:", recovered["entry_price"])
                print("Stop-loss:", recovered["stop_price"])
                print("Quantity BTC:", recovered["quantity_btc"])
                print("Recovery mode: MONITOR EXISTING POSITION")

            else:
                raise RuntimeError(
                    f"Startup blocked: {account['status']}"
                )


            print("Startup preflight: PASSED")
            print("Account balance:", account["cash_usdt"], "USDT")
            print("Monitor process lock: ACQUIRED")
            print("Starting market monitor...")

            return run_monitor(
                database_path=db,
                cycles=cycles,
                interval_seconds=interval_seconds,
            )

        finally:
            # Revoke authorization before releasing the lock.
            # Also runs on normal shutdown, errors, or Ctrl+C.
            revoke_heartbeat(db)


def main():
    print("SAFE PAPER MONITOR STARTUP TEST")
    print("-" * 45)

    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"

        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)

        # Use fictional mark prices: no network calls.
        with patch(
            "src.paper.live_equity_check.fetch_validated_mark_price",
            return_value=(Decimal("100000"), 0.2),
        ):
            completed = start_safe_monitor(
                db,
                cycles=3,
                interval_seconds=0,
            )

        assert completed == 3
        print("Monitoring cycles completed:", completed)

        # The heartbeat must be revoked after shutdown.
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(
                conn, "1000"
            )

        assert not approved
        print("After shutdown: BLOCKED")

        # Simulate a database from the previous UTC day.
        yesterday = (
            datetime.now(timezone.utc).date()
            - timedelta(days=1)
        ).isoformat()

        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE daily_risk SET utc_day=? WHERE id=1",
                (yesterday,),
            )

        try:
            start_safe_monitor(
                db,
                cycles=1,
                interval_seconds=0,
            )
        except RuntimeError as exc:
            assert "ROLLOVER_REQUIRED" in str(exc)
            print("Previous-day startup: BLOCKED")
        else:
            raise AssertionError("Unsafe startup was accepted.")

        with sqlite3.connect(db) as conn:
            saved_day = conn.execute(
                "SELECT utc_day FROM daily_risk WHERE id=1"
            ).fetchone()[0]

        assert saved_day == yesterday
        print("Automatic daily reset: DISABLED")

    print("SAFE STARTUP TEST: PASSED")
    print("Real exchange orders: NONE")


if __name__ == "__main__":
    main()
