
"""Paper stop-loss detection and settlement.

Uses supplied fictional prices for testing.
Does not fetch market data or place exchange orders.
"""

import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .paper_database import initialize_paper_database
from .sqlite_daily_risk import initialize_daily_risk_day
from .sqlite_monitor_heartbeat import record_verified_equity_check
from .atomic_trade_open import open_paper_position
from .atomic_trade_settlement import settle_position
from .sqlite_equity_monitor import monitor_account_equity


def evaluate_stop_exit(db_path, mark_price):
    path = Path(db_path)

    if not path.is_file():
        raise RuntimeError("Paper database missing.")

    try:
        mark = Decimal(str(mark_price))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid mark price.") from exc

    if not mark.is_finite() or mark <= 0:
        raise ValueError("Mark price must be positive and finite.")

    # Read the currently open position.
    with sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
    ) as conn:
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected database schema.")

        positions = conn.execute(
            "SELECT id, direction, stop_price "
            "FROM positions WHERE status='open'"
        ).fetchall()

    if len(positions) > 1:
        raise RuntimeError("Multiple open positions detected.")

    if not positions:
        return "NO_OPEN_POSITION", None

    position_id, direction, stop_text = positions[0]

    stop = Decimal(stop_text)

    if not stop.is_finite() or stop <= 0:
        raise RuntimeError("Invalid saved stop price.")

    if direction == "long":
        triggered = mark <= stop
    elif direction == "short":
        triggered = mark >= stop
    else:
        raise RuntimeError("Invalid position direction.")

    if not triggered:
        return "HOLD", None

    # Settle the paper trade at the observed mark price,
    # applying the existing simulated exit slippage and fee.
    status, balance = settle_position(
        path,
        position_id,
        str(mark),
    )

    return status, balance


def main():
    print("PAPER STOP-LOSS DETECTION TEST")
    print("-" * 45)

    scenarios = [
        ("long", "98000", "99000", "90000"),
        ("short", "102000", "101000", "110000"),
    ]

    with TemporaryDirectory() as directory:
        for direction, stop, hold_price, hit_price in scenarios:
            db = Path(directory) / f"{direction}.sqlite3"

            initialize_paper_database(db, "1000")
            initialize_daily_risk_day(db)

            # Fictional verified price for this isolated test.
            with patch(
                "src.paper.live_equity_check.fetch_validated_mark_price",
                return_value=(Decimal("100000"), 0.2),
            ):
                record_verified_equity_check(db)

            # Mock liveness only for this test.
            with patch(
                "src.paper.atomic_trade_open.monitor_is_running",
                return_value=True,
            ):
                open_paper_position(
                    db_path=db,
                    position_id=f"{direction}001",
                    direction=direction,
                    entry_price="100000",
                    stop_price=stop,
                    quantity_btc="0.002",
                )

            status, _ = evaluate_stop_exit(db, hold_price)
            assert status == "HOLD"
            print(f"{direction.upper()} before stop: HOLD")

            # Mark-to-market equity triggers the daily risk lock.
            snapshot = monitor_account_equity(db, hit_price)
            assert snapshot["trading_locked"]

            status, balance = evaluate_stop_exit(db, hit_price)
            assert status == "SETTLED"
            print(f"{direction.upper()} stop triggered: SETTLED")

            status, _ = evaluate_stop_exit(db, hit_price)
            assert status == "NO_OPEN_POSITION"
            print(f"{direction.upper()} repeat check: NO_OPEN_POSITION")

            with sqlite3.connect(db) as conn:
                open_count = conn.execute(
                    "SELECT COUNT(*) FROM positions "
                    "WHERE status='open'"
                ).fetchone()[0]

                locked = conn.execute(
                    "SELECT locked FROM daily_risk WHERE id=1"
                ).fetchone()[0]

                transactions = conn.execute(
                    "SELECT COUNT(*) FROM transactions"
                ).fetchone()[0]

            assert open_count == 0
            assert locked == 1
            assert transactions == 3

            print(f"{direction.upper()} final balance: {balance} USDT")
            print(f"{direction.upper()} daily lock: PERSISTED")
            print()

    print("PAPER STOP-LOSS TEST: PASSED")
    print("Real exchange orders: NONE")


if __name__ == "__main__":
    main()
