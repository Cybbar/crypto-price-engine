"""Require a recent successful, verified mark-price equity check for paper opens.

Read-only Binance market data; no exchange authentication or order submission.
This is a fail-closed pre-entry heartbeat, NOT a continuous monitoring service.
"""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .sqlite_equity_monitor import monitor_account_equity

# At most 10 seconds old at check time, then no more than 15 further seconds
# before a new paper opening. Both thresholds are preliminary testing limits.
MAX_MARK_AGE_SECONDS = 10
MAX_HEARTBEAT_AGE_SECONDS = 15


def _positive_decimal(value, name):
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name}; monitoring unavailable.") from exc
    if not amount.is_finite() or amount <= 0:
        raise RuntimeError(f"Invalid {name}; monitoring unavailable.")
    return amount


def record_verified_equity_check(db_path):
    """Fetch/validate a mark, monitor equity, then persist a fresh heartbeat.

    Fails closed if the account changes between the equity snapshot and stamp.
    Caller must treat any failure as disabling new paper-trade openings.
    """
    # Import here to keep offline tests free of exchange calls.
    from .live_equity_check import fetch_validated_mark_price

    mark, age = fetch_validated_mark_price()
    mark = _positive_decimal(mark, "mark price")
    try:
        age = float(age)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Invalid mark-price age.") from exc
    if not 0 <= age <= MAX_MARK_AGE_SECONDS:
        raise RuntimeError("Mark-price data too old for monitoring heartbeat.")

    snapshot = monitor_account_equity(db_path, str(mark))
    path = Path(db_path)
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected paper database schema.")

        account = conn.execute(
            "SELECT balance, currency FROM account WHERE id=1"
        ).fetchone()
        risk = conn.execute(
            "SELECT utc_day, locked FROM daily_risk WHERE id=1"
        ).fetchone()
        positions = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE status='open'"
        ).fetchone()[0]
        tx_count = conn.execute(
            "SELECT COUNT(*) FROM transactions"
        ).fetchone()[0]
        now = datetime.now(timezone.utc)
        if (account is None or account[1] != "USDT" or risk is None or
                risk[0] != now.date().isoformat() or
                Decimal(account[0]) != Decimal(snapshot["cash_balance_usdt"]) or
                positions != snapshot["open_positions"] or
                bool(risk[1]) != snapshot["trading_locked"]):
            raise RuntimeError("Account changed during monitoring; heartbeat not recorded.")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS monitor_heartbeat (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                utc_day TEXT NOT NULL,
                checked_at_utc TEXT NOT NULL,
                cash_balance TEXT NOT NULL,
                open_positions INTEGER NOT NULL,
                transaction_count INTEGER NOT NULL,
                equity TEXT NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO monitor_heartbeat
                (id, utc_day, checked_at_utc, cash_balance,
                 open_positions, transaction_count, equity)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                utc_day=excluded.utc_day,
                checked_at_utc=excluded.checked_at_utc,
                cash_balance=excluded.cash_balance,
                open_positions=excluded.open_positions,
                transaction_count=excluded.transaction_count,
                equity=excluded.equity
        """, (
            now.date().isoformat(), now.isoformat(),
            snapshot["cash_balance_usdt"], positions, tx_count,
            snapshot["equity_usdt"],
        ))

    return {"mark_price": str(mark), "mark_age_seconds": age, **snapshot}


def check_monitor_heartbeat_for_open(conn, current_cash_balance):
    """Inspect heartbeat inside the opening transaction; never auto-initialize."""
    try:
        row = conn.execute("""
            SELECT utc_day, checked_at_utc, cash_balance,
                   open_positions, transaction_count, equity
            FROM monitor_heartbeat WHERE id=1
        """).fetchone()
    except sqlite3.DatabaseError:
        return False, "Monitoring heartbeat missing or unavailable."
    if row is None:
        return False, "Monitoring heartbeat missing."

    try:
        day, checked_text, cash_text, positions_count, tx_count, equity_text = row
        now = datetime.now(timezone.utc)
        checked = datetime.fromisoformat(checked_text)
        if checked.tzinfo is None or checked.utcoffset() is None:
            raise ValueError("Naive timestamp")
        age = (now - checked).total_seconds()
        cash = _positive_decimal(cash_text, "saved cash")
        equity = _positive_decimal(equity_text, "saved equity")
        current_cash = _positive_decimal(current_cash_balance, "account cash")
        actual_tx_count = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        actual_open_positions = conn.execute(
            "SELECT COUNT(*) FROM positions WHERE status='open'"
        ).fetchone()[0]
        if (day != now.date().isoformat() or
                not 0 <= age <= MAX_HEARTBEAT_AGE_SECONDS or
                cash != current_cash or equity != current_cash or
                positions_count != 0 or actual_open_positions != 0 or
                tx_count != actual_tx_count):
            return False, "Monitor heartbeat stale or account changed."
    except (ValueError, TypeError, RuntimeError, sqlite3.DatabaseError, OverflowError):
        return False, "Invalid monitoring heartbeat."

    return True, "Recent equity monitoring confirmed."


def main():
    from .paper_database import initialize_paper_database
    from .sqlite_daily_risk import initialize_daily_risk_day

    print("PAPER MONITOR HEARTBEAT TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(conn, "1000")
        assert not approved
        print("Missing heartbeat: BLOCKED")

        # Mock only the market fetch for this test. Production obtains real data.
        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 0.5)):
            record_verified_equity_check(db)
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(conn, "1000")
        assert approved
        print("Recent verified heartbeat: APPROVED")

        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE monitor_heartbeat SET checked_at_utc=? WHERE id=1",
                         ("2000-01-01T00:00:00+00:00",))
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(conn, "1000")
        assert not approved
        print("Stale heartbeat: BLOCKED")

        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 0.5)):
            record_verified_equity_check(db)
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(conn, "999")
        assert not approved
        print("Changed balance: BLOCKED")

        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 45)):
            try:
                record_verified_equity_check(db)
            except RuntimeError:
                print("Stale Binance mark: BLOCKED")
            else:
                raise AssertionError("Stale mark was accepted")


if __name__ == "__main__":
    main()
