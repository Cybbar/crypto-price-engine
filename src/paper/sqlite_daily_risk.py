"""SQLite daily-loss gate for a single-process BTC perpetual paper account.

This is a pre-entry guard, NOT continuous monitoring. Call the explicit UTC-day
initializer only after reconciling the account and confirming no open positions.
Nothing here places exchange orders.
"""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from .paper_database import initialize_paper_database

DAILY_LOSS_LIMIT = Decimal("0.015")  # provisional 1.5% paper-account limit


def _utc_date():
    return datetime.now(timezone.utc).date().isoformat()


def _positive_decimal(value, label):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}.") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return result


def initialize_daily_risk_day(database_path):
    """Explicitly anchor today's daily loss ceiling to reconciled cash equity.

    Fail closed: requires an existing version-1 paper DB, no open positions,
    and refuses to reset an already-initialized day. Must only be called after
    accounting for all preceding activity; balance is equity only when flat.
    """
    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing. Day initialization blocked.")

    today = _utc_date()
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected paper database schema.")

        # Schema extension. Missing daily state is never auto-approved by open.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_risk (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                utc_day TEXT NOT NULL,
                starting_equity TEXT NOT NULL,
                locked INTEGER NOT NULL CHECK (locked IN (0, 1))
            )
        """)
        if conn.execute(
            "SELECT 1 FROM positions WHERE status = 'open' LIMIT 1"
        ).fetchone():
            raise RuntimeError("Open position exists; reconcile it before daily reset.")

        account = conn.execute(
            "SELECT balance, currency FROM account WHERE id = 1"
        ).fetchone()
        if account is None or account[1] != "USDT":
            raise RuntimeError("Invalid paper account.")
        balance = _positive_decimal(account[0], "account equity")

        previous = conn.execute(
            "SELECT utc_day FROM daily_risk WHERE id = 1"
        ).fetchone()
        if previous is not None:
            if previous[0] >= today:
                raise RuntimeError("Daily risk already initialized or future-dated. Reset blocked.")
            conn.execute(
                "UPDATE daily_risk SET utc_day=?, starting_equity=?, locked=0 WHERE id=1",
                (today, str(balance)),
            )
        else:
            conn.execute(
                "INSERT INTO daily_risk(id, utc_day, starting_equity, locked) "
                "VALUES(1, ?, ?, 0)",
                (today, str(balance)),
            )
    return today


def check_daily_risk_for_open(conn, account_equity, estimated_stop_loss):
    """Return (approved, message) using the CALLER'S open SQLite transaction.

    If the limit is reached, persists locked=1 in the current transaction.
    Callers MUST COMMIT that change before raising a rejection exception.
    Does not substitute for continuous mark-to-market equity monitoring.
    """
    try:
        state = conn.execute(
            "SELECT utc_day, starting_equity, locked FROM daily_risk WHERE id=1"
        ).fetchone()
    except sqlite3.DatabaseError:
        return False, "Daily risk state unavailable; opening blocked."
    if state is None:
        return False, "Daily risk state missing; opening blocked."
    day, starting_text, locked = state
    if day != _utc_date():
        return False, "Daily risk day stale; reconcile account before opening."
    if locked not in (0, 1):
        return False, "Daily risk state invalid; opening blocked."
    if locked:
        return False, "Daily loss guard locked."
    try:
        starting = _positive_decimal(starting_text, "starting equity")
        current = _positive_decimal(account_equity, "account equity")
        proposed = _positive_decimal(estimated_stop_loss, "estimated stop loss")
    except ValueError:
        return False, "Invalid daily risk inputs; opening blocked."

    floor = starting * (Decimal("1") - DAILY_LOSS_LIMIT)
    if current <= floor:
        conn.execute("UPDATE daily_risk SET locked=1 WHERE id=1")
        return False, "Daily loss limit reached; locked until reconciliation next day."
    if proposed >= current - floor:
        return False, "Estimated loss exceeds remaining daily allowance."
    return True, "Daily risk approved."


def main():
    print("SQLITE DAILY RISK TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        initialize_paper_database(db, "1000")

        with sqlite3.connect(db) as conn:
            allowed, _ = check_daily_risk_for_open(conn, "1000", "5")
        assert not allowed
        print("Missing daily state: BLOCKED")

        initialize_daily_risk_day(db)
        with sqlite3.connect(db) as conn:
            allowed, _ = check_daily_risk_for_open(conn, "1000", "5")
        assert allowed
        print("Within risk limits: APPROVED")

        with sqlite3.connect(db) as conn:
            allowed, _ = check_daily_risk_for_open(conn, "989", "5")
        assert not allowed
        print("Insufficient daily allowance: BLOCKED")

        with sqlite3.connect(db) as conn:
            allowed, _ = check_daily_risk_for_open(conn, "985", "5")
        assert not allowed
        print("Daily loss limit reached: BLOCKED")

        with sqlite3.connect(db) as conn:
            allowed, _ = check_daily_risk_for_open(conn, "990", "5")
        assert not allowed
        print("After simulated restart: BLOCKED")

        try:
            initialize_daily_risk_day(db)
        except RuntimeError:
            print("Same-day reset: BLOCKED")
        else:
            raise AssertionError("Reset should have been blocked.")


if __name__ == "__main__":
    main()
