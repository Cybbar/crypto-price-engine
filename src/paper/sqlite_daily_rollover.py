"""Explicit, reconciled daily rollover for the version-1 SQLite paper account.

Single-process paper-trading prototype. Never places exchange orders. Requires
an account with no open positions and a known original funding balance.
"""

import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from .paper_database import initialize_paper_database
from .sqlite_daily_risk import initialize_daily_risk_day


def _money(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label}; rollover blocked.") from exc
    if not number.is_finite():
        raise RuntimeError(f"Invalid {label}; rollover blocked.")
    return number


def rollover_reconciled_day(database_path, original_funding="1000"):
    """Rollover only after reconciling balance and confirming a flat account.

    The original funding amount is fixed for this paper-account prototype;
    later deposits/withdrawals must be recorded as ledger transactions.
    Returns the new UTC trading day. Repeated same-day calls are rejected.
    """
    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing; rollover blocked.")

    today = datetime.now(timezone.utc).date()
    funding = _money(original_funding, "original funding")
    if funding <= 0:
        raise RuntimeError("Original funding must be positive.")

    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected database schema; rollover blocked.")

        account = conn.execute(
            "SELECT balance, currency FROM account WHERE id = 1"
        ).fetchone()
        if account is None or account[1] != "USDT":
            raise RuntimeError("Invalid paper account; rollover blocked.")
        balance = _money(account[0], "account balance")
        if balance <= 0:
            raise RuntimeError("Nonpositive account equity; rollover blocked.")

        if conn.execute(
            "SELECT 1 FROM positions WHERE status = 'open' LIMIT 1"
        ).fetchone():
            raise RuntimeError("Open position exists; rollover blocked.")

        state = conn.execute(
            "SELECT utc_day, starting_equity, locked FROM daily_risk WHERE id=1"
        ).fetchone()
        if state is None:
            raise RuntimeError("Daily risk state missing; rollover blocked.")
        try:
            previous_day = date.fromisoformat(state[0])
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Invalid saved UTC day; rollover blocked.") from exc
        if previous_day >= today:
            raise RuntimeError("Same-day or future-day rollover blocked.")
        if _money(state[1], "saved starting equity") <= 0 or state[2] not in (0, 1):
            raise RuntimeError("Invalid saved daily risk state; rollover blocked.")

        expected = funding
        for (amount_text,) in conn.execute("SELECT amount FROM transactions"):
            expected += _money(amount_text, "transaction amount")
        if expected != balance:
            raise RuntimeError(
                f"Ledger mismatch: recorded {balance}, expected {expected}; rollover blocked."
            )

        # The old heartbeat must not authorize new entries after rollover.
        heartbeat_exists = conn.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type='table' AND name='monitor_heartbeat'"
        ).fetchone()
        if heartbeat_exists:
            conn.execute("DELETE FROM monitor_heartbeat WHERE id=1")

        conn.execute(
            "UPDATE daily_risk SET utc_day=?, starting_equity=?, locked=0 WHERE id=1",
            (today.isoformat(), str(balance)),
        )

    return today.isoformat()


def main():
    print("RECONCILED DAILY ROLLOVER TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "rollover.sqlite3"
        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)
        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute(
                "UPDATE daily_risk SET utc_day=?, locked=1 WHERE id=1",
                (yesterday,),
            )
            conn.execute("CREATE TABLE monitor_heartbeat(id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO monitor_heartbeat VALUES (1)")

        new_day = rollover_reconciled_day(db)
        with sqlite3.connect(db) as conn:
            day, start, locked = conn.execute(
                "SELECT utc_day, starting_equity, locked FROM daily_risk WHERE id=1"
            ).fetchone()
            heartbeat_count = conn.execute(
                "SELECT COUNT(*) FROM monitor_heartbeat"
            ).fetchone()[0]
        assert day == new_day and start == "1000" and locked == 0
        assert heartbeat_count == 0
        print("Reconciled new UTC day:", new_day)
        print("Starting equity: 1000 USDT")
        print("Old heartbeat: REVOKED")

        try:
            rollover_reconciled_day(db)
        except RuntimeError:
            print("Same-day rollover: BLOCKED")
        else:
            raise AssertionError("Same-day rollover should be blocked")

        # A corrupted balance must not be used as tomorrow's new baseline.
        damaged = Path(directory) / "mismatch.sqlite3"
        initialize_paper_database(damaged, "1000")
        initialize_daily_risk_day(damaged)
        with sqlite3.connect(damaged) as conn:
            conn.execute("UPDATE daily_risk SET utc_day=?, locked=1 WHERE id=1", (yesterday,))
            conn.execute("UPDATE account SET balance='999' WHERE id=1")
        try:
            rollover_reconciled_day(damaged)
        except RuntimeError as exc:
            assert "Ledger mismatch" in str(exc)
            print("Unreconciled balance: BLOCKED")
        else:
            raise AssertionError("Invalid balance should be blocked")
        with sqlite3.connect(damaged) as conn:
            day_after, locked_after = conn.execute(
                "SELECT utc_day, locked FROM daily_risk WHERE id=1"
            ).fetchone()
        assert day_after == yesterday and locked_after == 1
        print("Failed rollover left old lock intact: CONFIRMED")
        print("Permanent account modified: NO")


if __name__ == "__main__":
    main()
