"""Read-only pricing plus persistent daily loss lock for a paper account.

No exchange requests or orders. Caller must supply a fresh, verified BTC
perpetual mark price when a position is open. This is a one-shot check, not a
background/continuous price watcher. SQLite version 1 schema expected.
"""

import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory

from .sqlite_daily_risk import DAILY_LOSS_LIMIT


def _decimal(value, label, *, positive=False):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label}; monitoring failed closed.") from exc
    if not number.is_finite() or (number <= 0 if positive else number < 0):
        raise RuntimeError(f"Invalid {label}; monitoring failed closed.")
    return number


def monitor_account_equity(db_path, mark_price=None):
    """Recalculate equity and persist a sticky UTC-day lock if breached.

    Returns a snapshot; never opens or closes positions. Errors must disable
    trade authorization in the caller. This function does not establish mark
    price authenticity or freshness, and does not initialize a new UTC day.
    """
    path = Path(db_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing; monitoring failed closed.")

    today = datetime.now(timezone.utc).date().isoformat()
    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected paper database schema.")

        account = conn.execute(
            "SELECT balance, currency FROM account WHERE id = 1"
        ).fetchone()
        if account is None or account[1] != "USDT":
            raise RuntimeError("Invalid paper account; monitoring blocked.")
        cash = _decimal(account[0], "cash balance")

        risk = conn.execute(
            "SELECT utc_day, starting_equity, locked FROM daily_risk WHERE id = 1"
        ).fetchone()
        if risk is None:
            raise RuntimeError("Daily risk state missing; monitoring blocked.")
        day, start_text, locked = risk
        if day != today:
            raise RuntimeError("Daily risk day is stale; reconciliation required.")
        if locked not in (0, 1):
            raise RuntimeError("Invalid daily risk lock; monitoring blocked.")
        starting_equity = _decimal(start_text, "starting equity", positive=True)

        positions = conn.execute(
            "SELECT direction, entry_price, quantity_btc "
            "FROM positions WHERE status = 'open'"
        ).fetchall()
        if len(positions) > 1:
            raise RuntimeError("More than one open position; monitoring blocked.")

        unrealized = Decimal("0")
        if positions:
            if mark_price is None:
                raise RuntimeError("Open position has no mark price; monitoring blocked.")
            mark = _decimal(mark_price, "BTC mark price", positive=True)
            direction, entry_text, quantity_text = positions[0]
            entry = _decimal(entry_text, "entry price", positive=True)
            quantity = _decimal(quantity_text, "BTC quantity", positive=True)
            if direction == "long":
                unrealized = (mark - entry) * quantity
            elif direction == "short":
                unrealized = (entry - mark) * quantity
            else:
                raise RuntimeError("Invalid position direction; monitoring blocked.")

        equity = cash + unrealized
        floor = starting_equity * (Decimal("1") - DAILY_LOSS_LIMIT)
        if equity <= floor and not locked:
            conn.execute("UPDATE daily_risk SET locked = 1 WHERE id = 1")
            locked = 1

        return {
            "cash_balance_usdt": str(cash),
            "unrealized_pnl_usdt": str(unrealized),
            "equity_usdt": str(equity),
            "daily_floor_usdt": str(floor),
            "trading_locked": bool(locked),
            "open_positions": len(positions),
        }


def main():
    # A self-contained test database matching the current v1 schema.
    # The normal application creates this schema in paper_database.py.
    print("SQLITE PAPER EQUITY MONITOR TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                PRAGMA user_version = 1;
                CREATE TABLE account (
                    id INTEGER PRIMARY KEY, balance TEXT NOT NULL,
                    currency TEXT NOT NULL
                );
                CREATE TABLE positions (
                    id TEXT PRIMARY KEY, direction TEXT NOT NULL,
                    entry_price TEXT NOT NULL, stop_price TEXT,
                    quantity_btc TEXT NOT NULL, status TEXT NOT NULL,
                    exit_price TEXT
                );
                CREATE TABLE daily_risk (
                    id INTEGER PRIMARY KEY, utc_day TEXT NOT NULL,
                    starting_equity TEXT NOT NULL, locked INTEGER NOT NULL
                );
                INSERT INTO account VALUES (1, '999.79990000', 'USDT');
                INSERT INTO positions VALUES (
                    'demo001', 'long', '100050', '98000', '0.002', 'open', NULL
                );
            """)
            conn.execute(
                "INSERT INTO daily_risk VALUES (1, ?, '1000', 0)",
                (datetime.now(timezone.utc).date().isoformat(),),
            )

        normal = monitor_account_equity(db, "99000")
        assert normal["equity_usdt"] == "997.69990000"
        assert not normal["trading_locked"]
        print("Equity at 99000: 997.69990000 USDT | ALLOWED")

        breach = monitor_account_equity(db, "90000")
        assert breach["equity_usdt"] == "979.69990000"
        assert breach["trading_locked"]
        print("Equity at 90000: 979.69990000 USDT | BLOCKED")

        restarted = monitor_account_equity(db, "100000")
        assert restarted["trading_locked"]
        print("After recovery/restart: BLOCKED")

        with sqlite3.connect(db) as conn:
            assert conn.execute(
                "SELECT locked FROM daily_risk WHERE id = 1"
            ).fetchone()[0] == 1
        print("SQLite daily lock persisted: CONFIRMED")

        try:
            monitor_account_equity(db, "invalid")
        except RuntimeError:
            print("Invalid mark price: BLOCKED")
        else:
            raise AssertionError("Invalid mark price was accepted")


if __name__ == "__main__":
    main()
