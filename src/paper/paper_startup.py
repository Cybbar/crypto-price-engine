"""Read-only startup preflight for the BTC/USDT perpetual paper account.

Does not roll over the day, update heartbeats, start monitoring, open
positions, or place exchange orders. Intended for a single-process prototype.
"""

import argparse
import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory


def _amount(value, label):
    try:
        value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {label}; startup blocked.") from exc
    if not value.is_finite():
        raise RuntimeError(f"Invalid {label}; startup blocked.")
    return value


def inspect_paper_startup(database_path, original_funding="1000"):
    """Check saved account state without changing the database.

    Possible statuses: READY_FOR_MONITOR, DAILY_LOCKED, ROLLOVER_REQUIRED,
    OPEN_POSITION_REQUIRES_RECONCILIATION. Any invalid state raises RuntimeError.
    None of these statuses authorize a trade; the live heartbeat gate must
    independently pass for an opening.
    """
    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing; startup blocked.")
    funding = _amount(original_funding, "original funding")
    if funding <= 0:
        raise RuntimeError("Original funding must be positive.")

    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Database integrity failed; startup blocked.")
            if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise RuntimeError("Unsupported database version; startup blocked.")

            account = conn.execute(
                "SELECT balance, currency FROM account WHERE id=1"
            ).fetchone()
            if account is None or account[1] != "USDT":
                raise RuntimeError("Account invalid; startup blocked.")
            cash = _amount(account[0], "account balance")
            if cash <= 0:
                raise RuntimeError("Account balance must be positive.")

            expected = funding
            for (amount,) in conn.execute("SELECT amount FROM transactions"):
                expected += _amount(amount, "transaction amount")
            if cash != expected:
                raise RuntimeError("Ledger balance mismatch; startup blocked.")

            open_positions = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open'"
            ).fetchone()[0]
            if open_positions > 1:
                raise RuntimeError("More than one open position; startup blocked.")

            risk = conn.execute(
                "SELECT utc_day, starting_equity, locked FROM daily_risk WHERE id=1"
            ).fetchone()
            if risk is None:
                raise RuntimeError("Daily risk state missing; startup blocked.")
            try:
                saved_day = date.fromisoformat(risk[0])
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Invalid saved risk date; startup blocked.") from exc
            starting = _amount(risk[1], "starting equity")
            if starting <= 0 or risk[2] not in (0, 1):
                raise RuntimeError("Invalid daily risk state; startup blocked.")

    except sqlite3.DatabaseError as exc:
        raise RuntimeError("Unable to read SQLite account; startup blocked.") from exc

    today = datetime.now(timezone.utc).date()
    if saved_day > today:
        raise RuntimeError("Future-dated risk state; startup blocked.")
    if open_positions:
        status = "OPEN_POSITION_REQUIRES_RECONCILIATION"
    elif saved_day < today:
        status = "ROLLOVER_REQUIRED"
    elif risk[2]:
        status = "DAILY_LOCKED"
    else:
        status = "READY_FOR_MONITOR"

    return {
        "status": status,
        "cash_usdt": str(cash),
        "open_positions": open_positions,
        "saved_day": saved_day.isoformat(),
        "current_utc_day": today.isoformat(),
        "starting_equity_usdt": str(starting),
        "daily_locked": bool(risk[2]),
    }


def self_test():
    from .paper_database import initialize_paper_database
    from .sqlite_daily_risk import initialize_daily_risk_day

    print("PAPER STARTUP SAFETY TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "startup_test.sqlite3"
        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)
        assert inspect_paper_startup(db)["status"] == "READY_FOR_MONITOR"
        print("Valid current-day account: READY_FOR_MONITOR")

        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE daily_risk SET locked=1 WHERE id=1")
        assert inspect_paper_startup(db)["status"] == "DAILY_LOCKED"
        print("Locked current-day account: DAILY_LOCKED")

        yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE daily_risk SET utc_day=? WHERE id=1", (yesterday,))
        assert inspect_paper_startup(db)["status"] == "ROLLOVER_REQUIRED"
        print("Previous-day account: ROLLOVER_REQUIRED")

        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE account SET balance='999' WHERE id=1")
        try:
            inspect_paper_startup(db)
        except RuntimeError as exc:
            assert "balance mismatch" in str(exc)
            print("Unreconciled balance: BLOCKED")
        else:
            raise AssertionError("Unreconciled balance accepted")

        with sqlite3.connect(db) as conn:
            balance, saved_day = conn.execute(
                "SELECT balance, (SELECT utc_day FROM daily_risk WHERE id=1) FROM account WHERE id=1"
            ).fetchone()
        assert balance == "999" and saved_day == yesterday
        print("Startup check modified database: NO")
        print("Real orders: NONE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, help="Read-only inspection of an existing account DB")
    args = parser.parse_args()
    if args.db is None:
        self_test()
        return
    state = inspect_paper_startup(args.db)
    print("PAPER ACCOUNT STARTUP PREFLIGHT")
    print("-" * 40)
    for label, value in state.items():
        print(f"{label}: {value}")
    print("Actions taken: NONE (read-only)")
    print("New trades authorized: NO")


if __name__ == "__main__":
    main()
