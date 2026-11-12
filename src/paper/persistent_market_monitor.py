"""Read-only BTC perpetual paper-equity monitoring loop.

Records a verified heartbeat only after a successful Binance mark-price fetch
and SQLite equity evaluation. This does NOT place orders, enforce stop orders,
or initialize the daily risk state. For a single-process paper prototype only.
"""

import argparse
import sqlite3
import time
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .sqlite_monitor_heartbeat import record_verified_equity_check


def revoke_heartbeat(database_path):
    """Fail closed after a detected monitor error; no auto-created database."""
    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Database missing; cannot revoke heartbeat.")

    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected SQLite schema; cannot revoke heartbeat.")
        # If the table is missing, trade opening already fails closed; do not
        # silently treat an unexpected schema as a successful revocation.
        conn.execute("DELETE FROM monitor_heartbeat WHERE id = 1")


def run_monitor(database_path, cycles=3, interval_seconds=5):
    """Run finite monitoring cycles; 0 cycles means run until interrupted.

    Each call fetches a fresh Binance mark and records the equity heartbeat.
    Any failed check removes the previous heartbeat and stops this process.
    An abrupt process crash cannot revoke it immediately: opening remains
    possible until the existing 15-second heartbeat expires.
    """
    if not isinstance(cycles, int) or cycles < 0:
        raise ValueError("cycles must be a nonnegative integer")
    if not 0 <= interval_seconds <= 3600:
        raise ValueError("interval_seconds must be between 0 and 3600")

    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Existing paper database required; monitoring stopped.")

    count = 0
    while cycles == 0 or count < cycles:
        try:
            snapshot = record_verified_equity_check(path)
        except Exception as exc:
            try:
                revoke_heartbeat(path)
            except Exception as revoke_exc:
                raise RuntimeError(
                    "Monitoring failed and heartbeat revocation failed. "
                    "Do not authorize trades; any old heartbeat expires within 15 seconds."
                ) from revoke_exc
            raise RuntimeError(
                "Monitoring failed. Previous heartbeat revoked; no new paper opens."
            ) from exc

        count += 1
        status = "BLOCKED" if snapshot["trading_locked"] else "ALLOWED"
        print(
            f"Check {count}: "
            f"mark={Decimal(snapshot['mark_price']):.2f} USDT | "
            f"age={snapshot['mark_age_seconds']:.2f}s | "
            f"equity={snapshot['equity_usdt']} USDT | "
            f"daily risk={status}",
            flush=True,
        )
        if cycles == 0 or count < cycles:
            time.sleep(interval_seconds)
    return count


def self_test():
    """No network or persistent files; test fresh checks, breach, and failure."""
    from .paper_database import initialize_paper_database
    from .sqlite_daily_risk import initialize_daily_risk_day
    from .sqlite_monitor_heartbeat import check_monitor_heartbeat_for_open
    from .atomic_trade_open import open_paper_position

    print("PAPER MARKET MONITOR SELF-TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"
        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)

        # Initial approved heartbeat is necessary to open a paper position.
        with patch(
            "src.paper.live_equity_check.fetch_validated_mark_price",
            return_value=(Decimal("100000"), 0.3),
        ):
            record_verified_equity_check(db)
        with sqlite3.connect(db) as conn:
            approved, _ = check_monitor_heartbeat_for_open(conn, "1000")
        assert approved
        print("Initial verified heartbeat: APPROVED")

        open_paper_position(
            db, "monitor001", "long", "100000", "99000", "0.001"
        )
        print("Fictional open position: RECORDED")

        with patch(
            "src.paper.live_equity_check.fetch_validated_mark_price",
            side_effect=[
                (Decimal("99000"), 0.2),
                (Decimal("80000"), 0.2),
            ],
        ):
            assert run_monitor(db, cycles=2, interval_seconds=0) == 2
        with sqlite3.connect(db) as conn:
            locked = conn.execute(
                "SELECT locked FROM daily_risk WHERE id=1"
            ).fetchone()[0]
        assert locked == 1
        print("Unrealized loss breached daily limit: LOCKED")

        with patch(
            "src.paper.live_equity_check.fetch_validated_mark_price",
            side_effect=RuntimeError("Simulated Binance outage"),
        ):
            try:
                run_monitor(db, cycles=1, interval_seconds=0)
            except RuntimeError as exc:
                assert "heartbeat revoked" in str(exc)
            else:
                raise AssertionError("Network failure was not detected")
        print("Failed market-data fetch: MONITOR STOPPED")

        with sqlite3.connect(db) as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM monitor_heartbeat WHERE id=1"
            ).fetchone()[0]
            locked_after_restart = conn.execute(
                "SELECT locked FROM daily_risk WHERE id=1"
            ).fetchone()[0]
        assert remaining == 0
        assert locked_after_restart == 1
        print("Previous heartbeat: REVOKED")
        print("Daily loss lock: PERSISTED")
        print("Real orders: NONE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, help="Existing SQLite paper database; omit to self-test"
    )
    parser.add_argument(
        "--cycles", type=int, default=3,
        help="Number of checks, or 0 for continuous foreground checking (default 3)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0,
        help="Seconds between successful checks (default 5)",
    )
    args = parser.parse_args()
    if args.db is None:
        self_test()
    else:
        run_monitor(args.db, args.cycles, args.interval)


if __name__ == "__main__":
    main()
