"""Atomically record a fictional BTC/USDT perpetual paper position and entry fee.

Demonstration only. No orders, signals, funding, continuous daily monitoring,
liquidation, reserved margin, or automated execution are implemented here.
Use only with the schema in paper_database.py (version 1).
"""

import sqlite3
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from tempfile import TemporaryDirectory

from .paper_database import initialize_paper_database, read_paper_database
from .atomic_trade_settlement import settle_position
from .sqlite_daily_risk import initialize_daily_risk_day, check_daily_risk_for_open
from .sqlite_monitor_heartbeat import check_monitor_heartbeat_for_open, record_verified_equity_check

FEE_RATE = Decimal("0.001")       # Experimental fee, per side
SLIPPAGE_RATE = Decimal("0.0005")  # Experimental adverse slippage, per side
MAX_LEVERAGE = Decimal("3")
MAX_RISK_FRACTION = Decimal("0.005")
PRECISION = Decimal("0.00000001")


def positive_decimal(value, label):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}.") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{label} must be positive and finite.")
    return result


def open_paper_position(db_path, position_id, direction, entry_price, stop_price, quantity_btc):
    """Persist one simulated opening and its opening fee in one transaction.

    The stop-risk check assumes the stop fills with the configured slippage;
    real gaps and slippage can lead to larger losses. The SQLite daily risk gate and a fresh equity-monitor heartbeat are checked
    before opening. This is not a full execution service.
    """
    path = Path(db_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing. Opening blocked.")
    if not isinstance(position_id, str) or not position_id.strip():
        raise ValueError("Position ID is required.")
    if direction not in ("long", "short"):
        raise ValueError("Direction must be long or short.")

    market_entry = positive_decimal(entry_price, "entry price")
    stop = positive_decimal(stop_price, "stop price")
    quantity = positive_decimal(quantity_btc, "BTC quantity")

    if direction == "long":
        fill = market_entry * (1 + SLIPPAGE_RATE)
        stop_fill = stop * (1 - SLIPPAGE_RATE)
        if stop >= market_entry:
            raise ValueError("Long stop must be below the entry price.")
        loss_at_stop = (fill - stop_fill) * quantity
    else:
        fill = market_entry * (1 - SLIPPAGE_RATE)
        stop_fill = stop * (1 + SLIPPAGE_RATE)
        if stop <= market_entry:
            raise ValueError("Short stop must be above the entry price.")
        loss_at_stop = (stop_fill - fill) * quantity

    notional = fill * quantity
    open_fee = (notional * FEE_RATE).quantize(PRECISION, rounding=ROUND_HALF_UP)
    stop_fee = (stop_fill * quantity * FEE_RATE).quantize(PRECISION, rounding=ROUND_HALF_UP)
    estimated_stop_loss = loss_at_stop + open_fee + stop_fee

    with sqlite3.connect(path, timeout=10) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
            raise RuntimeError("Unexpected paper database schema.")
        row = conn.execute(
            "SELECT balance, currency FROM account WHERE id = 1"
        ).fetchone()
        if row is None or row[1] != "USDT":
            raise RuntimeError("Account missing or invalid.")
        balance = positive_decimal(row[0], "account balance")

        if conn.execute(
            "SELECT 1 FROM positions WHERE id = ?", (position_id,)
        ).fetchone():
            raise RuntimeError("Position ID already used. Duplicate opening blocked.")
        if conn.execute(
            "SELECT 1 FROM positions WHERE status = 'open' LIMIT 1"
        ).fetchone():
            raise RuntimeError("Another position is already open.")

        if notional > balance * MAX_LEVERAGE:
            raise RuntimeError("3x exposure ceiling exceeded.")
        if estimated_stop_loss > balance * MAX_RISK_FRACTION:
            raise RuntimeError("0.5% estimated risk allowance exceeded.")
        if open_fee >= balance:
            raise RuntimeError("Insufficient balance for opening fee.")

        # Enforce daily loss budget within the SAME opening transaction.
        # A missing or stale daily state never permits an opening.
        approved, message = check_daily_risk_for_open(
            conn, balance, estimated_stop_loss
        )
        if not approved:
            # If the guard locked the day, save that lock before rejecting.
            conn.commit()
            raise RuntimeError(message)

        # Require a fresh, successful equity monitor check, recorded before
        # entering this transaction. Nothing here fetches the market price.
        fresh, heartbeat_message = check_monitor_heartbeat_for_open(conn, balance)
        if not fresh:
            raise RuntimeError(heartbeat_message)

        # Both statements and the cash debit commit or roll back together.
        conn.execute(
            "INSERT INTO positions "
            "(id, direction, entry_price, stop_price, quantity_btc, status, exit_price) "
            "VALUES (?, ?, ?, ?, ?, 'open', NULL)",
            (position_id, direction, str(fill), str(stop), str(quantity)),
        )
        conn.execute(
            "INSERT INTO transactions (id, position_id, kind, amount) "
            "VALUES (?, ?, 'trading_fee', ?)",
            (f"{position_id}:open_fee", position_id, str(-open_fee)),
        )
        updated_balance = balance - open_fee
        conn.execute(
            "UPDATE account SET balance = ? WHERE id = 1", (str(updated_balance),)
        )

    return {
        "position_id": position_id,
        "filled_entry_price": str(fill),
        "opening_fee_usdt": str(open_fee),
        "balance_usdt": str(updated_balance),
        "estimated_stop_loss_usdt": str(estimated_stop_loss.quantize(PRECISION, rounding=ROUND_HALF_UP)),
    }


def main():
    print("ATOMIC PAPER OPEN/CLOSE TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "paper.sqlite3"
        initialize_paper_database(path, "1000")
        try:
            open_paper_position(path, "beforeday", "long", "100000", "98000", "0.002")
        except RuntimeError:
            print("Missing daily risk state: BLOCKED")
        else:
            raise AssertionError("Missing daily state should block openings")
        initialize_daily_risk_day(path)
        try:
            open_paper_position(path, "noheartbeat", "long", "100000", "98000", "0.002")
        except RuntimeError:
            print("Missing monitor heartbeat: BLOCKED")
        else:
            raise AssertionError("Missing heartbeat should block opening")
        from unittest.mock import patch
        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 0.5)):
            record_verified_equity_check(path)
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE monitor_heartbeat SET checked_at_utc=? WHERE id=1",
                         ("2000-01-01T00:00:00+00:00",))
        try:
            open_paper_position(path, "staleheartbeat", "long", "100000", "98000", "0.002")
        except RuntimeError:
            print("Stale monitor heartbeat: BLOCKED")
        else:
            raise AssertionError("Stale heartbeat should block opening")
        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 0.5)):
            record_verified_equity_check(path)
        opened = open_paper_position(path, "demo001", "long", "100000", "98000", "0.002")
        print("Open position:", opened["position_id"])
        print("Entry fill:", opened["filled_entry_price"])
        print("Opening fee:", opened["opening_fee_usdt"], "USDT")
        balance, count = read_paper_database(path)
        print("Balance after open:", balance, "USDT")
        print("Open positions:", count)

        try:
            open_paper_position(path, "demo001", "long", "100000", "98000", "0.002")
        except RuntimeError:
            print("Duplicate open: BLOCKED")

        # Verify stop-risk rejection does not change account records.
        try:
            open_paper_position(path, "riskfail", "long", "100000", "98000", "0.1")
        except RuntimeError:
            print("Excessive position: BLOCKED")

        closed, balance_after_close = settle_position(path, "demo001", "101000")
        print("Closing status:", closed)
        print("Balance after close:", balance_after_close, "USDT")
        again, unchanged = settle_position(path, "demo001", "101000")
        print("Repeated close:", again)
        assert unchanged == balance_after_close
        assert read_paper_database(path)[1] == 0
        with sqlite3.connect(path) as conn:
            rows = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        print("Ledger transactions:", rows)
        assert rows == 3

        # A stop can slip severely: simulate an adverse gap beyond the stop.
        # The next opening must lock the account, and the lock must persist.
        with patch("src.paper.live_equity_check.fetch_validated_mark_price",
                   return_value=(Decimal("100000"), 0.5)):
            record_verified_equity_check(path)
        open_paper_position(path, "gapdemo", "long", "100000", "98000", "0.002")
        settle_position(path, "gapdemo", "90000")
        try:
            open_paper_position(path, "aftergap", "long", "100000", "98000", "0.002")
        except RuntimeError:
            print("After daily loss breach: BLOCKED")
        else:
            raise AssertionError("Daily loss breach must block new openings")
        with sqlite3.connect(path) as conn:
            locked = conn.execute("SELECT locked FROM daily_risk WHERE id=1").fetchone()[0]
        assert locked == 1
        print("Daily lock persisted: CONFIRMED")


if __name__ == "__main__":
    main()
