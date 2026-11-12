"""Transactional paper-trade settlement prototype.

This is an isolated SQLite demonstration, NOT connected to JSON runtime state.
Opening fees and funding are not included in the sample balances.
"""

import sqlite3
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from .trade_close import calculate_trade_close
except ImportError:
    # Allows stand-alone testing in this environment; in project use package import.
    from trade_close import calculate_trade_close


def settle_position(database_path, position_id, exit_price):
    """Settle once; position, cash and transaction log commit atomically."""
    database_path = Path(database_path)
    if not database_path.is_file():
        raise RuntimeError("Settlement database missing. Operation blocked.")
    if not isinstance(position_id, str) or not position_id.strip():
        raise ValueError("Position ID is required.")

    with sqlite3.connect(database_path, timeout=10) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.row_factory = sqlite3.Row
        position = connection.execute(
            "SELECT id, direction, entry_price, quantity_btc, status "
            "FROM positions WHERE id = ?", (position_id,)
        ).fetchone()
        if position is None:
            raise RuntimeError("Position not found; no settlement recorded.")
        if position["status"] == "closed":
            cash = connection.execute(
                "SELECT balance FROM account WHERE id = 1"
            ).fetchone()
            if cash is None:
                raise RuntimeError("Account record missing.")
            return "ALREADY_CLOSED", cash["balance"]
        if position["status"] != "open":
            raise RuntimeError("Invalid position status.")

        cash = connection.execute(
            "SELECT balance FROM account WHERE id = 1"
        ).fetchone()
        if cash is None:
            raise RuntimeError("Account record missing.")

        result = calculate_trade_close(dict(position), exit_price)
        pnl = Decimal(result["realized_pnl_usdt"])
        fee = Decimal(result["closing_fee_usdt"])
        new_balance = Decimal(cash["balance"]) + pnl - fee

        # Every operation below is in the same SQLite transaction.
        connection.execute(
            "INSERT INTO transactions (id, position_id, kind, amount) "
            "VALUES (?, ?, 'realized_pnl', ?)",
            (f"{position_id}:pnl", position_id, str(pnl)),
        )
        connection.execute(
            "INSERT INTO transactions (id, position_id, kind, amount) "
            "VALUES (?, ?, 'trading_fee', ?)",
            (f"{position_id}:close_fee", position_id, str(-fee)),
        )
        connection.execute(
            "UPDATE account SET balance = ? WHERE id = 1",
            (str(new_balance),),
        )
        connection.execute(
            "UPDATE positions SET status = 'closed', exit_price = ? "
            "WHERE id = ? AND status = 'open'",
            (result["filled_exit_price"], position_id),
        )
        # Context-manager commits on success, rolls back on exceptions.
        return "SETTLED", str(new_balance)


def main():
    print("ATOMIC PAPER SETTLEMENT TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        db_path = Path(directory) / "paper_settlement.sqlite3"
        with sqlite3.connect(db_path) as connection:
            connection.executescript("""
                CREATE TABLE account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    balance TEXT NOT NULL
                );
                CREATE TABLE positions (
                    id TEXT PRIMARY KEY,
                    direction TEXT NOT NULL CHECK (direction IN ('long','short')),
                    entry_price TEXT NOT NULL,
                    quantity_btc TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('open','closed')),
                    exit_price TEXT
                );
                CREATE UNIQUE INDEX only_one_open_position
                    ON positions(status) WHERE status = 'open';
                CREATE TABLE transactions (
                    id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL REFERENCES positions(id),
                    kind TEXT NOT NULL,
                    amount TEXT NOT NULL
                );
                INSERT INTO account(id, balance) VALUES(1, '1000');
                INSERT INTO positions(id, direction, entry_price, quantity_btc, status)
                    VALUES('long001', 'long', '100000', '0.002', 'open');
            """)

        status, balance = settle_position(db_path, "long001", "101000")
        print(f"Long close: {status} | Balance: {balance} USDT")
        status, balance = settle_position(db_path, "long001", "101000")
        print(f"Retry long close: {status} | Balance: {balance} USDT")

        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "INSERT INTO positions(id, direction, entry_price, quantity_btc, status) "
                "VALUES (?, ?, ?, ?, 'open')",
                ("short001", "short", "100000", "0.002"),
            )
        status, balance = settle_position(db_path, "short001", "99000")
        print(f"Short close: {status} | Balance: {balance} USDT")
        with sqlite3.connect(db_path) as connection:
            count = connection.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        print(f"Transaction entries: {count}")
        print("Note: Demo balances exclude opening fees and funding.")


if __name__ == "__main__":
    main()
