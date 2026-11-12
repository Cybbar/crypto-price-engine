"""Initialize the single SQLite authority for a simulated paper account.

No orders, signals or trade approvals. Historical JSON files are NOT imported.
Do not use this prototype for unattended execution.
"""

import os
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory


def initialize_paper_database(database_path, starting_balance="1000"):
    path = Path(database_path)
    try:
        balance = Decimal(str(starting_balance))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Invalid starting balance") from exc
    if not balance.is_finite() or balance <= 0:
        raise ValueError("Starting balance must be positive and finite")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Fail if a database already exists; never reset an account silently.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError as exc:
        raise RuntimeError("Paper database already exists. Reset blocked.") from exc

    # If initialization fails the file is deliberately left behind so a
    # subsequent attempt cannot silently replace a potentially damaged DB.
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            CREATE TABLE account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                balance TEXT NOT NULL,
                currency TEXT NOT NULL CHECK (currency = 'USDT')
            )
        """)
        conn.execute("""
            CREATE TABLE positions (
                id TEXT PRIMARY KEY,
                direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
                entry_price TEXT NOT NULL,
                stop_price TEXT NOT NULL,
                quantity_btc TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('open', 'closed')),
                exit_price TEXT
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX only_one_open_position
            ON positions(status) WHERE status = 'open'
        """)
        conn.execute("""
            CREATE TABLE transactions (
                id TEXT PRIMARY KEY,
                position_id TEXT REFERENCES positions(id),
                kind TEXT NOT NULL,
                amount TEXT NOT NULL
            )
        """)
        conn.execute(
            "INSERT INTO account (id, balance, currency) VALUES (1, ?, 'USDT')",
            (str(balance),),
        )
        conn.execute("PRAGMA user_version = 1")


def read_paper_database(database_path):
    path = Path(database_path)
    if not path.is_file():
        raise RuntimeError("Paper database missing. Trading disabled.")
    try:
        # Explicit read-only mode prevents accidentally creating a missing DB.
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise RuntimeError("Database integrity check failed")
            if conn.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise RuntimeError("Unexpected database version")
            row = conn.execute(
                "SELECT balance, currency FROM account WHERE id=1"
            ).fetchone()
            if row is None or row[1] != "USDT":
                raise RuntimeError("Invalid account record")
            balance = Decimal(row[0])
            if not balance.is_finite() or balance < 0:
                raise RuntimeError("Invalid account balance")
            open_positions = conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open'"
            ).fetchone()[0]
            return balance, open_positions
    except (sqlite3.DatabaseError, InvalidOperation, ValueError) as exc:
        raise RuntimeError("Paper database invalid. Trading disabled.") from exc


def main():
    print("SQLITE PAPER ACCOUNT TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "paper.sqlite3"
        initialize_paper_database(path, "1000")
        balance, open_count = read_paper_database(path)
        print(f"Initial balance: {balance} USDT")
        print(f"Open positions: {open_count}")
        restarted_balance, _ = read_paper_database(path)
        print(f"After restart: {restarted_balance} USDT")
        try:
            initialize_paper_database(path, "2000")
        except RuntimeError:
            print("Accidental reset: BLOCKED")
        try:
            read_paper_database(Path(directory) / "missing.sqlite3")
        except RuntimeError:
            print("Missing database: BLOCKED")


if __name__ == "__main__":
    main()
