import os
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory


class PaperAccountLedger:

    def __init__(self, state_file):
        self.state_file = Path(state_file)

    def initialize(self, starting_balance):
        try:
            balance = Decimal(str(starting_balance))
        except InvalidOperation as exc:
            raise ValueError("Invalid starting balance.") from exc

        if not balance.is_finite() or balance <= 0:
            raise ValueError("Starting balance must be positive.")

        if balance != balance.quantize(Decimal("0.01")):
            raise ValueError("Use at most two decimal places.")

        self.state_file.parent.mkdir(
            parents=True, exist_ok=True
        )

        state = {
            "version": 1,
            "currency": "USDT",
            "starting_balance": str(balance),
            "cash_balance": str(balance),
            "transactions": [],
        }

        # Exclusive creation prevents accidental resets.
        try:
            with self.state_file.open("x") as file:
                json.dump(state, file, indent=2)
                file.flush()
        except FileExistsError as exc:
            raise RuntimeError(
                "Account already initialized. Reset blocked."
            ) from exc

    def load(self):
        if not self.state_file.is_file():
            raise RuntimeError(
                "Account ledger missing. Trading disabled."
            )

        try:
            state = json.loads(self.state_file.read_text())

            if state["version"] != 1:
                raise ValueError("Unsupported ledger version.")

            if state["currency"] != "USDT":
                raise ValueError("Invalid account currency.")

            balance = Decimal(state["cash_balance"])
            starting = Decimal(state["starting_balance"])

            if (
                not balance.is_finite()
                or not starting.is_finite()
                or balance < 0
                or starting <= 0
                or not isinstance(state["transactions"], list)
            ):
                raise ValueError("Invalid account state.")

            return state

        except (
            ValueError, KeyError, TypeError,
            InvalidOperation, OSError
        ) as exc:
            raise RuntimeError(
                "Account ledger invalid. Trading disabled."
            ) from exc

    def record_transaction(self, transaction_id, kind, amount):
        allowed_types = {"realized_pnl", "trading_fee", "funding"}

        if not isinstance(transaction_id, str) or not transaction_id.strip():
            raise ValueError("Transaction ID is required.")

        if kind not in allowed_types:
            raise ValueError("Unsupported transaction type.")

        try:
            amount = Decimal(str(amount))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("Invalid transaction amount.") from exc

        if not amount.is_finite() or amount == 0:
            raise ValueError("Amount must be finite and nonzero.")

        if amount != amount.quantize(Decimal("0.00000001")):
            raise ValueError("Maximum eight decimal places.")

        if kind == "trading_fee" and amount >= 0:
            raise ValueError("Trading fees must be negative.")

        state = self.load()

        # Check the existing transaction history before updating.
        try:
            expected_balance = Decimal(state["starting_balance"])
            existing_ids = set()

            for transaction in state["transactions"]:
                tx_id = transaction["id"]
                if not isinstance(tx_id, str) or not tx_id or tx_id in existing_ids:
                    raise ValueError("Invalid or duplicate transaction ID.")
                existing_ids.add(tx_id)
                expected_balance += Decimal(transaction["amount"])

            if expected_balance != Decimal(state["cash_balance"]):
                raise ValueError("Ledger balance mismatch.")
        except (KeyError, TypeError, InvalidOperation, ValueError) as exc:
            raise RuntimeError("Ledger reconciliation failed.") from exc

        if transaction_id in existing_ids:
            raise RuntimeError("Duplicate transaction blocked.")

        new_balance = expected_balance + amount

        if new_balance < 0:
            raise RuntimeError("Insufficient paper account balance.")

        state["transactions"].append({
            "id": transaction_id,
            "type": kind,
            "amount": str(amount),
        })
        state["cash_balance"] = str(new_balance)

        # Atomic replacement for this single-process prototype.
        temporary = self.state_file.with_name(
            self.state_file.name + ".tmp"
        )

        with temporary.open("x") as file:
            json.dump(state, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        temporary.replace(self.state_file)

        return new_balance


def main():
    print("PAPER ACCOUNT LEDGER TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "account.json"

        ledger = PaperAccountLedger(path)
        ledger.initialize(1000)

        state = ledger.load()
        print(f"Initial balance: ${state['cash_balance']}")

        # Simulate restarting the bot.
        restarted = PaperAccountLedger(path)
        state = restarted.load()

        print(f"Restored balance: ${state['cash_balance']}")

        # Attempt to reset the account.
        try:
            restarted.initialize(2000)
        except RuntimeError:
            print("Duplicate initialization: BLOCKED")

        # Attempt to load a missing account.
        missing = PaperAccountLedger(
            Path(directory) / "missing.json"
        )

        try:
            missing.load()
        except RuntimeError:
            print("Missing account ledger: BLOCKED")


if __name__ == "__main__":
    main()
