import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from tempfile import TemporaryDirectory


class PaperPositionStore:
    """Single-process persistence prototype. Does not authorize trades."""

    def __init__(self, state_file):
        self.state_file = Path(state_file)

    @staticmethod
    def _positive(value, name):
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid {name}.") from exc
        if not number.is_finite() or number <= 0:
            raise ValueError(f"{name} must be positive and finite.")
        return number

    def initialize(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {"version": 1, "symbol": "BTC/USDT:USDT", "positions": []}
        try:
            with self.state_file.open("x") as file:
                json.dump(state, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError as exc:
            raise RuntimeError("Position store already exists. Reset blocked.") from exc

    def load(self):
        if not self.state_file.is_file():
            raise RuntimeError("Position store missing. Trading disabled.")
        try:
            state = json.loads(self.state_file.read_text())
            if type(state) is not dict or state.get("version") != 1:
                raise ValueError("Invalid position store version.")
            if state.get("symbol") != "BTC/USDT:USDT":
                raise ValueError("Unexpected market symbol.")
            positions = state.get("positions")
            if type(positions) is not list or len(positions) > 1:
                raise ValueError("Expected at most one open position.")
            for position in positions:
                if type(position) is not dict:
                    raise ValueError("Invalid position record.")
                if not isinstance(position.get("id"), str) or not position["id"].strip():
                    raise ValueError("Invalid position ID.")
                direction = position.get("direction")
                if direction not in ("long", "short"):
                    raise ValueError("Invalid position direction.")
                entry = self._positive(position.get("entry_price"), "entry_price")
                stop = self._positive(position.get("stop_price"), "stop_price")
                self._positive(position.get("quantity_btc"), "quantity_btc")
                if (direction == "long" and stop >= entry) or (
                    direction == "short" and stop <= entry
                ):
                    raise ValueError("Stop is on the wrong side of entry.")
            return state
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise RuntimeError("Position store invalid. Trading disabled.") from exc

    def record_open_position(self, position_id, direction, entry_price, stop_price, quantity_btc):
        # Persistence only. A separate risk manager must approve the trade first.
        state = self.load()
        if state["positions"]:
            raise RuntimeError("An open position already exists.")
        if not isinstance(position_id, str) or not position_id.strip():
            raise ValueError("Position ID is required.")
        entry = self._positive(entry_price, "entry_price")
        stop = self._positive(stop_price, "stop_price")
        quantity = self._positive(quantity_btc, "quantity_btc")
        if direction not in ("long", "short"):
            raise ValueError("Invalid direction.")
        if (direction == "long" and stop >= entry) or (
            direction == "short" and stop <= entry
        ):
            raise ValueError("Stop is on the wrong side of entry.")

        state["positions"].append({
            "id": position_id,
            "direction": direction,
            "entry_price": str(entry),
            "stop_price": str(stop),
            "quantity_btc": str(quantity),
        })
        temporary = self.state_file.with_name(self.state_file.name + ".tmp")
        with temporary.open("x") as file:
            json.dump(state, file, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.state_file)


def main():
    print("PERSISTENT PAPER POSITION TEST")
    print("-" * 40)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "positions.json"
        store = PaperPositionStore(path)
        store.initialize()
        store.record_open_position("position001", "long", "100000", "98000", "0.002")
        restarted = PaperPositionStore(path)
        record = restarted.load()["positions"][0]
        print("Restored position:", record["id"], record["direction"])
        print("BTC quantity:", record["quantity_btc"])
        try:
            restarted.record_open_position("position002", "short", "100000", "102000", "0.002")
        except RuntimeError:
            print("Second open position: BLOCKED")
        try:
            store.initialize()
        except RuntimeError:
            print("Accidental reset: BLOCKED")


if __name__ == "__main__":
    main()
