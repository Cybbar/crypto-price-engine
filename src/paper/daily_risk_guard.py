
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path

MAX_DAILY_LOSS = 0.015  # 1.5%


@dataclass
class DailyRiskGuard:
    state_path: str | Path | None = None
    current_day: date | None = None
    starting_equity: float = 0.0
    trading_locked: bool = False

    def __post_init__(self):
        if self.state_path is None:
            return

        self.state_path = Path(self.state_path)

        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())

                self.current_day = date.fromisoformat(
                    data["current_day"]
                )
                self.starting_equity = float(
                    data["starting_equity"]
                )
                self.trading_locked = data["trading_locked"]

                if (
                    not isfinite(self.starting_equity)
                    or self.starting_equity <= 0
                    or type(self.trading_locked) is not bool
                ):
                    raise ValueError("Invalid risk state")

            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    "Invalid saved risk state. Trading must remain disabled."
                ) from exc

    def save_state(self):
        if self.state_path is None:
            return

        self.state_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = {
            "current_day": self.current_day.isoformat(),
            "starting_equity": self.starting_equity,
            "trading_locked": self.trading_locked,
        }

        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(self.state_path)

    def can_open_trade(self, account_equity, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        if timestamp.tzinfo is None:
            raise ValueError("Timestamp must have a timezone.")

        today = timestamp.astimezone(timezone.utc).date()

        if not isfinite(account_equity) or account_equity <= 0:
            if self.current_day == today:
                self.trading_locked = True
                self.save_state()
            return False

        if self.current_day != today:
            self.current_day = today
            self.starting_equity = account_equity
            self.trading_locked = False
            self.save_state()

        daily_loss = (
            self.starting_equity - account_equity
        ) / self.starting_equity

        if daily_loss >= MAX_DAILY_LOSS:
            self.trading_locked = True

        self.save_state()

        return not self.trading_locked


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        state_file = Path(directory) / "daily_risk.json"

        guard = DailyRiskGuard(state_path=state_file)

        print("PERSISTENT DAILY RISK TEST")
        print("-" * 40)

        for equity in [1000, 995, 990, 985]:
            allowed = guard.can_open_trade(equity)
            print(
                f"Equity: ${equity:.2f} | "
                f"{'ALLOWED' if allowed else 'BLOCKED'}"
            )

        # Simulate restarting the bot.
        restarted_guard = DailyRiskGuard(
            state_path=state_file
        )

        allowed = restarted_guard.can_open_trade(990)

        print(
            f"After restart, equity $990.00 | "
            f"{'ALLOWED' if allowed else 'BLOCKED'}"
        )
