
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .daily_risk_guard import DailyRiskGuard


ROOT = Path(__file__).resolve().parents[2]

STATE_FILE = (
    ROOT / "data" / "paper" / "state" / "daily_risk.json"
)


def load_runtime_guard(state_path=STATE_FILE):
    state_path = Path(state_path)

    # Never silently create replacement risk state.
    if not state_path.is_file():
        raise RuntimeError(
            "Risk state missing. New trades are disabled."
        )

    try:
        guard = DailyRiskGuard(state_path=state_path)
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            "Risk state cannot be loaded. Trading disabled."
        ) from exc

    today = datetime.now(timezone.utc).date()

    # A new day requires an explicit equity reconciliation.
    if guard.current_day != today:
        raise RuntimeError(
            "Risk state belongs to another UTC day. "
            "Trading disabled until reconciliation."
        )

    return guard


def main():
    print("RUNTIME RISK SAFETY TEST")
    print("-" * 40)

    with TemporaryDirectory() as directory:
        state_file = Path(directory) / "daily_risk.json"

        # Test 1: Missing state must block trading.
        try:
            load_runtime_guard(state_file)
        except RuntimeError:
            print("Missing state: BLOCKED")

        # Initialize test state.
        guard = DailyRiskGuard(state_path=state_file)
        guard.can_open_trade(1000)

        # Test 2: Valid saved state.
        restored = load_runtime_guard(state_file)

        print(
            "Valid state:",
            "LOADED" if not restored.trading_locked else "LOCKED"
        )

        # Trigger the daily loss limit.
        restored.can_open_trade(985)

        # Test 3: Lock survives restart.
        restarted = load_runtime_guard(state_file)

        print(
            "After restart:",
            "BLOCKED" if restarted.trading_locked else "ALLOWED"
        )


if __name__ == "__main__":
    main()
