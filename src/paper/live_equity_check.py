
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import ccxt

from .paper_database import initialize_paper_database
from .sqlite_daily_risk import initialize_daily_risk_day
from .atomic_trade_open import open_paper_position
from .sqlite_equity_monitor import monitor_account_equity


MAX_PRICE_AGE_SECONDS = 60


def fetch_validated_mark_price():
    exchange = ccxt.binanceusdm({
        "enableRateLimit": True,
        "timeout": 10000,
    })

    try:
        data = exchange.fapiPublicGetPremiumIndex({
            "symbol": "BTCUSDT"
        })

        mark_price = Decimal(str(data["markPrice"]))

        exchange_time = datetime.fromtimestamp(
            int(data["time"]) / 1000,
            tz=timezone.utc,
        )

        age = (
            datetime.now(timezone.utc) - exchange_time
        ).total_seconds()

        if (
            not mark_price.is_finite()
            or mark_price <= 0
            or not 0 <= age <= MAX_PRICE_AGE_SECONDS
        ):
            raise RuntimeError(
                "Invalid or stale BTC market data."
            )

        return mark_price, age

    finally:
        exchange.close()


def check_equity_with_live_price(db_path):
    mark_price, age = fetch_validated_mark_price()

    snapshot = monitor_account_equity(
        db_path,
        mark_price=str(mark_price),
    )

    return mark_price, age, snapshot


def main():
    print("LIVE BTC PAPER EQUITY TEST")
    print("-" * 40)

    # Temporary fictional account.
    # No permanent account is modified.
    with TemporaryDirectory() as directory:
        db = Path(directory) / "paper.sqlite3"

        initialize_paper_database(db, "1000")
        initialize_daily_risk_day(db)

        # Retrieve the real price for a fictional entry.
        mark_price, _ = fetch_validated_mark_price()

        # Simulate one small position.
        # This function writes only to our temporary SQLite DB.
        open_paper_position(
            db_path=db,
            position_id="demo_live001",
            direction="long",
            entry_price=str(mark_price),
            stop_price=str(mark_price * Decimal("0.99")),
            quantity_btc="0.001",
        )

        # Retrieve a fresh price and calculate equity.
        price, age, account = check_equity_with_live_price(db)

        print(f"BTC mark price: {price:.2f} USDT")
        print(f"Data age: {age:.2f} seconds")
        print("Market data: VALID")

        print(
            f"Cash balance: "
            f"{account['cash_balance_usdt']} USDT"
        )

        print(
            f"Unrealized P&L: "
            f"{account['unrealized_pnl_usdt']} USDT"
        )

        print(
            f"Account equity: "
            f"{account['equity_usdt']} USDT"
        )

        print(
            "Daily risk status:",
            "BLOCKED" if account["trading_locked"]
            else "ALLOWED"
        )

        print("Real exchange orders: NONE")


if __name__ == "__main__":
    main()
