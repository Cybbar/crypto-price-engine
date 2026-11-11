# BTC/USDT Perpetual Futures Research & Trading Bot

A Python-based research project focused on developing a systematic BTC/USDT perpetual futures trading bot using market data, technical indicators, market-regime detection, and rule-based trading strategies.

The project explores how market conditions can be analysed to identify potential trading opportunities while keeping risk management separate from strategy decisions.

## Project Objectives

- Collect and validate historical BTC/USDT perpetual futures market data.
- Engineer technical and market-condition features.
- Classify market regimes, including trending, ranging, and transitioning markets.
- Evaluate directional trading signals.
- Research ATR-based stop-loss and take-profit configurations.
- Develop a realistic backtesting framework.
- Build an independent risk-management engine.

## Technology Stack

- Python
- Pandas
- NumPy
- CCXT
- Scikit-learn (for machine-learning experiments)
- Git and GitHub

## Project Structure

```text
crypto-price-engine/
│
├── src/
│   ├── data/
│   │   ├── fetch_futures_data.py
│   │   ├── build_futures_features.py
│   │   ├── detect_futures_regime.py
│   │   ├── evaluate_futures_regimes.py
│   │   ├── evaluate_futures_regime_stability.py
│   │   ├── evaluate_futures_excursions.py
│   │   ├── evaluate_futures_atr_excursions.py
│   │   ├── evaluate_futures_entry_signals.py
│   │   ├── compare_futures_entry_baseline.py
│   │   └── backtest_futures_exit_rules.py
│   │
│   └── ...
│
├── data/          # Local market data (not tracked)
├── reports/       # Research outputs (not tracked)
├── .gitignore
└── README.md
```

## Current Research

The project has completed several stages of historical futures-market research:

- Historical 1-hour BTC/USDT perpetual futures data collection.
- Funding-rate collection and validation.
- Technical feature engineering, including EMA, ATR, ADX, and volatility measurements.
- Rule-based market-regime classification.
- Market-regime stability analysis across historical periods.
- Maximum favourable and adverse excursion analysis.
- ATR-normalised excursion analysis.
- Preliminary entry-signal and exit-rule evaluations.

### Preliminary Findings

Initial experiments have not established a reliable profitable trading edge after estimated transaction costs.

These findings are important because the goal is to develop a strategy based on evidence rather than assume that technical indicators or machine learning automatically produce profitable trades.

## Planned Architecture

The intended trading system separates signal generation from risk management.

1. **Market data:** Collect and validate futures market data.
2. **Feature engineering:** Calculate technical and volatility features.
3. **Market-regime detection:** Identify trending, ranging, and high-volatility conditions.
4. **Signal generation:** Use deterministic trading rules for entry and exit decisions.
5. **Risk management:** Independently control position sizing, stop-loss limits, and account-level drawdown.
6. **Backtesting:** Evaluate performance using chronological, non-overlapping tests.
7. **Paper trading:** Validate the system before considering live execution.

Machine learning is intended to assist with market-regime classification, not independently authorize trades or override risk controls.

## Research and Development Status

**Status: In active development**

The current repository contains research scripts and historical evaluation tools. The complete trading engine, independent risk controls, and paper-trading workflow are still under development.

This project is not financial advice and does not guarantee trading profits. No live-trading performance is claimed.

## Data and Security

Historical market data and generated research reports are stored locally and excluded from Git tracking.

API credentials and other sensitive information should never be committed to this repository.

## Author

**Cybbar**

GitHub: [@Cybbar](https://github.com/Cybbar)

---

*Built through practical experimentation, historical data analysis, and iterative Python development.*