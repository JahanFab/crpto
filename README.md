# Crypto Arbitrage Scanner

Detects cross-exchange pricing inefficiencies across 6 major cryptocurrency exchanges in real time (simulated) and models whether each opportunity is actually executable after accounting for fees, latency, and liquidity.

---

## Overview

Cryptocurrency prices across exchanges rarely move in perfect lockstep. A price dislocation — where exchange A's ask is below exchange B's bid for the same asset — is a potential arbitrage. The challenge: by the time both legs are submitted and confirmed, the spread may have closed. This project scans for these opportunities and scores each one for execution feasibility.

---

## How It Works

### 1. Exchange Model

| Exchange | Taker Fee | Latency |
|----------|-----------|---------|
| Binance | 0.10% | 5ms |
| OKX | 0.10% | 8ms |
| Bybit | 0.10% | 10ms |
| Coinbase | 0.60% | 15ms |
| Kraken | 0.40% | 20ms |
| Bitfinex | 0.20% | 25ms |

### 2. Price Simulation
- 200 snapshots across BTC, ETH, SOL, BNB
- Each exchange price lags the "true" mid with AR(1) mean reversion and independent noise
- Spreads are wider on slower/smaller exchanges
- Liquidity depth scales with exchange size

### 3. Arbitrage Detection
For each snapshot and each buy-exchange → sell-exchange pair:

```
gross_spread_bps = (sell_bid - buy_ask) / buy_ask × 10,000
net_spread_bps   = gross_spread_bps - buy_taker_fee_bps - sell_taker_fee_bps
```

Opportunity flagged if `net_spread_bps > 3.0`.

### 4. Feasibility Model

Spread erodes during execution latency (round-trip = buy_latency + sell_latency):

```
spread_erosion = latency_ms × vol_per_ms
P(profitable) = Φ(expected_net_spread / σ_erosion)
expected_pnl  = P(profitable) × net_spread × order_size
```

### 5. Feasibility Score

Composite score balancing three factors:

```
feasibility = 0.4 × exp(-latency/50) + 0.3 × (size/max_size) + 0.3 × min(net_bps/50, 1)
```

---

## Results (200 snapshots, 4 symbols, 6 exchanges)

- **3,000+ raw opportunities** detected
- **Binance ↔ OKX** pairs dominate by expected PnL (low fees, low latency)
- **Bitfinex** pairs show P(profitable) < 30% even on wide gross spreads due to 25ms+ latency
- Average net spread: ~8–15 bps depending on symbol and exchange pair

---

## Output Plots

| File | Description |
|------|-------------|
| `crypto_arbitrage_dashboard.png` | 7-panel dashboard: opportunity count over time, spread distribution, exchange-pair heatmap, PnL vs feasibility scatter, latency vs profitability, top pairs by expected PnL, symbol breakdown |

---

## Usage

```bash
pip install numpy pandas matplotlib scipy
python crypto_arbitrage_scanner.py
```

---

## Extending to Live Data

To connect real exchange feeds, replace `simulate_exchange_prices()` with WebSocket orderbook subscriptions using `ccxt` or `websockets`:

```python
import ccxt
exchange = ccxt.binance()
orderbook = exchange.fetch_order_book('BTC/USDT')
```

---

## Dependencies

```
numpy
pandas
matplotlib
scipy
```
