"""
Crypto Arbitrage Scanner
Detects cross-exchange pricing inefficiencies in real time (simulated) and
models execution feasibility (latency, fees, slippage, capital constraints).
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time


# Exchange & Fee Model 

@dataclass
class Exchange:
    name: str
    taker_fee: float        # fraction of trade value
    maker_fee: float
    withdrawal_fee: float   # flat fee per withdrawal in USD
    latency_ms: float       # typical order execution latency
    min_order_usd: float    # minimum order size



EXCHANGES = {
    "Binance":   Exchange("Binance",   taker_fee=0.001, maker_fee=0.001, withdrawal_fee=0.5,  latency_ms=5,   min_order_usd=10),
    "Coinbase":  Exchange("Coinbase",  taker_fee=0.006, maker_fee=0.004, withdrawal_fee=1.0,  latency_ms=15,  min_order_usd=1),
    "Kraken":    Exchange("Kraken",    taker_fee=0.004, maker_fee=0.002, withdrawal_fee=0.75, latency_ms=20,  min_order_usd=10),
    "OKX":       Exchange("OKX",       taker_fee=0.001, maker_fee=0.0008,withdrawal_fee=0.3,  latency_ms=8,   min_order_usd=5),
    "Bybit":     Exchange("Bybit",     taker_fee=0.001, maker_fee=0.001, withdrawal_fee=0.5,  latency_ms=10,  min_order_usd=5),
    "Bitfinex":  Exchange("Bitfinex",  taker_fee=0.002, maker_fee=0.001, withdrawal_fee=2.0,  latency_ms=25,  min_order_usd=25),
}


# Price Simulation 

@dataclass
class Orderbook:
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_size: float   # USD available at bid
    ask_size: float   # USD available at ask
    timestamp: float  # unix timestamp



def simulate_exchange_prices(
    symbols: List[str],
    true_prices: Dict[str, float],
    seed: int = 42,
    n_snapshots: int = 200,
) -> List[Dict[str, Orderbook]]:
    
    """
    Simulate correlated price streams across exchanges with:
    - Small persistent dislocations between exchanges (lagged price discovery)
    - Random bid-ask spreads (wider on smaller exchanges)
    - Liquidity depth proportional to exchange size

    """
    rng = np.random.default_rng(seed)
    snapshots = []

    # Exchange-specific spread factors and price drift persistence
    spread_factor = {
        "Binance": 1.0, "Coinbase": 1.5, "Kraken": 1.3,
        "OKX": 1.1, "Bybit": 1.2, "Bitfinex": 2.0,
    }
    size_factor = {
        "Binance": 3.0, "Coinbase": 2.0, "Kraken": 1.2,
        "OKX": 2.5, "Bybit": 1.5, "Bitfinex": 0.8,
    }

    prices = {sym: {ex: true_prices[sym] for ex in EXCHANGES} for sym in symbols}

    for snap_i in range(n_snapshots):
        snapshot = {}
        for sym in symbols:
            # Update true price (random walk)
            true_prices[sym] *= np.exp(rng.normal(0, 0.0005))


            for ex_name in EXCHANGES:
                # Exchange price lags true price with mean reversion + noise
                lag = rng.normal(0, true_prices[sym] * 0.0008)
                mean_rev = 0.3 * (true_prices[sym] - prices[sym][ex_name])
                prices[sym][ex_name] = prices[sym][ex_name] + mean_rev + lag


                mid = prices[sym][ex_name]
                base_spread = mid * 0.0005 * spread_factor[ex_name]
                spread_noise = abs(rng.normal(0, base_spread * 0.3))
                half_spread = base_spread / 2 + spread_noise

                bid = mid - half_spread
                ask = mid + half_spread
                base_size = 50_000 * size_factor[ex_name]
                bid_size = base_size * rng.lognormal(0, 0.5)
                ask_size = base_size * rng.lognormal(0, 0.5)


                key = f"{sym}_{ex_name}"
                snapshot[key] = Orderbook(
                    exchange=ex_name,
                    symbol=sym,
                    bid=bid,
                    ask=ask,
                    bid_size=bid_size,
                    ask_size=ask_size,
                    timestamp=time.time() + snap_i * 0.1,
                )
        snapshots.append(snapshot)

    return snapshots


#  Arbitrage Detection

@dataclass
class ArbitrageOpportunity:
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float      # ask on buy exchange
    sell_price: float      # bid on sell exchange
    gross_spread_bps: float
    net_spread_bps: float      # after fees
    max_size_usd: float   # constrained by orderbook depth
    feasibility_score: float
    latency_ms: float         # total round-trip latency
    snapshot_idx: int


def scan_for_arbitrage(
    snapshot: Dict[str, Orderbook],
    symbols: List[str],
    min_net_spread_bps: float = 5.0,
    max_order_usd: float = 10_000,
) -> List[ArbitrageOpportunity]:
    """
    Simple cross-exchange arbitrage: buy on exchange A (at ask), sell on B (at bid).
    Fees: taker on both legs. No withdrawal (assumes pre-positioned capital).
    """
    opportunities = []

    for sym in symbols:
        books = {ex: snapshot[f"{sym}_{ex}"] for ex in EXCHANGES if f"{sym}_{ex}" in snapshot}

        exchanges = list(books.keys())
        for i, buy_ex in enumerate(exchanges):
            for sell_ex in exchanges:
                if buy_ex == sell_ex:
                    continue

                buy_book = books[buy_ex]
                sell_book = books[sell_ex]

                buy_price = buy_book.ask    #  buy at the ask
                sell_price = sell_book.bid   # sell at the bid

                if sell_price <= buy_price:
                    continue   # no raw spread

                gross_bps = (sell_price - buy_price) / buy_price * 10000

                # Fee cost (taker on both legs)
                buy_fee_bps  = EXCHANGES[buy_ex].taker_fee  * 10000
                sell_fee_bps = EXCHANGES[sell_ex].taker_fee * 10000
                net_bps = gross_bps - buy_fee_bps - sell_fee_bps

                if net_bps < min_net_spread_bps:
                    continue

                # Size constrained by available liquidity
                max_size = min(buy_book.ask_size, sell_book.bid_size, max_order_usd)
                if max_size < EXCHANGES[buy_ex].min_order_usd:
                    continue

                # Feasibility: penalize high latency and low liquidity

                total_latency = EXCHANGES[buy_ex].latency_ms + EXCHANGES[sell_ex].latency_ms
                latency_penalty = np.exp(-total_latency / 50)   # exponential decay
                size_score = min(max_size / max_order_usd, 1.0)
                spread_score = min(net_bps / 50, 1.0)
                feasibility = latency_penalty * 0.4 + size_score * 0.3 + spread_score * 0.3

                opportunities.append(ArbitrageOpportunity(
                    symbol=sym,
                    buy_exchange=buy_ex,
                    sell_exchange=sell_ex,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    gross_spread_bps=gross_bps,
                    net_spread_bps=net_bps,
                    max_size_usd=max_size,
                    feasibility_score=feasibility,
                    latency_ms=total_latency,
                    snapshot_idx=0,
                ))

    return opportunities


#  Execution Feasibility Model



def model_execution_feasibility(opp: ArbitrageOpportunity) -> Dict:
    """
    Estimate the probability that the arbitrage is still exploitable by
    the time our order arrives (factoring in latency and price dynamics).
    """
    # Price drift during latency: assume 20bps/second vol in crypto
    vol_per_ms = 0.20 / (1000 * 10000)  # bps per ms
    spread_erosion_bps = opp.latency_ms * vol_per_ms * 10000
    expected_net_bps = opp.net_spread_bps - spread_erosion_bps

    # P(still profitable) modeled as normal CDF

    from scipy.stats import norm
    sigma = spread_erosion_bps * 1.5  # uncertainty in spread erosion
    prob_profitable = norm.cdf(expected_net_bps / max(sigma, 0.1))

    pnl_usd = expected_net_bps / 10000 * opp.max_size_usd
    expected_pnl = prob_profitable * pnl_usd

    return {
        "expected_net_bps": expected_net_bps,
        "prob_profitable": prob_profitable,
        "expected_pnl_usd": expected_pnl,
        "spread_erosion_bps": spread_erosion_bps,
    }


# ─── Analytics 

def run_scan(
    symbols: List[str] = None,
    n_snapshots: int = 200,
    min_net_bps: float = 3.0,
) -> Tuple[List[ArbitrageOpportunity], pd.DataFrame]:
    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "BNB"]

    true_prices = {"BTC": 67500, "ETH": 3500, "SOL": 185, "BNB": 600}

    snapshots = simulate_exchange_prices(symbols, true_prices, n_snapshots=n_snapshots)

    all_opps = []
    for i, snap in enumerate(snapshots):
        opps = scan_for_arbitrage(snap, symbols, min_net_spread_bps=min_net_bps)
        for opp in opps:
            opp.snapshot_idx = i
        all_opps.extend(opps)

    if not all_opps:
        return [], pd.DataFrame()

    records = []
    for opp in all_opps:
        feas = model_execution_feasibility(opp)
        records.append({
            "snapshot": opp.snapshot_idx,
            "symbol": opp.symbol,
            "buy_ex": opp.buy_exchange,
            "sell_ex": opp.sell_exchange,
            "gross_bps": opp.gross_spread_bps,
            "net_bps": opp.net_spread_bps,
            "max_size_usd": opp.max_size_usd,
            "latency_ms": opp.latency_ms,
            "feasibility": opp.feasibility_score,
            "prob_profitable": feas["prob_profitable"],
            "expected_pnl_usd": feas["expected_pnl_usd"],
            "expected_net_bps": feas["expected_net_bps"],
        })

    df = pd.DataFrame(records)
    return all_opps, df



# Visualization 

def plot_arbitrage_dashboard(df: pd.DataFrame, save_path: str = None):
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("Crypto Cross-Exchange Arbitrage Scanner", fontsize=14, fontweight="bold")

    # 1. Opportunity count over time by symbol
    ax1 = fig.add_subplot(gs[0, :2])
    for sym in df["symbol"].unique():
        sub = df[df["symbol"] == sym].groupby("snapshot").size()
        ax1.plot(sub.index, sub.values, label=sym, linewidth=1.2)
    ax1.set_xlabel("Snapshot #"); ax1.set_ylabel("# Opportunities")
    ax1.set_title("Arbitrage Opportunity Count Over Time")
    ax1.legend(fontsize=8)


    # 2. Net spread distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df["net_bps"], bins=40, color="#3498db", alpha=0.8, edgecolor="white")
    ax2.axvline(df["net_bps"].mean(), color="red", lw=1.5, linestyle="--",
                label=f"Mean {df['net_bps'].mean():.1f} bps")
    ax2.set_xlabel("Net Spread (bps)"); ax2.set_ylabel("Count")
    ax2.set_title("Net Spread Distribution"); ax2.legend(fontsize=8)


    # 3. Exchange pair heatmap (average net bps)
    ax3 = fig.add_subplot(gs[1, :2])
    pair_stats = df.groupby(["buy_ex", "sell_ex"])["net_bps"].mean().reset_index()
    pivot = pair_stats.pivot(index="buy_ex", columns="sell_ex", values="net_bps").fillna(0)
    im = ax3.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax3.set_xticks(range(len(pivot.columns))); ax3.set_xticklabels(pivot.columns, rotation=45, fontsize=8)
    ax3.set_yticks(range(len(pivot.index))); ax3.set_yticklabels(pivot.index, fontsize=8)
    ax3.set_title("Avg Net Spread (bps): Buy→Sell Exchange Pairs")
    plt.colorbar(im, ax=ax3)


    # 4. Expected PnL vs feasibility
    ax4 = fig.add_subplot(gs[1, 2])
    sc = ax4.scatter(df["feasibility"], df["expected_pnl_usd"],
                     c=df["net_bps"], cmap="viridis", alpha=0.3, s=10)
    ax4.set_xlabel("Feasibility Score"); ax4.set_ylabel("Expected PnL (USD)")
    ax4.set_title("Expected PnL vs Feasibility")
    plt.colorbar(sc, ax=ax4, label="Net bps")





    # 5. Prob profitable by latency
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(df["latency_ms"], df["prob_profitable"], alpha=0.2, s=5, color="#e74c3c")
    ax5.set_xlabel("Round-trip Latency (ms)"); ax5.set_ylabel("P(profitable)")
    ax5.set_title("Latency vs Profitability Probability")


    # 6. Top exchange pairs by total expected PnL
    ax6 = fig.add_subplot(gs[2, 1])
    pair_pnl = df.groupby(["buy_ex", "sell_ex"])["expected_pnl_usd"].sum().nlargest(8)
    pair_labels = [f"{b[:3]}→{s[:3]}" for b, s in pair_pnl.index]
    ax6.barh(pair_labels, pair_pnl.values, color="#2ecc71", alpha=0.8)
    ax6.set_xlabel("Total Expected PnL (USD)")
    ax6.set_title("Top Exchange Pairs by Expected PnL")


    # 7. Symbol breakdown
    ax7 = fig.add_subplot(gs[2, 2])
    sym_pnl = df.groupby("symbol")["expected_pnl_usd"].sum()
    ax7.pie(sym_pnl.values, labels=sym_pnl.index, autopct="%1.1f%%", startangle=90)
    ax7.set_title("Expected PnL by Symbol")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.show()



# Main 

def run(plot: bool = True):
    print("\n" + "="*60)
    print("  Crypto Arbitrage Scanner — Cross-Exchange Inefficiency Detector")
    print("="*60 + "\n")

    symbols = ["BTC", "ETH", "SOL", "BNB"]
    n_snapshots = 200

    print(f"► Simulating {n_snapshots} price snapshots across {len(EXCHANGES)} exchanges …")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Exchanges: {', '.join(EXCHANGES.keys())}")

    all_opps, df = run_scan(symbols=symbols, n_snapshots=n_snapshots, min_net_bps=3.0)

    if df.empty:
        print("  No arbitrage opportunities found above threshold.")
        return

    print(f"\n► Scan Results")
    print(f"  Total opportunities detected: {len(df):,}")
    print(f"  Unique exchange pairs:        {df.groupby(['buy_ex','sell_ex']).ngroups}")
    print(f"  Avg net spread:               {df['net_bps'].mean():.2f} bps")
    print(f"  Max net spread:               {df['net_bps'].max():.2f} bps")
    print(f"  Avg feasibility score:        {df['feasibility'].mean():.3f}")
    print(f"  Avg P(profitable):            {df['prob_profitable'].mean():.3f}")
    print(f"  Total expected PnL:           ${df['expected_pnl_usd'].sum():,.2f}")

    print(f"\n► Top 10 Opportunities by Expected PnL:")
    top = df.nlargest(10, "expected_pnl_usd")[
        ["symbol", "buy_ex", "sell_ex", "net_bps", "max_size_usd",
         "prob_profitable", "expected_pnl_usd", "latency_ms"]
    ]
    print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n► By Symbol:")
    print(df.groupby("symbol").agg(
        count=("net_bps", "count"),
        avg_net_bps=("net_bps", "mean"),
        total_exp_pnl=("expected_pnl_usd", "sum"),
    ).round(2).to_string())

    print(f"\n► By Exchange Pair (buy→sell, top 8 by count):")
    pair_stats = df.groupby(["buy_ex", "sell_ex"]).agg(
        count=("net_bps", "count"),
        avg_net_bps=("net_bps", "mean"),
        avg_pnl=("expected_pnl_usd", "mean"),
    ).nlargest(8, "count").round(2)
    print(pair_stats.to_string())

    if plot:
        print("\n► Generating dashboard …")
        plot_arbitrage_dashboard(df, save_path="crypto_arbitrage_dashboard.png")

    return df


if __name__ == "__main__":
    run(plot=True)
