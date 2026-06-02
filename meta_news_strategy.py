"""
Modular Meta-Learning Trading Strategy
=============================================================
Inspired by "Fusing Narrative Semantics for Financial Volatility Forecasting"
(Kong et al., ICAIF '25) — the M2VN paper.

This version is highly modularized into:
- core/: config, dataset, model
- data_sources/: news, earnings
- fusion/: fusion strategies
- methods/: meta-learning methods (FOMAML, ANIL, ProtoNet)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import *
from core.dataset import MultiModalDataset
from core.model import M2VNLite
from methods.utils import collate
from data_sources import NewsEmbedder, EarningsFetcher
from fusion import FUSION_STRATEGIES
from methods import METHODS

# ═══════════════════════════════════════════════════════════════
#  Meta-Training and Backtesting Engine
# ═══════════════════════════════════════════════════════════════

def meta_train(model, dataset, task_months, method_name="fomaml",
               meta_lr=META_LR, epochs=META_EPOCHS,
               batch_size=BATCH_SIZE, inner_lr=INNER_LR,
               inner_steps=INNER_STEPS, lam=LAMBDA_ALIGN):
    opt = torch.optim.Adam(model.parameters(), lr=meta_lr)
    method_obj = METHODS[method_name]()

    print(f"\n{'═'*60}")
    print(f"  Meta-Training  ·  {method_name.upper()}")
    print(f"  Tasks: {len(task_months)} months  ·  Epochs: {epochs}")
    print(f"{'═'*60}")

    for epoch in range(1, epochs + 1):
        losses = []
        np.random.shuffle(task_months)

        for year, month in task_months:
            idx = dataset.get_month_indices(year, month)
            if len(idx) < 4:
                continue

            mid     = len(idx) // 2
            support = collate(dataset, idx[:mid],  batch_size)
            query   = collate(dataset, idx[mid:],  batch_size)
            if not support or not query:
                continue

            opt.zero_grad()
            tl = method_obj.train_step(model, support, query, inner_lr, inner_steps, lam)
            # gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(tl)

        avg = np.mean(losses) if losses else float("nan")
        print(f"  Epoch {epoch:2d}/{epochs}  ·  Avg task loss: {avg:.6f}")

    return model


def backtest(model, dataset, test_months, client, method_name="fomaml",
             batch_size=BATCH_SIZE, inner_lr=INNER_LR,
             inner_steps=INNER_STEPS, max_weight=MAX_WEIGHT,
             txn_cost=TXN_COST, lam=LAMBDA_ALIGN, run_api_backtest=True):
    """
    For each test month:
      1. Adapt the model using support set.
      2. Predict positions on the query set.
    """
    method_obj = METHODS[method_name]()
    all_pos, all_tgt = [], []

    print(f"\n{'═'*60}")
    print(f"  Backtesting  ·  {method_name.upper()}")
    print(f"  Months: {len(test_months)}")
    print(f"{'═'*60}")

    for year, month in test_months:
        idx = dataset.get_month_indices(year, month)
        if len(idx) < 4:
            print(f"  {year}-{month:02d}: skipped (too few samples)")
            continue

        mid         = len(idx) // 2
        support_idx = idx[:mid]
        query_idx   = idx[mid:]

        support = collate(dataset, support_idx, batch_size)

        # --- clone & adapt ---
        adapted_state = method_obj.adapt(model, support, inner_lr, inner_steps, lam)

        # --- generate positions on query ---
        m_pos, m_tgt = [], []
        for b in collate(dataset, query_idx, batch_size):
            price, news, tf, target = b
            pos = method_obj.predict(adapted_state, price, news, tf)
            
            pos_np = np.clip(pos.numpy(), -max_weight, max_weight)
            cash   = np.clip(1.0 - np.abs(pos_np).sum(axis=1, keepdims=True),
                             0.0, 1.0)
            m_pos.append(np.concatenate([cash, pos_np], axis=1).astype(np.float32))
            m_tgt.append(target.numpy())

        if m_pos:
            p = np.concatenate(m_pos)
            t = np.concatenate(m_tgt)
            all_pos.append(p)
            all_tgt.append(t)
            print(f"  {year}-{month:02d}: {p.shape[0]} positions")

    if not all_pos:
        print("  ✗ No positions generated")
        return None

    positions = np.concatenate(all_pos)
    targets   = np.concatenate(all_tgt)
    print(f"\n  Total positions: {positions.shape}  ·  targets: {targets.shape}")

    if run_api_backtest:
        results = client.backtest_positions(positions,
                                            torch.tensor(targets),
                                            txn_cost=txn_cost)
        return results, positions, targets
    else:
        return positions, targets


# ═══════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════

def plot_comparison(all_results: dict, save_path: str):
    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5.5 * n), squeeze=False)
    palette = {"fomaml": "#E84855", "anil": "#2EC4B6", "protonet": "#FF9F1C"}

    for i, (method, (res, pos, _)) in enumerate(all_results.items()):
        ax = axes[i][0]
        inc_pnl   = res["incremental_pnl"]
        cum_pnl   = np.cumsum(inc_pnl)
        invested  = 1.0 - pos[:, 0]

        color = palette.get(method, "#333333")
        ax.plot(cum_pnl, color=color, linewidth=2,
                label=f"{method.upper()} cumulative PnL")
        ax.fill_between(range(len(cum_pnl)), cum_pnl, alpha=0.10, color=color)
        ax.set_xlabel("Trading Days (query-set steps)")
        ax.set_ylabel("Cumulative PnL", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(invested, color="mediumpurple", linewidth=1,
                 linestyle="--", alpha=0.55, label="Invested capital")
        ax2.set_ylabel("Invested (max 1.0)", color="mediumpurple")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor="mediumpurple")

        lines  = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper left")

        profit  = res["total_profit"] * 100
        sharpe  = res["sharpe_ratio"]
        mean_r  = res["mean_return"]
        ax.set_title(f"{method.upper()}  ·  Profit: {profit:+.2f}%  "
                     f"·  Sharpe: {sharpe:.3f}  ·  Mean ret: {mean_r:.6f}",
                     fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


# ═══════════════════════════════════════════════════════════════
#  Main Loop
# ═══════════════════════════════════════════════════════════════

def main():
    from numin2 import Numin2API

    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║  Meta-Learning Trading Strategy with News Embeddings     ║
    ║  Inspired by M2VN  (Kong et al., ICAIF '25)              ║
    ║  Algorithms: FOMAML  ·  ANIL  ·  PROTONET                ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── 1. API & data download ────────────────────────────
    print("[1/6] Initializing numin2 API …")
    client = Numin2API()

    if not os.path.exists(DATA_PATH):
        print("  Downloading daily returns …")
        client.download_data(type="daily", features="returns")
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_parquet(DATA_PATH)
        max_date = df.index.max()
        if hasattr(max_date, 'tz_localize') and max_date.tz is not None:
            max_date = max_date.tz_localize(None)
        # Update if older than 2 days
        if (pd.Timestamp.now() - max_date).days > 2:
            print(f"  Existing market data is up to {max_date.date()} (out of date). Downloading latest …")
            client.download_data(type="daily", features="returns")
            df = pd.read_parquet(DATA_PATH)
        else:
            print(f"  Market data is already up-to-date ({max_date.date()}) at {DATA_PATH}")

    # ── 2. Load market data ───────────────────────────────
    print("\n[2/6] Loading market data …")
    print(f"  Shape: {df.shape}  ·  "
          f"Range: {df.index.min().date()} → {df.index.max().date()}")

    # ── 3. Data Sources ───────────────────────────────────
    print("\n[3/6] Fetching data via Data Sources …")
    news_embedder = NewsEmbedder(client)
    news_emb = news_embedder.fetch_all()

    earnings_fetcher = EarningsFetcher(df.columns.tolist())
    earnings_dates = earnings_fetcher.fetch_all()

    # ── 4. Build dataset ──────────────────────────────────
    print("\n[4/6] Building multi-modal dataset …")
    ds = MultiModalDataset(df, news_emb, earnings_dates)
    print(f"  Dataset: {len(ds)} samples")

    # ── 5. Meta-train & backtest each method ──────────────
    print("\n[5/6] Training & backtesting (Walk-Forward 2023-2026) …")
    all_results = {}

    walk_forward_periods = [
        (2023, 2024),
        (2024, 2025),
        (2025, 2026),
    ]

    # Select Fusion Strategy (Modular)
    fusion_strategy_class = FUSION_STRATEGIES["gated"]

    for method in ["protonet"]:
        print(f"\n{'═'*60}")
        print(f"  EVALUATING METHOD: {method.upper()}")
        print(f"{'═'*60}")
        
        all_method_pos = []
        all_method_tgt = []

        for train_yr, test_yr in walk_forward_periods:
            print(f"\n  ► Walk-Forward Period: Train {train_yr} → Test {test_yr}")
            train_months = [(train_yr, m) for m in range(1, 13)]
            test_months  = [(test_yr, m) for m in range(1, 13) if not (test_yr == 2026 and m > 5)]
            
            # filter months with enough data
            train_months = [(y, m) for y, m in train_months if len(ds.get_month_indices(y, m)) >= 4]
            test_months  = [(y, m) for y, m in test_months if len(ds.get_month_indices(y, m)) >= 4]
            
            if not train_months or not test_months:
                print(f"  Skipping {train_yr}-{test_yr} due to insufficient data.")
                continue

            fusion_instance = fusion_strategy_class(LATENT_DIM)
            model = M2VNLite(fusion_strategy=fusion_instance)

            if train_yr == 2023:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  Model: {n_params:,} parameters")

            model = meta_train(model, ds, train_months, method_name=method)

            try:
                out = backtest(model, ds, test_months, client, method_name=method, run_api_backtest=False)
                if out is None:
                    continue
                positions, targets = out
                all_method_pos.append(positions)
                all_method_tgt.append(targets)
            except Exception as exc:
                print(f"  ✗ Backtest failed ({method} {test_yr}): {exc}")
                import traceback; traceback.print_exc()

        if not all_method_pos:
            print(f"  ✗ No valid positions generated for {method} across any period.")
            continue

        # Combine all walk-forward predictions
        final_positions = np.concatenate(all_method_pos)
        final_targets = np.concatenate(all_method_tgt)
        
        # Run standard numin2 backtest on the combined continuous predictions
        results = client.backtest_positions(final_positions,
                                            torch.tensor(final_targets),
                                            txn_cost=TXN_COST)
                                            
        all_results[method] = (results, final_positions, final_targets)

        invested_total = np.sum(1.0 - final_positions[:, 0])
        profit_pct     = results["total_profit"] * 100
        roi            = ((results["total_profit"] / invested_total) * 100
                          if invested_total > 0 else 0)

        print(f"\n  ┌─── {method.upper()} Walk-Forward Results (2024-2026) {'─'*2}┐")
        print(f"  │  Profit (vs portfolio) : {profit_pct:+.2f}%")
        print(f"  │  Cumulative invested   : {invested_total:.2f} units")
        print(f"  │  Profit / invested     : {roi:+.4f}%")
        print(f"  │  Sharpe ratio           : {results['sharpe_ratio']:.3f}")
        print(f"  │  Mean return            : {results['mean_return']:.6f}")
        print(f"  └{'─'*(40 + len(method))}┘")


    # ── 6. Plot ───────────────────────────────────────────
    print("\n[6/6] Plotting …")
    if all_results:
        plot_comparison(all_results, PLOT_PATH)
    else:
        print("  No results to plot.")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
