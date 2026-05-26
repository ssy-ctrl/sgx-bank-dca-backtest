"""
SGX Three Major Bank Stocks - 10-Year DCA Backtest
Stocks: DBS (D05.SI), OCBC (O39.SI), UOB (U11.SI)
Strategies tested:
  1. Monthly DCA - invest fixed amount every month
  2. Quarterly DCA - invest fixed amount every quarter
  3. Weekly DCA - invest fixed amount every week
  4. Equal-Weight Portfolio (monthly, split evenly across 3 stocks)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
TICKERS   = {'DBS': 'D05.SI', 'OCBC': 'O39.SI', 'UOB': 'U11.SI'}
START     = '2014-01-01'
END       = '2024-01-01'
MONTHLY_INVEST = 1000   # SGD per month baseline

# ── 1. Download price data ──────────────────────────────────────
print("Downloading 10-year price data for DBS, OCBC, UOB ...")
raw = yf.download(list(TICKERS.values()), start=START, end=END, auto_adjust=True, progress=False)
prices = raw['Close'].copy()
prices.columns = list(TICKERS.keys())
prices.dropna(how='all', inplace=True)
prices.ffill(inplace=True)
print(f"Data loaded: {prices.index[0].date()} to {prices.index[-1].date()}  ({len(prices)} trading days)
")

# ── 2. Helper: simulate DCA for one ticker ─────────────────────
def simulate_dca(price_series, invest_dates, amount_per_period):
    shares = 0.0
    total_invested = 0.0
    records = []
    for d in invest_dates:
        avail = price_series.index[price_series.index >= d]
        if avail.empty:
            continue
        p = price_series.loc[avail[0]]
        if pd.isna(p) or p <= 0:
            continue
        bought = amount_per_period / p
        shares += bought
        total_invested += amount_per_period
        records.append({'date': avail[0], 'price': p, 'shares_bought': bought,
                        'total_shares': shares, 'total_invested': total_invested})
    df = pd.DataFrame(records).set_index('date')
    final_price = price_series.iloc[-1]
    final_value = shares * final_price
    profit = final_value - total_invested
    roi    = profit / total_invested * 100 if total_invested > 0 else 0
    years  = (price_series.index[-1] - price_series.index[0]).days / 365.25
    cagr   = ((final_value / total_invested) ** (1 / years) - 1) * 100 if total_invested > 0 else 0
    avg_cost = total_invested / shares if shares > 0 else 0
    return dict(ticker=str(price_series.name), total_invested=total_invested,
                final_value=final_value, profit=profit, roi=roi, cagr=cagr,
                total_shares=shares, avg_cost=avg_cost, final_price=final_price,
                records=df)

# ── 3. Build invest date schedules ─────────────────────────────
start_dt = pd.Timestamp(START)
end_dt   = pd.Timestamp(END)

monthly_dates   = pd.date_range(start=start_dt, end=end_dt, freq='MS')
quarterly_dates = pd.date_range(start=start_dt, end=end_dt, freq='QS')
weekly_dates    = pd.date_range(start=start_dt, end=end_dt, freq='W-MON')

# ── 4. Run all strategies ───────────────────────────────────────
strategies = {
    'Monthly (SGD 1000/mo)':    (monthly_dates,   MONTHLY_INVEST),
    'Quarterly (SGD 3000/qtr)': (quarterly_dates, MONTHLY_INVEST * 3),
    'Weekly (SGD 250/wk)':      (weekly_dates,    MONTHLY_INVEST / 4),
}

results = []
for strat_name, (dates, amount) in strategies.items():
    for ticker in TICKERS:
        r = simulate_dca(prices[ticker].rename(ticker), dates, amount)
        r['strategy'] = strat_name
        results.append(r)

# ── 5. Equal-weight portfolio (monthly, split evenly) ──────────
def simulate_portfolio(prices_df, invest_dates, total_amount):
    tickers = prices_df.columns.tolist()
    per_stock = total_amount / len(tickers)
    shares = {t: 0.0 for t in tickers}
    total_invested = 0.0
    records = []
    for d in invest_dates:
        avail = prices_df.index[prices_df.index >= d]
        if avail.empty:
            continue
        row = prices_df.loc[avail[0]]
        invested_this_period = 0
        for t in tickers:
            p = row[t]
            if pd.isna(p) or p <= 0:
                continue
            shares[t] += per_stock / p
            invested_this_period += per_stock
        total_invested += invested_this_period
        pv = sum(shares[t] * prices_df[t].iloc[-1] for t in tickers)
        records.append({'date': avail[0], 'total_invested': total_invested, 'portfolio_value': pv})
    df = pd.DataFrame(records).set_index('date')
    final_value = sum(shares[t] * prices_df[t].iloc[-1] for t in tickers)
    profit = final_value - total_invested
    roi    = profit / total_invested * 100 if total_invested > 0 else 0
    years  = (prices_df.index[-1] - prices_df.index[0]).days / 365.25
    cagr   = ((final_value / total_invested) ** (1 / years) - 1) * 100 if total_invested > 0 else 0
    return dict(strategy='Equal-Weight Portfolio (Monthly)', ticker='DBS+OCBC+UOB',
                total_invested=total_invested, final_value=final_value,
                profit=profit, roi=roi, cagr=cagr, records=df)

port_result = simulate_portfolio(prices, monthly_dates, MONTHLY_INVEST)
results.append(port_result)

# ── 6. Summary table ────────────────────────────────────────────
summary_rows = []
for r in results:
    summary_rows.append({
        'Strategy':               r['strategy'],
        'Ticker':                 r['ticker'],
        'Total Invested (SGD)':   round(r['total_invested'], 0),
        'Final Value (SGD)':      round(r['final_value'], 0),
        'Profit (SGD)':           round(r['profit'], 0),
        'ROI (%)':                round(r['roi'], 1),
        'CAGR (%)':               round(r['cagr'], 2),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.sort_values('CAGR (%)', ascending=False, inplace=True)

print("=" * 90)
print("SGX BANK STOCKS -- 10-YEAR DCA BACKTEST RESULTS (2014-01-01 to 2024-01-01)")
print("=" * 90)
print(df_summary.to_string(index=False))
print()

best = df_summary.iloc[0]
print("=" * 90)
print(f"BEST STRATEGY: {best['Strategy']}  |  Ticker: {best['Ticker']}")
print(f"  Total Invested : SGD {best['Total Invested (SGD)']:,.0f}")
print(f"  Final Value    : SGD {best['Final Value (SGD)']:,.0f}")
print(f"  Profit         : SGD {best['Profit (SGD)']:,.0f}")
print(f"  ROI            : {best['ROI (%)']:.1f}%")
print(f"  CAGR           : {best['CAGR (%)']:.2f}% per year")
print("=" * 90)

# ── 7. Charts ───────────────────────────────────────────────────
colors = {'DBS': '#003087', 'OCBC': '#C8102E', 'UOB': '#EF3340'}
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SGX Bank Stocks -- 10-Year DCA Backtest (2014-2024)', fontsize=15, fontweight='bold')

# 7a. CAGR bar chart
ax = axes[0, 0]
monthly_res = [r for r in results if 'Monthly' in r['strategy'] and r['ticker'] != 'DBS+OCBC+UOB']
tl = [r['ticker'] for r in monthly_res]
cg = [r['cagr']   for r in monthly_res]
bc = [colors.get(t, '#888') for t in tl]
bars = ax.bar(tl, cg, color=bc, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.set_title('CAGR by Stock (Monthly DCA)', fontsize=12)
ax.set_ylabel('CAGR (%)')
ax.set_ylim(0, max(cg) * 1.3)
ax.grid(axis='y', alpha=0.3)

# 7b. Portfolio value growth
ax = axes[0, 1]
for r in results:
    if r['strategy'] == 'Monthly (SGD 1000/mo)' and r['ticker'] != 'DBS+OCBC+UOB':
        rec = r['records']
        value_line = rec['total_shares'] * prices[r['ticker']].reindex(rec.index, method='ffill')
        ax.plot(value_line.index, value_line.values, label=r['ticker'],
                color=colors[r['ticker']], linewidth=2)
invested_ref = results[0]['records']['total_invested']
ax.plot(invested_ref.index, invested_ref.values, 'k--', linewidth=1.5, label='Invested')
ax.set_title('Portfolio Value Growth (Monthly DCA)', fontsize=12)
ax.set_ylabel('SGD')
ax.legend()
ax.grid(alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# 7c. DBS ROI by strategy
ax = axes[1, 0]
dbs_res = [r for r in results if r['ticker'] == 'DBS']
sn = [r['strategy'] for r in dbs_res]
rv = [r['roi']      for r in dbs_res]
bars = ax.barh(sn, rv, color=['#003087', '#0055A4', '#4A90D9'][:len(sn)], edgecolor='white')
for bar, val in zip(bars, rv):
    ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}%', va='center', fontweight='bold')
ax.set_title('DBS -- ROI by DCA Strategy', fontsize=12)
ax.set_xlabel('ROI (%)')
ax.grid(axis='x', alpha=0.3)

# 7d. Final value comparison
ax = axes[1, 1]
sub = df_summary[df_summary['Ticker'] != 'DBS+OCBC+UOB']
pivot = sub.pivot_table(index='Strategy', columns='Ticker', values='Final Value (SGD)')
pivot.plot(kind='bar', ax=ax, color=[colors.get(c, '#888') for c in pivot.columns],
           edgecolor='white', linewidth=1)
ax.set_title('Final Value by Strategy & Stock', fontsize=12)
ax.set_ylabel('SGD')
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=30)
ax.legend(title='Ticker')
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
plt.savefig('dca_backtest_results.png', dpi=150, bbox_inches='tight')
print("
Chart saved to dca_backtest_results.png")
print("
Done!")
