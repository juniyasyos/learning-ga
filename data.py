import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ======== Tickers IDX (bisa ditambah sesuai kebutuhan) ========
def get_all_idx_tickers():
    return [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "UNVR.JK",
        "ASII.JK", "BBTN.JK", "BRPT.JK", "ICBP.JK", "INDF.JK",
        "ANTM.JK", "ADRO.JK", "PGAS.JK", "MDKA.JK", "TOWR.JK",
        "PTBA.JK", "SMGR.JK", "INCO.JK", "CPIN.JK", "AKRA.JK",
    ]

# ======== Ambil data harga historis ========
def download_stock_data(tickers, start="2018-01-01", end="2024-12-31"):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)["Close"]
    return data.dropna(axis=1, how='any')

# ======== Hitung return harian ========
def compute_daily_returns(price_df):
    return price_df.pct_change().dropna()

# ======== Ambil data fundamental dari yfinance ========
def get_fundamentals(tickers):
    data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            data.append({
                "Ticker": ticker,
                "PER": info.get("trailingPE", None),
                "PBV": info.get("priceToBook", None),
                "ROE": info.get("returnOnEquity", None),
                "DebtToEquity": info.get("debtToEquity", None),
                "EPS": info.get("trailingEps", None),
                "MarketCap": info.get("marketCap", None),
                "Sector": info.get("sector", None),
                "RevenueGrowth": info.get("revenueGrowth", None),
            })
        except Exception as e:
            print(f"Gagal mengambil data {ticker}: {e}")
            continue

    return pd.DataFrame(data)

# ======== Seleksi saham terbaik berdasarkan fundamental + Sharpe ========
def select_top_stocks(price_df, risk_free_rate=0.0, top_n=5):
    all_tickers = price_df.columns.tolist()
    fundamentals_df = get_fundamentals(all_tickers)

    # Filter berdasarkan kriteria fundamental
    is_financial = fundamentals_df["Sector"] == "Financial Services"
    is_non_financial = ~is_financial

    screened = fundamentals_df[
        (
            (fundamentals_df["PER"] <= 25) &
            (fundamentals_df["ROE"] >= 0.1) &
            (fundamentals_df["RevenueGrowth"] >= 0) &
            (
                (is_financial) |
                ((fundamentals_df["DebtToEquity"] < 1.5) & is_non_financial)
            )
        )
    ]

    selected_tickers = screened["Ticker"].tolist()
    print(f"📊 Setelah filter fundamental: {len(selected_tickers)} saham lolos")

    if not selected_tickers:
        print("⚠️ Tidak ada saham yang lolos filter fundamental.")
        return []

    # Filter harga saham sesuai ticker
    filtered_prices = price_df[selected_tickers].dropna(axis=1, how='any')
    print(f"📉 Setelah filter harga (drop NaN): {filtered_prices.shape[1]} saham tersedia")

    # Hitung Sharpe Ratio
    returns = compute_daily_returns(filtered_prices)
    mean_returns = returns.mean()
    volatilities = returns.std()
    sharpe_ratios = (mean_returns - risk_free_rate) / volatilities.replace(0, np.nan)
    sharpe_ratios = sharpe_ratios.dropna()

    print(f"📈 Setelah hitung Sharpe Ratio: {len(sharpe_ratios)} saham valid")

    top_tickers = sharpe_ratios.sort_values(ascending=False).head(top_n).index.tolist()

    if len(top_tickers) < top_n:
        print(f"⚠️ Hanya {len(top_tickers)} saham tersedia, kurang dari top_n={top_n}. Menyesuaikan hasil.")

    return top_tickers


# ======== Split Data ========
def split_data(tickers, train_start, train_end, test_start, test_end):
    data = yf.download(tickers, start=train_start, end=test_end, auto_adjust=False)["Close"].dropna(axis=1)
    train_data = data[train_start:train_end]
    test_data = data[test_start:test_end]
    return train_data.pct_change().dropna(), test_data.pct_change().dropna()

# ======== Statistik portofolio ========
def get_statistics(returns):
    exp_returns = returns.mean()
    cov_matrix = returns.cov()
    return exp_returns, cov_matrix

# ======== Rolling data tahunan ========
def get_rolling_data(start_year, end_year, tickers):
    all_returns = yf.download(tickers, start=f"{start_year-3}-01-01", end=f"{end_year+1}-01-01", auto_adjust=False)["Adj Close"]
    returns = all_returns.pct_change(fill_method=None).dropna()

    rolling_data = {}
    for year in range(start_year, end_year + 1):
        train_start = f"{year-3}-01-01"
        train_end = f"{year-1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"

        train = returns.loc[train_start:train_end]
        test = returns.loc[test_start:test_end]

        if len(train) > 50 and len(test) > 20:
            rolling_data[year] = (train, test)

    return rolling_data
