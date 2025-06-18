import numpy as np
import os
from tabulate import tabulate
from data import (
    get_all_idx_tickers, download_stock_data, select_top_stocks,
    split_data, get_statistics, get_rolling_data, get_fundamentals
)
from ga import GeneticAlgorithm
from analysis import (
    plot_evolution, display_allocation, display_history,
    display_weights_by_generation, display_raw_data
)

# === Konfigurasi ===
initial_investment = 500_000_000
train_range = ("2018-01-01", "2023-12-31")
test_range = ("2024-01-01", "2024-12-31")
generations = 100
top_n_stocks = 5

# === Inisialisasi dan Unduh Data ===
print("🔄 Mengunduh data saham IDX...")
tickers_all = get_all_idx_tickers()
full_data = download_stock_data(tickers_all)
fundamentals = get_fundamentals(tickers_all)

# === Tampilkan Info Inisialisasi ===
print("\n📋 === INFORMASI INISIALISASI ===")
print(f"🧾 Jumlah Total Saham IDX: {len(tickers_all)}")
print(f"📅 Periode Training: {train_range[0]} s/d {train_range[1]}")
print(f"📅 Periode Testing : {test_range[0]} s/d {test_range[1]}")
print(f"💰 Investasi Awal  : Rp {initial_investment:,.0f}")
print(f"⚙️ Generasi GA     : {generations}")
print(f"🔍 Top Saham Dipilih: {top_n_stocks}")
print("\n📊 Data Fundamental Saham Teratas:")
print(tabulate(fundamentals.head(10), headers="keys", tablefmt="pretty"))

# === Seleksi Saham ===
tickers = select_top_stocks(full_data, top_n=top_n_stocks)
if len(tickers) < 2:
    print("❌ Saham tidak cukup untuk portofolio.")
    exit()
print(f"\n✅ Saham Terpilih untuk Portofolio: {tickers}")

# === Proses Data dan Inisialisasi GA ===
train_returns, test_returns = split_data(tickers, *train_range, *test_range)
exp_returns, cov_matrix = get_statistics(train_returns)

ga = GeneticAlgorithm(exp_returns, cov_matrix, generations=generations)
best_weights = ga.run()

# === Fungsi Ringkasan Portofolio ===
def summary(weights, returns, label=""):
    mean = np.dot(weights, returns.mean())
    vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe = mean / vol if vol > 0 else 0
    total_return = (1 + mean) ** len(returns) - 1
    final_value = initial_investment * (1 + total_return)

    print(f"\n=== {label} ===")
    print(f"Return Harian  : {mean:.4f}")
    print(f"Volatilitas    : {vol:.4f}")
    print(f"Sharpe Ratio   : {sharpe:.4f}")
    print(f"Total Return   : {total_return:.2%}")
    print(f"Nilai Akhir    : Rp {final_value:,.0f}")

# === Menu Interaktif ===
def menu():
    global ga, best_weights
    while True:
        # Tunggu Enter Sebelum Menu
        input("\nTekan [Enter] untuk masuk ke menu...")
        print("""
=== MENU INTERAKTIF ===
1. Grafik Evolusi
2. Solusi Terbaik
3. History Generasi
4. Bobot per Generasi
5. Data Harga Saham
6. Validasi 2024
7. Rolling Validation
8. Keluar
9. Clear Layar
        """)
        choice = input("Pilih opsi (1-9): ")

        if choice == '1':
            plot_evolution(ga.best_fitness, ga.avg_fitness)
        elif choice == '2':
            display_allocation(ga.best_solutions[-1], tickers=tickers, initial_investment=initial_investment)
            summary(best_weights, train_returns, f"Train Portfolio - Investasi Rp {initial_investment:,.0f}")
        elif choice == '3':
            display_history(ga.best_fitness, ga.avg_fitness)
        elif choice == '4':
            display_weights_by_generation(ga.best_solutions)
        elif choice == '5':
            print("\n📈 Data Harga Saham (Terbaru):")
            print(test_returns.tail())
        elif choice == '6':
            summary(best_weights, test_returns, f"Validasi 2024 - Investasi Rp {initial_investment:,.0f}")
        elif choice == '7':
            print("\n🔁 Rolling Validation...")
            for year, (train, test) in get_rolling_data(2018, 2024, tickers).items():
                print(f"\n-- Tahun {year} --")
                exp_ret, cov = train.mean(), train.cov()
                ga = GeneticAlgorithm(exp_ret, cov, generations=generations)
                weights = ga.run()
                summary(weights, test, f"Rolling Tahun {year}")
        elif choice == '8':
            print("👋 Keluar dari program.")
            break
        elif choice == '9':
            os.system('clear' if os.name == 'posix' else 'cls')
        else:
            print("❌ Opsi tidak valid. Coba lagi.")

# Jalankan Menu
menu()
