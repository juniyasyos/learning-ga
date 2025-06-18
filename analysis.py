import matplotlib.pyplot as plt
import numpy as np

def plot_evolution(best_fitness, avg_fitness):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness, label='Best Sharpe Ratio')
    plt.plot(avg_fitness, label='Average Sharpe Ratio', linestyle='--')
    plt.xlabel('Generasi')
    plt.ylabel('Sharpe Ratio')
    plt.title('Perkembangan Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_allocation(weights, tickers, initial_investment=100_000_000):
    print("\n=== Solusi Terbaik ===")
    weights = [round(float(w), 4) for w in weights]
    print(f"Bobot Portofolio: {weights}")
    nominal = [round(w * initial_investment) for w in weights]
    print(f"\nAlokasi Dana (dari Rp{initial_investment:,}):")
    for ticker, w, n in zip(tickers, weights, nominal):
        print(f"{ticker:<8}: {w:.2%} = Rp{n:,}")

def display_history(best_fitness, avg_fitness):
    print("\n=== Riwayat Sharpe Ratio ===")
    for i, (best, avg) in enumerate(zip(best_fitness, avg_fitness)):
        print(f"Gen {i+1:3d}: Best = {best:.4f}, Avg = {avg:.4f}")

def display_weights_by_generation(best_solutions):
    try:
        gen = int(input(f"Masukkan nomor generasi (1 - {len(best_solutions)}): "))
        if 1 <= gen <= len(best_solutions):
            weights = [round(w, 4) for w in best_solutions[gen - 1]]
            print(f"\nGenerasi ke-{gen}")
            print(f"Bobot Portofolio: {weights}")
        else:
            print("Nomor generasi di luar jangkauan.")
    except ValueError:
        print("Input tidak valid.")

def display_raw_data(data):
    print("\n=== Data Harga Saham (Adj Close) ===")
    print(data.tail(10))
