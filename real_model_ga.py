import yfinance as yf
import pandas as pd
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# ============================
# KONFIGURASI AWAL
# ============================

# Daftar saham (kode IDX di Yahoo Finance harus ditambah .JK)
stocks = ['BBRI.JK', 'BBCA.JK', 'TLKM.JK', 'ASII.JK', 'UNVR.JK']

# Range waktu
start_train = '2018-01-01'
end_train = f'{datetime.datetime.now().year - 1}-12-31'
start_test = f'{datetime.datetime.now().year}-01-01'
end_test = datetime.datetime.now().strftime('%Y-%m-%d')

# ============================
# AMBIL DATA HARGA
# ============================

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close']
    return data.dropna()

train_data = download_data(stocks, start_train, end_train)
test_data = download_data(stocks, start_test, end_test)

returns = train_data.pct_change().dropna()

# ============================
# SETUP GENETIC ALGORITHM
# ============================

num_assets = len(stocks)

# Fungsi evaluasi: memaksimalkan Sharpe Ratio
def evaluate(weights):
    weights = np.array(weights)
    port_return = np.dot(returns.mean(), weights) * 252  # tahunan
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = port_return / port_volatility if port_volatility != 0 else 0
    return sharpe_ratio,

# Inisialisasi DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # maksimasi Sharpe Ratio
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_weight", lambda: random.random())
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_weight, n=num_assets)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def normalize(individual):
    arr = np.array(individual)
    arr[arr < 0] = 0  
    total = np.sum(arr)
    if total == 0:
        return np.ones_like(arr) / len(arr) 
    return arr / total


toolbox.register("evaluate", lambda ind: evaluate(normalize(ind)))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ============================
# JALANKAN GA
# ============================

population = toolbox.population(n=100)
NGEN = 50
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, k=1)[0]
best_weights = normalize(best_ind[:])
best_weights = np.array(best_weights)


# ============================
# ANALISIS PORTOFOLIO TERBAIK
# ============================

print("Portofolio Optimal (berdasarkan Sharpe Ratio):")
for stock, weight in zip(stocks, best_weights):
    print(f"{stock}: {weight:.2%}")

# Evaluasi di data test (tahun ini)
test_returns = test_data.pct_change().dropna()
test_port_return = np.dot(test_returns.mean(), best_weights) * 252
cov_matrix = test_returns.cov() * 252
test_port_volatility = np.sqrt(np.dot(best_weights, np.dot(cov_matrix, best_weights)))
test_sharpe = test_port_return / test_port_volatility

print(f"\nEvaluasi Tahun Ini:")
print(f"Return: {test_port_return:.2%}")
print(f"Volatilitas: {test_port_volatility:.2%}")
print(f"Sharpe Ratio: {test_sharpe:.2f}")
