import random
import numpy as np
from deap import base, creator, tools, algorithms

# ==== Konfigurasi Sistem Pabrik ====
NUM_MACHINES = {
    'A': 10,
    'B': 30,
    'C': 20,
    'D': 15,
    'E': 5
}

MACHINE_PARAMS = {
    'A': {'output': 100, 'energy': 5, 'emission': 2},
    'B': {'output': 60, 'energy': 3, 'emission': 1.5},
    'C': {'output': 120, 'energy': 6, 'emission': 2.5},
    'D': {'output': 80, 'energy': 4, 'emission': 2},
    'E': {'output': 200, 'energy': 10, 'emission': 5},
}

MAX_WORK_HOURS = 16
TOTAL_ENERGY_CAP = 1500
EMISSION_CAP = 3000
ALPHA = 1.0
BETA = 0.05
GAMMA = 2.0

# ==== GA Setup ====
total_genes = sum(NUM_MACHINES.values())

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_hours", random.uniform, 0, MAX_WORK_HOURS)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_hours, n=total_genes)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    idx = 0
    total_output = 0
    total_energy = 0
    total_emission = 0

    for mtype in NUM_MACHINES:
        params = MACHINE_PARAMS[mtype]
        for _ in range(NUM_MACHINES[mtype]):
            hours = individual[idx]
            total_output += params['output'] * hours
            total_energy += params['energy'] * hours
            total_emission += params['emission'] * hours
            idx += 1

    penalty = 0
    if total_energy > TOTAL_ENERGY_CAP:
        penalty += (total_energy - TOTAL_ENERGY_CAP)
    if total_emission > EMISSION_CAP:
        penalty += (total_emission - EMISSION_CAP)

    fitness = ALPHA * total_output - BETA * total_energy - GAMMA * penalty
    return (fitness,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=8, sigma=5, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga():
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)

    return hof[0], log

# Tambahan fungsi: interpretasi hasil terbaik
def interpret_solution(individual):
    idx = 0
    summary = {
        'total_output': 0,
        'total_energy': 0,
        'total_emission': 0,
        'by_type': {}
    }

    for mtype in NUM_MACHINES:
        params = MACHINE_PARAMS[mtype]
        mesin_list = []

        for i in range(NUM_MACHINES[mtype]):
            hours = individual[idx]
            mesin_output = params['output'] * hours
            mesin_energy = params['energy'] * hours
            mesin_emission = params['emission'] * hours

            summary['total_output'] += mesin_output
            summary['total_energy'] += mesin_energy
            summary['total_emission'] += mesin_emission

            mesin_list.append({
                'id': f"{mtype}-{i+1}",
                'hours': round(hours, 2),
                'output': round(mesin_output, 2),
                'energy': round(mesin_energy, 2),
                'emission': round(mesin_emission, 2)
            })
            idx += 1

        summary['by_type'][mtype] = mesin_list

    return summary

# Jalankan algoritma
best_ind, logbook = run_ga()

# Interpretasi hasil terbaik
result = interpret_solution(best_ind)

# === Rangkuman Ringkas ===
print("\n=== RANGKUMAN PORTOFOLIO MESIN ===")
print(f"Total Output   : {round(result['total_output'], 2)} unit produksi")
print(f"Total Energi   : {round(result['total_energy'], 2)} kWh")
print(f"Total Emisi    : {round(result['total_emission'], 2)} kg CO2")
print(f"Fitness Score  : {round(best_ind.fitness.values[0], 2)}")

# === Detail per Tipe Mesin ===
print("\n=== DETAIL MESIN PER TIPE ===")
for mtype, mesin_list in result['by_type'].items():
    print(f"\nTipe Mesin {mtype} ({len(mesin_list)} unit):")
    for mesin in mesin_list:
        print(f"  {mesin['id']}: {mesin['hours']} jam | Output: {mesin['output']} | Energi: {mesin['energy']} | Emisi: {mesin['emission']}")
