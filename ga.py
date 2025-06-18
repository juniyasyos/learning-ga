import numpy as np
import random
from utils import normalize, crossover, mutate
from constraints import is_valid

class GeneticAlgorithm:
    def __init__(self, exp_returns, cov_matrix, rf_rate=0.02, pop_size=50, generations=100, alpha=1.0, beta=1.0):
        self.exp_returns = exp_returns
        self.cov_matrix = cov_matrix
        self.rf = rf_rate
        self.pop_size = pop_size
        self.generations = generations
        self.alpha = alpha 
        self.beta = beta   
        
        # Tracking
        self.best_fitness = []
        self.avg_fitness = []
        self.best_returns = []
        self.best_vols = []
        self.best_solutions = []

    def fitness(self, weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        port_return = np.dot(weights, self.exp_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0

        max_weight = np.max(weights)
        concentration_penalty = max(0, max_weight - 0.4)
        volatility_penalty = max(0, port_volatility - 0.03)

        fitness = sharpe_ratio - self.alpha * concentration_penalty - self.beta * volatility_penalty
        return fitness

    def init_population(self):
        return [normalize(np.random.rand(len(self.exp_returns))) for _ in range(self.pop_size)]

    def select(self, pop, fitnesses, k=3):
        selected = random.sample(list(zip(pop, fitnesses)), k)
        return max(selected, key=lambda x: x[1])[0]

    def run(self):
        pop = self.init_population()
        for gen in range(self.generations):
            fitnesses = [self.fitness(ind) for ind in pop]
            best_idx = np.argmax(fitnesses)
            best_sol = pop[best_idx]
            best_fit = fitnesses[best_idx]

            self.best_fitness.append(best_fit)
            self.avg_fitness.append(np.mean(fitnesses))
            self.best_solutions.append(best_sol)
            self.best_returns.append(np.dot(best_sol, self.exp_returns))
            self.best_vols.append(np.sqrt(np.dot(best_sol.T, np.dot(self.cov_matrix, best_sol))))

            elite_count = 2
            elite_idxs = np.argsort(fitnesses)[-elite_count:]
            elites = [pop[i] for i in elite_idxs]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self.select(pop, fitnesses)
                p2 = self.select(pop, fitnesses)
                child = crossover(p1, p2)
                mutate_rate = max(0.1, 1 - gen / self.generations)
                child = mutate(child, rate=mutate_rate) 
                if is_valid(child):
                    new_pop.append(child)

            pop = new_pop

        best_index = np.argmax(self.best_fitness)
        return self.best_solutions[best_index]
