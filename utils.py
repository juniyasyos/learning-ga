import numpy as np
import random

def normalize(weights):
    weights = np.maximum(0, weights)
    total = np.sum(weights)
    return weights / total if total > 0 else np.ones_like(weights) / len(weights)

def crossover(p1, p2, strategy="blend"):
    p1, p2 = np.array(p1), np.array(p2)
    if strategy == "uniform":
        mask = np.random.rand(len(p1)) > 0.5
        child = np.where(mask, p1, p2)
    else: 
        alpha = random.random()
        child = alpha * p1 + (1 - alpha) * p2
    return normalize(child)

def mutate(weights, rate=0.2, scale=0.05):
    """
    Mutasi pada bobot portofolio.
    - rate : peluang mutasi per gen.
    - scale: skala noise gaussian.
    """
    weights = np.array(weights)
    noise = np.random.normal(0, scale, size=len(weights))
    mask = np.random.rand(len(weights)) < rate
    weights[mask] += noise[mask]
    return normalize(weights)
