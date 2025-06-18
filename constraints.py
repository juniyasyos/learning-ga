def is_valid(weights, min_weight=0.05, max_weight=0.5):
    return all(min_weight <= w <= max_weight for w in weights) and abs(sum(weights) - 1) < 0.01
