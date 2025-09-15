import numpy as np

def objective_function(x):
    return np.sum(x**2)

def levy_flight(Lambda):
    sigma1 = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
              (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=x.shape)
    v = np.random.normal(0, sigma2, size=x.shape)
    step = u / (abs(v) ** (1 / Lambda))
    return step

def cuckoo_search(n=25, dim=5, n_iterations=1000, pa=0.25):
    nests = np.random.uniform(-10, 10, size=(n, dim))
    fitness = np.array([objective_function(nest) for nest in nests])
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]

    for _ in range(n_iterations):
        new_nests = nests + 0.01 * levy_flight(1.5)
        new_fitness = np.array([objective_function(nest) for nest in new_nests])
        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]
        abandon = np.random.rand(n) < pa
        nests[abandon] = np.random.uniform(-10, 10, size=(np.sum(abandon), dim))
        fitness[abandon] = [objective_function(nest) for nest in nests[abandon]]
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()

    return best_nest, best_fitness

best_solution, best_value = cuckoo_search()
print("Best solution:", best_solution)
print("Best objective value:", best_value)
