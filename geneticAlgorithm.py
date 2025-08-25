import numpy as np

def sphere_function(x):
    return np.sum(x**2)

population_size = 50
num_genes = 2
mutation_rate = 0.1
crossover_rate = 0.9
num_generations = 100
bounds = (-10, 10)

population = np.random.uniform(bounds[0], bounds[1], (population_size, num_genes))

def evaluate_fitness(population):
    return np.array([sphere_function(individual) for individual in population])

for generation in range(num_generations):
    fitness = evaluate_fitness(population)
    
    total_fitness = np.sum(1 / (1 + fitness))
    selection_probabilities = (1 / (1 + fitness)) / total_fitness
    selected_indices = np.random.choice(np.arange(population_size), size=population_size, p=selection_probabilities)
    selected_population = population[selected_indices]

    offspring = []
    for i in range(0, population_size, 2):
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, num_genes)
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            offspring.extend([offspring1, offspring2])
        else:
            offspring.extend([selected_population[i], selected_population[i + 1]])
    offspring = np.array(offspring)

    for i in range(population_size):
        if np.random.rand() < mutation_rate:
            mutation = np.random.normal(0, 0.5, num_genes)
            offspring[i] += mutation
            offspring[i] = np.clip(offspring[i], bounds[0], bounds[1])

    population = offspring

    best_fitness = np.min(fitness)
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    
    print(f"Generation {generation + 1}/{num_generations}, Best Fitness: {best_fitness:.5f}, Best Solution: {best_solution}")

print("\nFinal Best Solution:")
print("Position:", best_solution)
print("Fitness:", best_fitness)
