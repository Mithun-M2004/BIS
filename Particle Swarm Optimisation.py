import numpy as np

def fitness(x):
    return x**2

num_particles = 30
num_iterations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

positions = np.random.uniform(-10, 10, num_particles)
velocities = np.zeros(num_particles)
pbest_positions = positions.copy()
pbest_scores = fitness(positions)
gbest_position = pbest_positions[np.argmin(pbest_scores)]
gbest_score = np.min(pbest_scores)

for _ in range(num_iterations):
    r1 = np.random.rand(num_particles)
    r2 = np.random.rand(num_particles)

    velocities = (w * velocities +
                  c1 * r1 * (pbest_positions - positions) +
                  c2 * r2 * (gbest_position - positions))

    positions += velocities

    fitness_values = fitness(positions)

    better_mask = fitness_values < pbest_scores
    pbest_positions[better_mask] = positions[better_mask]
    pbest_scores[better_mask] = fitness_values[better_mask]

    if np.min(pbest_scores) < gbest_score:
        gbest_score = np.min(pbest_scores)
        gbest_position = pbest_positions[np.argmin(pbest_scores)]

print(f"Best position: {gbest_position}")
print(f"Best fitness: {gbest_score}")
