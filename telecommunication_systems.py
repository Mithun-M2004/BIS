import random
import math

# ----- Telecom System Parameters -----
num_users = 3
Pmax = 10
noise = 1

# Fitness function (Throughput maximization)
def fitness(power):
    throughput = sum(math.log2(1 + p / noise) for p in power)
    interference = sum(power) ** 2 * 0.01
    return throughput - interference

# ----- PSO Parameters -----
n_particles = 10
iterations = 30
w, c1, c2 = 0.5, 1.5, 1.5

# Initialize particles
pos = [[random.uniform(0, Pmax) for _ in range(num_users)] for _ in range(n_particles)]
vel = [[0]*num_users for _ in range(n_particles)]

pbest = pos[:]
gbest = max(pbest, key=fitness)

# ----- PSO Loop -----
for _ in range(iterations):
    for i in range(n_particles):
        for d in range(num_users):
            r1, r2 = random.random(), random.random()
            vel[i][d] = (w * vel[i][d] +
                         c1 * r1 * (pbest[i][d] - pos[i][d]) +
                         c2 * r2 * (gbest[d] - pos[i][d]))
            pos[i][d] += vel[i][d]
            pos[i][d] = max(0, min(Pmax, pos[i][d]))

        if fitness(pos[i]) > fitness(pbest[i]):
            pbest[i] = pos[i]
            if fitness(pbest[i]) > fitness(gbest):
                gbest = pbest[i]

print("Optimal Power Allocation:", gbest)
print("Maximum Throughput Value:", fitness(gbest))

print("Minimum value:", f(gbest))

