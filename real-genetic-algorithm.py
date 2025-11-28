# -*- coding: utf-8 -*-
"""
Real-Valued Genetic Algorithm from Scratch in Python
====================================================

A clean, efficient, and fully documented real-coded genetic algorithm
for continuous global optimization.

Features:
- Real-valued chromosomes (no binary encoding)
- Roulette wheel selection (fitness-proportional)
- Arithmetic (symmetric) crossover
- Gaussian-like uniform mutation
- Elitism
- Tested on Sphere and Levy functions

Perfect for learning evolutionary computation!
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = 3.14159265359

# Global lists for storing results across multiple runs
all_avg_history = []
all_best_history = []
generations_record = []


# ================================
# 1. Population Initialization
# ================================

def initialize_population(pop_size, dim, min_val, max_val):
    """Generate random real-valued population"""
    return np.random.uniform(min_val, max_val, (pop_size, dim))


def population_stats(fitness_values):
    """Return best and average fitness"""
    return fitness_values.min(), fitness_values.mean()


# ================================
# 2. Selection: Roulette Wheel
# ================================

def roulette_wheel_selection(population, fitness_values):
    """Select parents using fitness-proportional (roulette) selection"""
    # For minimization: invert fitness (higher = better chance)
    max_fitness = fitness_values.max()
    inverted_fitness = max_fitness - fitness_values + 1e-6
    probabilities = inverted_fitness / inverted_fitness.sum()

    parents = []
    for _ in range(0, len(population), 2):
        parent1 = population[np.random.choice(len(population), p=probabilities)]
        parent2 = population[np.random.choice(len(population), p=probabilities)]
        parents.append((parent1, parent2))
    return parents


# ================================
# 3. Crossover: Arithmetic (Symmetric)
# ================================

def arithmetic_crossover(parents):
    """Symmetric arithmetic crossover: y1 = r*a + (1-r)*b"""
    offspring = []
    for parent1, parent2 in parents:
        r = random.random()  # Random weight in [0,1]
        child1 = r * parent1 + (1 - r) * parent2
        child2 = (1 - r) * parent1 + r * parent2
        offspring.extend([child1, child2])
    return np.array(offspring)


# ================================
# 4. Mutation: Uniform Random Perturbation
# ================================

def mutate(offspring, mutation_rate=0.2, mutation_strength=1.0, bounds=None):
    """Add random perturbation to genes with probability"""
    mutated = offspring.copy()
    mask = np.random.random(offspring.shape) < mutation_rate
    perturbation = mutation_strength * (np.random.random(offspring.shape) - 0.5) * 2
    mutated += mask * perturbation

    # Optional: clip to bounds
    if bounds is not None:
        np.clip(mutated, bounds[0], bounds[1], out=mutated)
    return mutated


# ================================
# 5. Elitism
# ================================

def apply_elitism(best_old_individuals, new_population, elitism_rate):
    """Keep top individuals from previous generation"""
    n_elite = max(1, int(len(new_population) * elitism_rate))
    elite = best_old_individuals[:n_elite]
    survivors = new_population[n_elite:]
    return np.vstack([elite, survivors])


# ================================
# 6. Objective Functions
# ================================

def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2), minimum at (0,0,...)"""
    return np.sum(x**2)


def levy_function(x):
    """Levy function - complex multimodal, minimum at (1,1,...)"""
    x = np.array(x)
    term1 = np.sin(3 * pi * x[0])**2
    term2 = (x[:-1] - 1)**2 * (1 + np.sin(3 * pi * x[1:])**2)
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * pi * x[-1])**2)
    return term1 + np.sum(term2) + term3


# ================================
# 7. Main Genetic Algorithm
# ================================

def real_genetic_algorithm(
    cost_function,
    bounds,
    population_size=100,
    dimensions=2,
    max_generations=500,
    mutation_rate=0.3,
    mutation_strength=1.0,
    elitism_rate=0.1,
    target_error=1e-6
):
    min_val, max_val = bounds

    # Initialize population
    population = initialize_population(population_size, dimensions, min_val, max_val)
    best_history = []
    avg_history = []
    stagnation_counter = 0
    best_fitness_ever = float('inf')

    print(f"Starting Real-Valued GA | Pop: {population_size} | Dim: {dimensions}")
    print(f"{'Gen':>4} | {'Best':>12} | {'Average':>12} | {'x':>15} {'y':>15}")

    for generation in range(max_generations):
        # Evaluate fitness
        fitness = np.array([cost_function(ind) for ind in population])
        best_fitness = fitness.min()
        avg_fitness = fitness.mean()

        # Track history
        best_history.append(best_fitness)
        avg_history.append(avg_fitness)

        # Get best individual
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # Print progress
        if generation % 20 == 0 or generation < 10 or generation == max_generations - 1:
            x_vals = ", ".join([f"{v:8.4f}" for v in best_individual[:2]])
            if dimensions > 2:
                x_vals += " ..."
            print(f"{generation:4d} | {best_fitness:12.6f} | {avg_fitness:12.6f} | [{x_vals}]")

        # Check convergence
        if best_fitness < target_error:
            print(f"\nSolution found! Fitness: {best_fitness:.10f}")
            print(f"Best solution: {best_individual}")
            break

        if best_fitness < best_fitness_ever:
            best_fitness_ever = best_fitness
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter > 50:
            print(f"\nConverged (no improvement). Best fitness: {best_fitness:.10f}")
            break

        # Selection
        parents = roulette_wheel_selection(population, fitness)

        # Crossover
        offspring = arithmetic_crossover(parents)

        # Mutation
        offspring = mutate(offspring, mutation_rate, mutation_strength, bounds)

        # Elitism: keep best from previous generation
        ranked_indices = np.argsort(fitness)
        best_old = population[ranked_indices]
        population = apply_elitism(best_old, offspring, elitism_rate)

    # Save results for plotting
    all_avg_history.append(avg_history)
    all_best_history.append(best_history)
    generations_record.append(generation + 1)

    return best_individual, best_fitness


# ================================
# 8. Plotting Results
# ================================

def plot_convergence():
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i, avg in enumerate(all_avg_history):
        plt.plot(avg, label=f'Run {i+1}', color=colors[i % len(colors)])
    plt.title('Average Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i, best in enumerate(all_best_history):
        plt.plot(best, label=f'Run {i+1}', color=colors[i % len(colors)])
    plt.title('Best Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ================================
# 9. Run Experiments
# ================================

if __name__ == "__main__":
    pop_sizes = [50, 100, 150]
    runs_per_size = 3

    for pop_size in pop_sizes:
        print(f"\n{'='*70}")
        print(f"RUNNING REAL-VALUED GA WITH POPULATION SIZE = {pop_size}")
        print(f"{'='*70}")

        for run in range(runs_per_size):
            print(f"\n--- Run {run+1}/{runs_per_size} ---")
            real_genetic_algorithm(
                cost_function=levy_function,
                bounds=(-10, 10),
                population_size=pop_size,
                dimensions=2,
                max_generations=400,
                mutation_rate=0.25,
                mutation_strength=2.0,
                elitism_rate=0.1
            )

        plot_convergence()
        # Reset for next population size
        all_avg_history.clear()
        all_best_history.clear()
        generations_record.clear()
