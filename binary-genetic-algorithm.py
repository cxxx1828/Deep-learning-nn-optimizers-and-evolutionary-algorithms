# -*- coding: utf-8 -*-
"""
Binary Genetic Algorithm - From Scratch in Python
Optimization of Sphere and Levy functions using binary-encoded chromosomes

Features:
- Binary encoding/decoding with configurable precision
- One-point and two-point crossover
- Mutation by inversion and bit-flip
- Roulette wheel selection + elitism
- Visualization of convergence
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pi = 3.14159265359

# Global lists to store results across runs
all_avg_list = []
all_best_list = []
generations_list = []


# ================================
# 1. Encoding / Decoding Functions
# ================================

def bin_encode(chromosome, bin_step, min_val, precision):
    """Encode a single real-valued gene into binary string"""
    ret = ""
    for gene in chromosome:
        val = int(round((gene - min_val) / bin_step))
        ret += bin(val)[2:].zfill(precision)  # Pad with zeros
    return ret


def bin_encode_chromosomes(chromosomes, precision, max_val, min_val):
    """Encode entire population (list of individuals) into binary"""
    bin_step = (max_val - min_val) / (2**precision - 1)
    return [bin_encode(ind, bin_step, min_val, precision) for ind in chromosomes]


def bin_decode(chromosome, bin_step, min_val, precision):
    """Decode a binary chromosome back to real values"""
    ret = []
    for i in range(0, len(chromosome), precision):
        bits = chromosome[i:i + precision]
        val = int(bits, 2)
        real_val = val * bin_step + min_val
        ret.append(real_val)
    return ret


def bin_decode_chromosomes(chromosomes, precision, max_val, min_val):
    """Decode entire population"""
    bin_step = (max_val - min_val) / (2**precision - 1)
    return [bin_decode(ind, bin_step, min_val, precision) for ind in chromosomes]


# ================================
# 2. Population Initialization
# ================================

def generate_initial_population(length, max_val, min_val, pop_size):
    """Generate random real-valued initial population"""
    return [[random.uniform(min_val, max_val) for _ in range(length)] for _ in range(pop_size)]


def population_stats(costs):
    """Return best and average fitness"""
    return costs[0], sum(costs) / len(costs)


# ================================
# 3. Selection Methods
# ================================

def rank_chromosomes(cost_func, chromosomes):
    """Rank individuals by fitness (lower = better)"""
    costs = [cost_func(ind) for ind in chromosomes]
    ranked = sorted(zip(chromosomes, costs), key=lambda x: x[1])
    chromosomes_sorted, costs_sorted = zip(*ranked)
    return list(chromosomes_sorted), list(costs_sorted)


def natural_selection(chromosomes, n_keep):
    """Keep only the top n individuals (elitism style)"""
    return chromosomes[:n_keep]


def roulette_selection(parents):
    """Roulette wheel selection - higher fitness = higher chance"""
    pairs = []
    for _ in range(0, len(parents), 2):
        weights = [(len(parents) - i) * random.random() for i in range(len(parents))]
        # Find two best weighted individuals
        max1 = weights.index(max(weights))
        weights[max1] = -1
        max2 = weights.index(max(weights))
        pairs.append([parents[max1], parents[max2]])
    return pairs


# ================================
# 4. Crossover Operators
# ================================

def one_point_crossover(pairs):
    """Single-point crossover"""
    children = []
    for parent1, parent2 in pairs:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        children.extend([child1, child2])
    return children


def two_point_crossover(pairs):
    """Two-point crossover"""
    children = []
    for parent1, parent2 in pairs:
        p1 = random.randint(0, len(parent1) - 1)
        p2 = random.randint(0, len(parent1) - 1)
        start, end = min(p1, p2), max(p1, p2)
        child1 = parent1[:start] + parent2[start:end] + parent1[end:]
        child2 = parent2[:start] + parent1[start:end] + parent2[end:]
        children.extend([child1, child2])
    return children


# ================================
# 5. Mutation Operators
# ================================

def inversion_mutation(chromosomes, mutation_rate):
    """Mutation by inverting a segment (reversal)"""
    mutated = []
    for chrom in chromosomes:
        if random.random() < mutation_rate:
            i = random.randint(0, len(chrom) - 1)
            j = random.randint(0, len(chrom) - 1)
            start, end = min(i, j), max(i, j)
            mutated_part = chrom[start:end][::-1]
            new_chrom = chrom[:start] + mutated_part + chrom[end:]
            mutated.append(new_chrom)
        else:
            mutated.append(chrom)
    return mutated


def bitflip_mutation(chromosomes, mutation_rate):
    """Classic bit-flip mutation"""
    mutated = []
    for chrom in chromosomes:
        if random.random() < mutation_rate:
            pos = random.randint(0, len(chrom) - 1)
            flipped_bit = '1' if chrom[pos] == '0' else '0'
            new_chrom = chrom[:pos] + flipped_bit + chrom[pos+1:]
            mutated.append(new_chrom)
        else:
            mutated.append(chrom)
    return mutated


# ================================
# 6. Elitism
# ================================

def apply_elitism(best_old, new_population, elitism_rate, pop_size):
    """Keep top individuals from previous generation"""
    n_elite = int(pop_size * elitism_rate)
    return best_old[:n_elite] + new_population[:pop_size - n_elite]


# ================================
# 7. Objective Functions
# ================================

def sphere_function(individual):
    """Sphere function: f(x,y) = x² + y², minimum at (0,0)"""
    x, y = individual
    return x**2 + y**2


def levy_function(individual):
    """Levy function - complex multimodal function, minimum at (1,1)"""
    x, y = individual
    term1 = (np.sin(3 * np.pi * x))**2
    term2 = (x - 1)**2 * (1 + (np.sin(3 * np.pi * y))**2)
    term3 = (y - 1)**2 * (1 + (np.sin(2 * np.pi * y))**2)
    return term1 + term2 + term3


# ================================
# 8. Main Genetic Algorithm Loop
# ================================

def genetic_algorithm(
    cost_func,
    search_range,
    population_size,
    mutation_rate=0.3,
    elitism_rate=0.1,
    chromosome_length=2,
    precision=16,
    max_generations=500
):
    min_val, max_val = search_range

    # Lists for plotting
    avg_history = []
    best_history = []
    stagnation_count = 0
    best_fitness = float('inf')

    # Initialize population
    population = generate_initial_population(chromosome_length, max_val, min_val, population_size)

    for generation in range(max_generations):
        # Evaluate and rank
        ranked_pop, costs = rank_chromosomes(cost_func, population)
        best_fit, avg_fit = population_stats(costs)

        print(f"Gen {generation+1:3d} | Avg: {avg_fit:8.4f} | Best: {best_fit:8.4f} | "
              f"x,y = [{ranked_pop[0][0]:.4f}, {ranked_pop[0][1]:.4f}]")

        avg_history.append(avg_fit)
        best_history.append(best_fit)

        # Check convergence
        if best_fit < 1e-4:
            print(f"\nSolution found! x,y = [{ranked_pop[0][0]:.6f}, {ranked_pop[0][1]:.6f}]")
            save_results(avg_history, best_history, generation)
            return

        if best_fit >= best_fitness:
            stagnation_count += 1
        else:
            best_fitness = best_fit
            stagnation_count = 0

        if stagnation_count > 30:
            print(f"\nConverged (no improvement). Best: x,y = [{ranked_pop[0][0]:.6f}, {ranked_pop[0][1]:.6f}]")
            save_results(avg_history, best_history, generation)
            return

        # Encode to binary
        binary_pop = bin_encode_chromosomes(ranked_pop, precision, max_val, min_val)
        pairs = roulette_selection(binary_pop)
        offspring = two_point_crossover(pairs)
        offspring = inversion_mutation(offspring, mutation_rate)
        # offspring = bitflip_mutation(offspring, mutation_rate)  # Alternative

        # Decode back
        population = bin_decode_chromosomes(offspring, precision, max_val, min_val)

        # Apply elitism
        population = apply_elitism(ranked_pop, population, elitism_rate, population_size)

    print(f"\nMax generations reached. Best: x,y = [{ranked_pop[0][0]:.6f}, {ranked_pop[0][1]:.6f}]")
    save_results(avg_history, best_history, max_generations)


def save_results(avg_hist, best_hist, gens):
    all_avg_list.append(avg_hist)
    all_best_list.append(best_hist)
    generations_list.append(gens)


# ================================
# 9. Plotting Results
# ================================

def plot_convergence():
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    for i, (avg, best, gens) in enumerate(zip(all_avg_list, all_best_list, generations_list)):
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(avg, label=f'Run {i+1}', color=colors[i])
        plt.title('Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(best, label=f'Run {i+1}', color=colors[i])
        plt.title('Best Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Cost')
        plt.yscale('log')
        plt.grid(True)

        plt.suptitle(f'Population Size: {len(avg)*10}')  # Rough estimate
        plt.tight_layout()
        plt.show()


# ================================
# 10. Run Experiments
# ================================

if __name__ == "__main__":
    pop_sizes = [20, 50, 100]

    for size in pop_sizes:
        print(f"\n{'='*60}")
        print(f"RUNNING WITH POPULATION SIZE = {size}")
        print(f"{'='*60}\n")
        for run in range(3):  # 3 runs per size
            print(f"--- Run {run+1} ---")
            genetic_algorithm(
                cost_func=levy_function,
                search_range=(-10, 10),
                population_size=size,
                mutation_rate=0.4,
                elitism_rate=0.1,
                precision=16,
                max_generations=400
            )

        plot_convergence()
        # Reset for next population size
        all_avg_list.clear()
        all_best_list.clear()
        generations_list.clear()
