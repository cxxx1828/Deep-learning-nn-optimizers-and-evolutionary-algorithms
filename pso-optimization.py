# -*- coding: utf-8 -*-
"""
Particle Swarm Optimization (PSO) from Scratch in Python
=======================================================

A clean, efficient, and fully commented implementation of the classic PSO algorithm
with linear decreasing inertia weight and cognitive/social coefficients.

Tested on:
- Ackley function (multimodal, difficult)
- Griewank function (many local minima)

Author: Educational Implementation (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# ================================
# Configuration & Options
# ================================

class PSOOptions:
    def __init__(self):
        self.n_particles = 50        # Number of particles in swarm
        self.n_iterations = 300      # Maximum number of iterations
        self.w_start = 0.9           # Initial inertia weight
        self.w_end = 0.4             # Final inertia weight
        self.c1_start = 2.5          # Initial cognitive coefficient
        self.c1_end = 0.5            # Final cognitive coefficient
        self.c2_start = 0.5          # Initial social coefficient
        self.c2_end = 2.5            # Final social coefficient
        self.v_max = None            # Absolute velocity limit (optional)
        self.bounds = None           # (min, max) bounds for position clamping


# ================================
# Particle Class
# ================================

class Particle:
    def __init__(self, position, options):
        self.dim = len(position)
        self.position = np.array(position, dtype=float)
        self.velocity = np.random.uniform(-1, 1, self.dim) * 0.5  # Small random initial velocity
        
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')

        self.options = options

    def evaluate(self, cost_function):
        """Evaluate current fitness and update personal best"""
        self.fitness = cost_function(self.position)
        
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_position = self.position.copy()

    def update_velocity(self, global_best_position, iteration, max_iterations):
        """Update velocity using linearly decreasing coefficients"""
        # Linearly decreasing inertia
        w = self.options.w_end + (self.options.w_start - self.options.w_end) * \
            (max_iterations - iteration) / max_iterations
        
        # Linearly changing cognitive and social factors
        c1 = self.options.c1_end + (self.options.c1_start - self.options.c1_end) * \
             (max_iterations - iteration) / max_iterations
        c2 = self.options.c2_end + (self.options.c2_start - self.options.c2_end) * \
             (max_iterations - iteration) / max_iterations

        r1, r2 = np.random.rand(2)

        # Cognitive component (personal best)
        cognitive = c1 * r1 * (self.best_position - self.position)
        # Social component (global best)
        social = c2 * r2 * (global_best_position - self.position)

        # Update velocity
        self.velocity = w * self.velocity + cognitive + social

        # Optional: velocity clamping
        if self.options.v_max is not None:
            speed = np.linalg.norm(self.velocity)
            if speed > self.options.v_max:
                self.velocity = self.velocity / speed * self.options.v_max

    def update_position(self):
        """Update position and apply bounds if specified"""
        self.position += self.velocity
        
        if self.options.bounds is not None:
            low, high = self.options.bounds
            self.position = np.clip(self.position, low, high)


# ================================
# Main PSO Class
# ================================

class ParticleSwarmOptimization:
    def __init__(self, cost_function, dim, options=None):
        self.cost_function = cost_function
        self.dim = dim
        self.options = options or PSOOptions()

        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.history = []  # Store best fitness per iteration

    def initialize_swarm(self, bounds=None):
        """Initialize particles randomly within bounds"""
        low = -10 if bounds is None else bounds[0]
        high = 10 if bounds is None else bounds[1]
        
        for _ in range(self.options.n_particles):
            position = np.random.uniform(low, high, self.dim)
            particle = Particle(position, self.options)
            self.swarm.append(particle)

    def run(self, bounds=None):
        """Main PSO optimization loop"""
        self.options.bounds = bounds
        self.initialize_swarm(bounds)

        print("Starting PSO optimization...")
        print(f"{'Iter':>5} | {'Best Fitness':>15} | {'Position'}")

        for iteration in range(self.options.n_iterations):
            current_best_fitness = float('inf')
            current_best_position = None

            # Evaluate all particles
            for particle in self.swarm:
                particle.evaluate(self.cost_function)

                if particle.fitness < current_best_fitness:
                    current_best_fitness = particle.fitness
                    current_best_position = particle.position.copy()

                # Update global best
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()

            # Update velocities and positions
            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, iteration, self.options.n_iterations)
                particle.update_position()

            self.history.append(self.global_best_fitness)

            if iteration % 20 == 0 or iteration == self.options.n_iterations - 1:
                pos_str = ", ".join([f"{x:8.4f}" for x in self.global_best_position[:3]])
                if self.dim > 3:
                    pos_str += " ..."
                print(f"{iteration:5d} | {self.global_best_fitness:15.8f} | [{pos_str}]")

        print("\nOptimization complete!")
        print(f"Global Best Fitness: {self.global_best_fitness:.10f}")
        print(f"Global Best Position: {self.global_best_position}")
        return self.global_best_position, self.global_best_fitness


# ================================
# Test Functions
# ================================

def ackley(x):
    """Ackley function - global minimum at f(0,0,...,0) = 0"""
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.exp(1)


def griewank(x):
    """Griewank function - global minimum at f(0,0,...,0) = 0"""
    sum_part = np.sum(x**2) / 4000
    prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return 1 + sum_part - prod_part


def sphere(x):
    """Simple sphere function"""
    return np.sum(x**2)


def rastrigin(x):
    """Rastrigin function - many local minima"""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


# ================================
# Visualization (Optional)
# ================================

def plot_surface(func, title, xlim=(-10, 10), ylim=(-10, 10)):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[func([xi, yi]) for xi in x] for yi in y])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    plt.colorbar(surf, shrink=0.5)
    plt.show()


# ================================
# Run Examples
# ================================

if __name__ == "__main__":
    # Customize options
    options = PSOOptions()
    options.n_particles = 80
    options.n_iterations = 400

    print("=== Testing PSO on Ackley Function (5D) ===")
    pso = ParticleSwarmOptimization(cost_function=ackley, dim=5, options=options)
    best_pos, best_val = pso.run(bounds=(-32, 32))

    print("\n=== Testing PSO on Griewank Function (5D) ===")
    pso2 = ParticleSwarmOptimization(cost_function=griewank, dim=5, options=options)
    best_pos2, best_val2 = pso2.run(bounds=(-600, 600))

    # Optional: Plot convergence
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pso.history)
    plt.title('Ackley - Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.plot(pso2.history)
    plt.title('Griewank - Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()
