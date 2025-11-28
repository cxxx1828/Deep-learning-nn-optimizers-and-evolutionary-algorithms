# -*- coding: utf-8 -*-
"""
Optimization Toolbox - Classical & Evolutionary Methods
=======================================================

A comprehensive collection of 1D and multi-dimensional optimization algorithms
implemented from scratch in pure Python (NumPy).

Includes:
- 1D Methods: Fibonacci, Golden Section, Newton-Raphson, Secant, Parabolic
- Evolutionary: Binary & Real-coded Genetic Algorithm
- Swarm Intelligence: Particle Swarm Optimization (PSO)
- Gradient-based: Steepest Descent, Momentum, Nesterov, AdaGrad, RMSProp, Adam

Author: Nina Dragićević
"""

import numpy as np
import random
import math
from scipy import linalg


# ================================
# 1. 1D OPTIMIZATION METHODS
# ================================

def fibonacci_sequence(n):
    """Generate nth Fibonacci number (F1 = F2 = 1)"""
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


def fibonacci_method(func, a, b, tol=1e-6):
    """Fibonacci search for unimodal function"""
    n = 1
    while (b - a) / tol > fibonacci_sequence(n):
        n += 1

    x1 = a + fibonacci_sequence(n - 2) / fibonacci_sequence(n) * (b - a)
    x2 = a + fibonacci_sequence(n - 1) / fibonacci_sequence(n) * (b - a)

    for _ in range(2, n):
        f1, f2 = func(x1), func(x2)
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + fibonacci_sequence(n - 3) / fibonacci_sequence(n - 1) * (b - a)
        else:
            a = x1
            x1 = x2
            x2 = a + fibonacci_sequence(n - 1) / fibonacci_sequence(n - 1) * (b - a)

    x_opt = (a + b) / 2
    return x_opt, func(x_opt), n


def golden_section_search(func, a, b, tol=1e-6):
    """Golden section search (0.618 ratio)"""
    phi = (1 + math.sqrt(5)) / 2
    c = (phi - 1)  # ~0.618

    x1 = a + (1 - c) * (b - a)
    x2 = a + c * (b - a)
    f1, f2 = func(x1), func(x2)
    iterations = 0

    while (b - a) > tol:
        iterations += 1
        if f1 <= f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + (1 - c) * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + c * (b - a)
            f2 = func(x2)

    return (a + b) / 2, func((a + b) / 2), iterations


def newton_raphson(func, dfunc, ddfunc, x0, tol=1e-8, max_iter=100):
    """Newton-Raphson method (requires 1st and 2nd derivatives)"""
    x = x0
    for i in range(max_iter):
        fx = dfunc(x)
        fxx = ddfunc(x)
        if abs(fxx) < 1e-12:
            break
        step = fx / fxx
        x_new = x - step
        if abs(x_new - x) < tol:
            return x_new, func(x_new), i + 1
        x = x_new
    return x, func(x), max_iter


def secant_method(func, dfunc, x0, x1, tol=1e-8, max_iter=100):
    """Secant method (approximates derivative)"""
    x_prev, x_curr = x0, x1
    for i in range(max_iter):
        f_prev = dfunc(x_prev)
        f_curr = dfunc(x_curr)
        if abs(f_curr - f_prev) < 1e-12:
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if abs(x_next - x_curr) < tol:
            return x_next, func(x_next), i + 1
        x_prev, x_curr = x_curr, x_next
    return x_curr, func(x_curr), max_iter


# ================================
# 2. GENETIC ALGORITHMS
# ================================

# Binary GA helpers
def binary_encode(value, min_val, max_val, precision):
    steps = (max_val - min_val) / (2**precision - 1)
    val = int(round((value - min_val) / steps))
    return format(val, f'0{precision}b')

def binary_decode(bits, min_val, max_val, precision):
    steps = (max_val - min_val) / (2**precision - 1)
    val = int(bits, 2)
    return val * steps + min_val


# Real-coded GA crossover & mutation
def arithmetic_crossover(parents):
    offspring = []
    for p1, p2 in parents:
        alpha = random.random()
        child1 = alpha * p1 + (1 - alpha) * p2
        child2 = (1 - alpha) * p1 + alpha * p2
        offspring.extend([child1, child2])
    return np.array(offspring)


def gaussian_mutation(population, rate=0.2, scale=1.0, bounds=None):
    mutated = population.copy()
    mask = np.random.random(population.shape) < rate
    noise = np.random.normal(0, scale, population.shape)
    mutated += mask * noise
    if bounds:
        np.clip(mutated, bounds[0], bounds[1], out=mutated)
    return mutated


# ================================
# 3. PARTICLE SWARM OPTIMIZATION
# ================================

class Particle:
    def __init__(self, position, velocity_scale=1.0):
        self.position = np.array(position)
        self.velocity = np.random.uniform(-1, 1, len(position)) * velocity_scale
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def update_velocity(self, global_best, w, c1, c2):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds=None):
        self.position += self.velocity
        if bounds:
            self.position = np.clip(self.position, bounds[0], bounds[1])


def pso(func, dim, bounds, n_particles=40, max_iter=300, w=0.7, c1=1.4, c2=1.4):
    particles = [Particle(np.random.uniform(bounds[0], bounds[1], dim)) for _ in range(n_particles)]
    global_best = particles[0].position.copy()
    global_fitness = float('inf')

    for _ in range(max_iter):
        for p in particles:
            fitness = func(p.position)
            if fitness < p.best_fitness:
                p.best_fitness = fitness
                p.best_position = p.position.copy()
            if fitness < global_fitness:
                global_fitness = fitness
                global_best = p.position.copy()
        for p in particles:
            p.update_velocity(global_best, w, c1, c2)
            p.update_position(bounds)

    return global_best, global_fitness


# ================================
# 4. GRADIENT-BASED METHODS
# ================================

def gradient_descent(gradf, x0, lr=0.01, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(max_iter):
        grad = gradf(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= lr * grad
        path.append(x.copy())
    return path


def momentum_descent(gradf, x0, lr=0.01, momentum=0.9, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    path = [x.copy()]
    for _ in range(max_iter):
        grad = gradf(x)
        if np.linalg.norm(grad) < tol:
            break
        v = momentum * v + lr * grad
        x -= v
        path.append(x.copy())
    return path


def adam_optimizer(gradf, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    path = [x.copy()]
    t = 0

    for _ in range(max_iter):
        t += 1
        g = gradf(x)
        if np.linalg.norm(g) < tol:
            break

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())

    return path


# ================================
# Example Usage
# ================================

if __name__ == "__main__":
    # Test function: f(x) = x^4 - 4x^2 + x
    def f(x): return x**4 - 4*x**2 + x
    def df(x): return 4*x**3 - 8*x + 1
    def d2f(x): return 12*x**2 - 8

    print("=== 1D Optimization Methods ===")
    print("Fibonacci:", fibonacci_method(f, -3, 2, 1e-6))
    print("Golden Section:", golden_section_search(f, -3, 2, 1e-6))
    print("Newton-Raphson:", newton_raphson(f, df, d2f, 0.0))
    print("Secant:", secant_method(f, df, -1.0, 1.0))

    print("\n=== Gradient Descent Variants ===")
    path = momentum_descent(df, x0=[1.5], lr=0.01, momentum=0.9)
    print(f"Momentum converged in {len(path)} steps to {path[-1]}")

    path = adam_optimizer(df, x0=[1.5], lr=0.1)
    print(f"Adam converged in {len(path)} steps to {path[-1]}")
