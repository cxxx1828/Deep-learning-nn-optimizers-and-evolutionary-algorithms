# Neuro-Inspired Optimization Toolbox
### Every optimizer that powers modern deep learning + cutting-edge evolutionary & swarm intelligence algorithms — 100% from scratch in pure Python

This repository is the ultimate collection of **optimization algorithms that drive both classical deep learning and neuroevolution**, meticulously re-implemented from the ground up — no PyTorch, no TensorFlow, no black boxes.

Whether you're training a 100-billion-parameter LLM with **AdamW**, evolving neural network architectures with **NEAT-style Genetic Algorithms**, or performing hyper-parameter search using **Golden Section**, this toolbox contains the **exact mathematical core** of every method — with crystal-clear code, detailed comments, and convergence visualizations.

### What’s Inside

| Category                    | Algorithms Included                                                                                   | Real-World Use Case                                      |
|----------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| First & Second-Order       | Newton-Raphson • Secant • Parabolic interpolation • Fibonacci • Golden Section Search                 | Hyper-parameter tuning, 1D optimization                   |
| Gradient-Based (Backprop)  | Vanilla SGD • Momentum • Nesterov Accelerated Gradient • AdaGrad • RMSProp • Adam • AdamW (bias-corrected) | Training any neural network from scratch                 |
| Swarm Intelligence         | Particle Swarm Optimization (PSO) with linear inertia & dynamic c1/c2                                 | Weight optimization, non-differentiable objectives       |
| Evolutionary Computation  | Real-coded Genetic Algorithm (arithmetic crossover, Gaussian mutation) <br> Binary Genetic Algorithm (1/2-point crossover, inversion mutation) | Neuroevolution, architecture search, robust global search |
| Advanced Adaptive Methods  | AdaDelta • Nadam • AMSGrad • Lookahead (experimental)                                                 | State-of-the-art deep learning training                  |

### Why This Repository Exists

- You want to **truly understand** how Adam’s bias correction works — not just call `torch.optim.Adam`
- You’re building **Neuroevolution** systems (evolving topologies, weights, or hyperparameters)
- You need **global optimization** on noisy, multimodal, or non-differentiable fitness landscapes
- You’re writing a **thesis, paper, or blog post** and want beautiful, reproducible results
- You want to impress recruiters with a **world-class, from-scratch ML portfolio project**

All algorithms include:
- Clean, heavily commented code
- Convergence history tracking
- Professional matplotlib visualizations
- Side-by-side comparison notebooks (Levy, Ackley, Rastrigin, Sphere, etc.)
- Ready-to-use examples and benchmarks
