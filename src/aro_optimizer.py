import numpy as np
import random
import torch

class AROOptimizer:
    """
    Artificial Rabbits Optimization (ARO)
    Simplified but functional version:
    - Each rabbit = vector of hyperparameters
    - Fitness = validation loss from train.py
    """

    def __init__(self, num_rabbits, search_space, fitness_func, iterations=10):
        """
        search_space: dict with ranges
            {
                "hidden_dim": (16, 64),
                "num_layers": (1, 3),
                "learning_rate": (0.0005, 0.01)
            }
        fitness_func: function(hparams) -> float
        """
        self.num_rabbits = num_rabbits
        self.search_space = search_space
        self.fitness_func = fitness_func
        self.iterations = iterations

        self.population = self._init_population()

    def _init_population(self):
        pop = []
        for _ in range(self.num_rabbits):
            hparams = {
                "hidden_dim": random.randint(*self.search_space["hidden_dim"]),
                "num_layers": random.randint(*self.search_space["num_layers"]),
                "learning_rate": random.uniform(*self.search_space["learning_rate"])
            }
            pop.append(hparams)
        return pop

    def _explore(self, rabbit):
        """Exploration step: random jump in search space"""
        new_rabbit = rabbit.copy()
        key = random.choice(list(self.search_space.keys()))

        if key == "hidden_dim" or key == "num_layers":
            new_rabbit[key] = random.randint(*self.search_space[key])
        else:
            new_rabbit[key] = random.uniform(*self.search_space[key])

        return new_rabbit

    def _exploit(self, rabbit, best_rabbit):
        """Exploitation step: move slightly toward best solution"""
        new_rabbit = rabbit.copy()
        for key in rabbit:
            if key == "hidden_dim" or key == "num_layers":
                step = random.choice([-1, 1])
                new_rabbit[key] += step
                # clamp
                low, high = self.search_space[key]
                new_rabbit[key] = int(np.clip(new_rabbit[key], low, high))
            else:
                # learning_rate
                diff = best_rabbit[key] - rabbit[key]
                new_rabbit[key] += 0.3 * diff

                # clamp
                low, high = self.search_space[key]
                new_rabbit[key] = float(np.clip(new_rabbit[key], low, high))

        return new_rabbit

    def optimize(self):
        """Main ARO loop"""
        best_rabbit = None
        best_loss = float("inf")

        for it in range(self.iterations):
            new_population = []

            for rabbit in self.population:
                # random choice: explore or exploit
                if random.random() < 0.5:
                    candidate = self._explore(rabbit)
                else:
                    candidate = (
                        self._exploit(rabbit, best_rabbit)
                        if best_rabbit is not None
                        else self._explore(rabbit)
                    )

                # evaluate candidate
                loss = self.fitness_func(candidate)

                # update best solution
                if loss < best_loss:
                    best_loss = loss
                    best_rabbit = candidate

                new_population.append(candidate)

            self.population = new_population
            print(f"[Iteration {it+1}/{self.iterations}] Best loss: {best_loss:.6f}")

        return best_rabbit, best_loss
