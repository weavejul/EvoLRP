# optim/cma_es.py
import cma
import numpy as np
import torch

class CMAESOptimizer:
    def __init__(self, num_params, init_sigma=0.5, population_size=20, 
                 init_x=None, bounds=None, maxiter=100, tolx=1e-12, tolfun=1e-12):
        """
        CMA-ES optimizer for LRP parameter optimization.

        Args:
            num_params: Number of LRP parameters to optimize (N = number of layers)
            init_sigma: Initial step size for CMA-ES
            population_size: Number of candidates per generation
            init_x: Initial solution vector. If None, randomly initialized
            bounds: Tuple of (lower_bounds, upper_bounds) for parameters. If None, unbounded
            maxiter: Maximum number of iterations
            tolx: Tolerance on x-changes
            tolfun: Tolerance on function value changes
        """
        self.num_params = num_params
        
        # Set up initial solution
        if init_x is None:
            init_x = np.random.rand(num_params)
            
        # Configure CMA-ES options
        opts = {
            'popsize': population_size,
            'maxiter': maxiter,
            'tolx': tolx,
            'tolfun': tolfun
        }
        
        # Add bounds if specified
        if bounds is not None:
            lower, upper = bounds
            opts['bounds'] = [lower, upper]
            
        self.es = cma.CMAEvolutionStrategy(
            x0=init_x,
            sigma0=init_sigma,
            options=opts
        )

    def step(self, fitness_function):
        """
        Runs one generation of CMA-ES.
        
        Args:
            fitness_function: Function that evaluates solution fitness
        """
        solutions = self.es.ask()
        # Convert tensor scores to Python numbers
        scores = [fitness_function(sol).item() if torch.is_tensor(fitness_function(sol)) else fitness_function(sol) 
                 for sol in solutions]
        self.es.tell(solutions, scores)

    def best_solution(self):
        """
        Returns the best-found solution.
        """
        return self.es.result.xbest