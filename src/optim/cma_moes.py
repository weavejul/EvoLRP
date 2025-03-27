# optim/cma_es.py
import cma, comocma
import numpy as np
import torch

class CMAMOESOptimizer:
    def __init__(self, num_params, init_sigma=0.5, population_size=20, 
                 init_x=None, bounds=None, maxiter=100, tolx=1e-12, tolfun=1e-12, multi_opt=False):
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
        num_kernels = 5
        
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
    
        self.solvers = comocma.get_cmas(
            [init_x.copy() for _ in range(num_kernels)],  # Initial points
            init_sigma,
            opts
        )
        self.es = comocma.Sofomore(
            self.solvers, 
            reference_point=[0, 0],  # TODO: Worst case acceptable objective values?
        )
    

    def step(self, fitness_function1, fitness_function2):
        """
        Runs one generation of CMA-ES.
        
        Args:
            fitness_function: Function that evaluates solution fitness
        """
        combined_fitness = comocma.FitFun(fitness_function1, fitness_function2)
        solutions = self.es.ask()
        
        scores = [torch.tensor(combined_fitness(sol)).detach().cpu() if torch.is_tensor(combined_fitness(sol)) else torch.tensor(combined_fitness(sol)) 
                for sol in solutions]
        self.es.tell(solutions, scores)


    def best_solution(self):
        """
        Returns the best-found solution from multi-objective optimization.
        
        For multi-objective optimization, this returns a solution from the Pareto front.
        We can use different strategies to select one solution:
        1. Return the solution with the best hypervolume contribution
        2. Return a knee point on the Pareto front
        3. Return a solution with balanced objectives
        """
        # Check if we're using multi-objective optimization
        if hasattr(self.es, 'pareto_front') and self.es.pareto_front is not None:
            # If we have a Pareto front, we need to select one solution
            if len(self.es.pareto_front) == 0:
                # If Pareto front is empty, return a random solution
                return self.es.ask()[0]
            
            # Option 1: Return the solution with highest hypervolume contribution
            # This is often a good compromise solution
            if hasattr(self.es, 'best_hypervolume_solution'):
                return self.es.best_hypervolume_solution
            
            # Option 2: Return the first solution in the Pareto front
            # This is a simple fallback
            return self.es.pareto_front[0][0]  # [0][0] because pareto_front contains [solution, fitness] pairs
        
        # For single-objective optimization or if no Pareto front is available
        # Try different attributes that might contain the best solution
        # if hasattr(self.es, 'result') and hasattr(self.es.result, 'xbest'):
        #     return self.es.result.xbest
        if hasattr(self.es, 'xbest'):
            return self.es.xbest
        elif hasattr(self.es, 'best_solution'):
            return self.es.best_solution
        
        # If we can't find a best solution attribute, return the best from the current population
        # This is a fallback option
        solutions = self.es.ask()
        if solutions:
            return solutions[0]  # Just return the first solution as a fallback
        
        # If all else fails, return None
        return None