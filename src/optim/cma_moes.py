# optim/cma_es.py
import cma, comocma
import numpy as np
import torch

class CMAMOESOptimizer:
    def __init__(self, num_params, init_sigma=0.5, population_size=20, 
                 init_x=None, bounds=None, maxiter=100, tolx=1e-12, tolfun=1e-12, metrics=["sparseness","faithfulness"]):
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
        num_kernels = 10
        self.metrics = metrics
        
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
            reference_point=[-0, -0],  
        )
    

    def step(self, fitness_function1, fitness_function2):
        """
        Runs one generation of CMA-ES.
        
        Args:
            fitness_function: Function that evaluates solution fitness
        """
        original_fitness_function2 = fitness_function2

        if self.metrics[1] == "faithfulness":
            min_val = -1.0  # Replace with your actual minimum value
            
            # Negate and normalize to be between -1 and 0
            fitness_function2 = lambda x: -original_fitness_function2(x) / min_val
        elif self.metrics[1] == "average_sensitivity":
            fitness_function2 = lambda x: -(1-original_fitness_function2(x))
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
        # Check if we have solutions in the Pareto front
        if hasattr(self.es, 'pareto_set_cut') and len(self.es.pareto_set_cut) > 0:
            # Get the Pareto front solutions and their fitness values
            pareto_solutions = self.es.pareto_set_cut
            pareto_fitness = self.es.pareto_front_cut
            
            # You can analyze or visualize the Pareto front
            # For example, plot the Pareto front
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter([f[0] for f in pareto_fitness], [f[1] for f in pareto_fitness])
            # plt.xlabel('Objective 1')
            # plt.ylabel('Objective 2')
            # plt.title('Pareto Front')
            # plt.grid(True)
            # plt.show()
            # For negative metrics, find the solution closest to [-1000, -1000]
            # This prioritizes solutions that are strongly negative in both objectives
            target_point = np.array([-1000, -1000])
            distances = np.sqrt(np.sum(np.square(pareto_fitness - target_point), axis=1))
            best_idx = np.argmin(distances)
            
            return pareto_solutions[best_idx]
        
        # If no Pareto front is available yet, return the best solution from the current population
        solutions = self.es.ask()
        if solutions:
            # We don't have fitness values here, so just return the first solution
            return solutions[0]
        
        # If all else fails, return None
        return None