import torch
import numpy as np
from .complexity import sparseness
from .faithfulness import pixel_flipping, faithfulness_correlation
from .robustness import average_sensitivity
from lrp import UniformRuleLRP
import traceback

def create_fitness_function(model, images, labels=None, criterion='sparseness', rule_type='alpha_beta', 
                          weights=(0.7, 0.3), baseline='zero', num_steps=5, correct_class_only=False):
    """
    Creates a fitness function for optimizing LRP parameters based on different evaluation criteria.
    
    Args:
        model: The neural network model to explain
        images: Batch of input images (B, C, H, W)
        labels: Labels corresponding to images (needed for pixel_flipping and average_sensitivity)
        criterion: Evaluation criterion to use ('sparseness', 'pixel_flipping', 'average_sensitivity')
        rule_type: LRP rule type ('epsilon', 'alpha_beta', 'gamma')
        weights: Tuple of (mean_weight, min_weight) for combining batch scores
        baseline: Baseline for average_sensitivity ('zero' or 'mean')
        num_steps: Number of steps for pixel_flipping and average_sensitivity
        correct_class_only: If True, calculate relevance only for the correct class
    
    Returns:
        Function that takes a parameter set and returns a fitness score
    """
    def fitness_function(params):
        try:
            # Constrain parameters to be positive for alpha, with lower bound depending on layer type
            constrained_params = []
            for i, p in enumerate(params):
                min_val = 1.0
                
                # Constrain the parameter with layer-specific minimum
                constrained_params.append(max(min_val, abs(p) + 1.0))
            
            constrained_params = torch.tensor(constrained_params)
            
            # Create LRP with constrained parameters
            lrp = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_params, 
                              correct_class_only=correct_class_only)
            
            with torch.no_grad():  # No need for gradients during LRP
                relevance_map = lrp.compute_relevance(images, labels if correct_class_only else None)
            
            # Compute scores based on the chosen criterion
            batch_scores = []
            
            # for i in range(images.shape[0]):
            if criterion == 'sparseness':
                # Calculate sparseness for this image
                batch_scores = sparseness(
                    relevance_map  # Get individual relevance map but keep batch dim
                )
                # For sparseness, higher is better
                # batch_scores.append(score)
            
            elif criterion == 'pixel_flipping':
                if labels is None:
                    raise ValueError("Labels must be provided for pixel_flipping criterion")
                # Calculate pixel flipping score for this image
                batch_scores = pixel_flipping(
                    relevance_map,
                    model,
                    images,
                    labels,
                    num_steps=num_steps
                )
                # For pixel_flipping, higher score is better (indicates larger drop in prediction)
                # batch_scores.append(score)
            
            elif criterion == 'average_sensitivity':
                if labels is None:
                    raise ValueError("Labels must be provided for average_sensitivity criterion")
                # Calculate average sensitivity for this image
                batch_scores, _ = average_sensitivity(
                    relevance_map,
                    model,
                    images,
                    labels,
                    num_steps=num_steps,
                    baseline=baseline
                )
                # For average_sensitivity, higher AUC is better
                # batch_scores.append(auc)
            elif criterion == 'faithfulness':
                if labels is None:
                    raise ValueError("Labels must be provided for faithfulness criterion")
                # Calculate faithfulness score for this image
                kwargs = {
                    'subset_size': 10, # check this
                }

                batch_scores = faithfulness_correlation(
                    relevance_map,
                    model,
                    images,
                    labels,
                    num_steps=num_steps,
                    **kwargs
                )
                # batch_scores.append(score)
            
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            # Use a weighted combination of mean and minimum for robust optimization
            mean_score = sum(batch_scores) / len(batch_scores)
            min_score = min(batch_scores)
            combined_score = weights[0] * mean_score + weights[1] * min_score
            
            # Return negated score for minimization
            sign = 1 if criterion == 'average_sensitivity' else -1
            return sign * combined_score
            
        except Exception as e:
            print(f"Error in fitness function: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            return float('inf')  # Return bad score on error
    
    return fitness_function
