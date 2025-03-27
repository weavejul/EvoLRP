import torch
import numpy as np
from .complexity import sparseness
from .faithfulness import pixel_flipping, faithfulness_correlation
from .robustness import average_sensitivity
from lrp import UniformRuleLRP, UniformValueLRP
import traceback

def create_fitness_function(model, images, labels=None, criterion='sparseness', rule_type='alpha_beta', 
                          weights=(0.7, 0.3), baseline='zero', num_steps=5, correct_class_only=False,
                          optimization_type='UniformRule'):
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
        optimization_type: Type of optimization to perform ('UniformRule' or 'UniformValue')
    
    Returns:
        Function that takes a parameter set and returns a fitness score
    """
    # For UniformValueLRP, we'll encode rule types as integers in the parameters
    rule_mapping = {
        0: 'alpha_beta',
        1: 'epsilon',
        2: 'gamma',
        3: 'zero'
    }
    
    # Get initial default rule values
    default_rule_values = {
        'alpha_beta': 2.0, 
        'epsilon': 0.1,
        'gamma': 1.0,
        'zero': 0.0
    }
    
    # Get number of layers in the model
    num_layers = 0
    if hasattr(model, 'features'):
        num_layers += len(list(model.features))
    if hasattr(model, 'classifier'):
        num_layers += len(list(model.classifier))
    
    # Do a single forward pass to collect MaxPool information
    # This is important for UniformValueLRP to track layer-specific MaxPool indices
    if optimization_type == 'UniformValue':
        # Use default rule type for initial forward pass
        temp_rule_types = [rule_type] * num_layers
        temp_lrp = UniformValueLRP(model, rule_types=temp_rule_types, rule_values=default_rule_values, 
                                 correct_class_only=correct_class_only)
        
        # Just do a forward pass to collect MaxPool indices, we don't need the actual relevance
        with torch.no_grad():
            temp_lrp.compute_relevance(images, labels if correct_class_only else None)
        
        # Save the MaxPool indices for reuse
        maxpool_info = temp_lrp.max_pool_indices
        
        # Clean up
        del temp_lrp
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def fitness_function(params):
        try:
            if optimization_type == 'UniformRule':
                # Handle UniformRuleLRP (original behavior)
                # Constrain parameters to be positive for alpha, with lower bound depending on layer type
                constrained_params = []
                for i, p in enumerate(params):
                    min_val = 1.0
                    
                    # Constrain the parameter with layer-specific minimum
                    constrained_params.append(max(min_val, abs(p) + 1.0))
                
                constrained_params = torch.tensor(constrained_params)
                # constrained_params = torch.ones_like(constrained_params) * 2.0
                # print(constrained_params)
                
                # Create LRP with constrained parameters
                lrp = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_params, 
                                  correct_class_only=correct_class_only)
                
            elif optimization_type == 'UniformValue':
                # Handle UniformValueLRP optimization
                # In this case, params encodes which rule type to use for each layer
                rule_types = []
                for i, p in enumerate(params):
                    # Map the continuous parameter to one of the four rule types
                    # Use modulo 4 to handle any real number
                    # rule_index = int(abs(p) % 4)

                    rule_index = int((p * 4) % 4)
    
                    rule_types.append(rule_mapping[rule_index])

                
                # Use default rule values
                rule_values = default_rule_values.copy()
                
                # Create LRP with the selected rule types
                lrp = UniformValueLRP(model, rule_types=rule_types, rule_values=rule_values,
                                   correct_class_only=correct_class_only)
                
                # Reuse the MaxPool indices from the initial forward pass
                if 'maxpool_info' in locals():
                    lrp.max_pool_indices = maxpool_info
                
            else:
                raise ValueError(f"Unsupported optimization type: {optimization_type}")
            
            with torch.no_grad():  # No need for gradients during LRP
                relevance_map = lrp.compute_relevance(images, labels if correct_class_only else None)
            
            # Compute scores based on the chosen criterion
            batch_scores = []
            
            if criterion == 'sparseness':
                # Calculate sparseness for this image
                batch_scores = sparseness(
                    relevance_map  # Get individual relevance map but keep batch dim
                )
                # For sparseness, higher is better
            
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
                    baseline=baseline,
                    attr_model=lrp,
                    method='lrp'
                )
                # For average_sensitivity, lower is better (less sensitivity to perturbations)
            
            elif criterion == 'faithfulness':
                if labels is None:
                    raise ValueError("Labels must be provided for faithfulness criterion")
                # Calculate faithfulness score for this image
                kwargs = {
                    'subset_size': min(100, relevance_map.shape[-1]//10), # check this
                }

                batch_scores = faithfulness_correlation(
                    relevance_map,
                    model,
                    images,
                    labels,
                    num_steps=num_steps,
                    **kwargs
                )
                # For faithfulness, higher is better
            
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            # Use a weighted combination of mean and minimum for robust optimization
            mean_score = sum(batch_scores) / len(batch_scores)
            # print(batch_scores)
            # Determine if we want min or max based on the criterion
            if criterion == 'average_sensitivity':
                # For average_sensitivity, lower is better, so we want the maximum value (worst case)
                extreme_score = max(batch_scores)
                # We'll minimize this score
                sign = 1
            else:
                # For all other metrics (sparseness, pixel_flipping, faithfulness), higher is better
                # so we want the minimum value (worst case)
                extreme_score = min(batch_scores)
                # We'll maximize these scores
                sign = -1
            
            combined_score = weights[0] * mean_score + weights[1] * extreme_score
            
            # Return score with appropriate sign for optimization direction
            return sign * combined_score
            
        except Exception as e:
            print(f"Error in fitness function: {e}")
            print("Traceback:")
            print(traceback.format_exc())
            return float('inf')  # Return bad score on error
    
    return fitness_function
