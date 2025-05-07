import torch
import numpy as np
from .complexity import sparseness
from .faithfulness import pixel_flipping, faithfulness_correlation
from .robustness import average_sensitivity
from lrp import UniformRuleLRP, UniformValueLRP, UniformLayerLRP
import traceback
import itertools

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
        optimization_type: Type of optimization to perform ('UniformRule', 'UniformValue', or 'UniformLayer')
    
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
        'epsilon': 0.25,
        # 'gamma': 1.0,
        'zero': 0.0  # Zero rule doesn't use a parameter value
    }
    
    # Get number of layers in the model
    num_layers = 0
    if hasattr(model, 'features'):
        num_layers += len(list(model.features))
    if hasattr(model, 'classifier'):
        num_layers += len(list(model.classifier))
    
    # For UniformLayerLRP, identify the unique layer types in the model
    layer_types = set()
    if optimization_type == 'UniformLayer':
        # Collect all layer types in the model
        if hasattr(model, 'features'):
            for layer in model.features:
                layer_types.add(layer.__class__.__name__)
        if hasattr(model, 'classifier'):
            for layer in model.classifier:
                layer_types.add(layer.__class__.__name__)
        
        print(f"Detected layer types: {layer_types}")
        
        # Dictionary of layer type to possible rule combinations
        # For most layer types, consider the basic rule types
        # Some layer types like BatchNorm2d might only need simpler rules
        layer_rule_possibilities = {
            layer_type: ['alpha_beta', 'epsilon', 'zero'] 
            for layer_type in layer_types 
            if layer_type not in ['Dropout', 'ReLU', 'MaxPool2d', 'BatchNorm2d']
        }
        
        # For layers that are typically treated as identity in LRP or don't affect relevance as much
        for layer_type in layer_types:
            if layer_type in ['Dropout', 'ReLU', 'MaxPool2d', 'BatchNorm2d']:
                print(f"Using zero rule for non-important layer type: {layer_type}")
                layer_rule_possibilities[layer_type] = ['zero']  # Use zero rule for non-important layers
        
        print(f"Layer rule possibilities: {layer_rule_possibilities}")
        
        # Generate all possible combinations of rules for each layer type
        rule_combinations = list(itertools.product(*[[(layer_type, rule) for rule in rules] 
                                                  for layer_type, rules in layer_rule_possibilities.items()]))
        
        # Convert to dictionary format for each combination
        rule_combinations_dict = []
        for combo in rule_combinations:
            rule_dict = {layer_type: rule for layer_type, rule in combo}
            rule_combinations_dict.append(rule_dict)
        
        print(f"Generated {len(rule_combinations_dict)} rule combinations for testing")
        if len(rule_combinations_dict) <= 5:
            print(f"Rule combinations: {rule_combinations_dict}")
        else:
            print(f"First 3 combinations: {rule_combinations_dict[:3]}")
            print(f"Last 3 combinations: {rule_combinations_dict[-3:]}")
        
        # Store best configuration and its score
        best_rule_config = None
        best_score = float('-inf')  # Start with worst possible score
        
        # Test each rule combination briefly and find the best starting point
        print("Testing rule combinations to find best starting point")
        for i, rule_config in enumerate(rule_combinations_dict):
            # Create layer values based on the default values
            layer_values = {layer_type: default_rule_values[rule] for layer_type, rule in rule_config.items()}
            
            # Create LRP with this configuration
            lrp = UniformLayerLRP(model, layer_rule_mapping=rule_config, layer_value_mapping=layer_values,
                                correct_class_only=correct_class_only)
            
            # Evaluate on a subset of images if there are many
            subset_size = len(images)
            print(f"Evaluating rule combination {i+1}/{len(rule_combinations_dict)}")
            
            # Print a summary of the rule configuration
            rule_counts = {}
            for rule_type in set(rule_config.values()):
                rule_counts[rule_type] = list(rule_config.values()).count(rule_type)
            print(f"  Rule distribution: {rule_counts}")
            
            subset_images = images[:subset_size]
            subset_labels = labels[:subset_size] if labels is not None else None
            
            try:
                # Compute relevance map
                with torch.no_grad():
                    relevance_map = lrp.compute_relevance(subset_images, subset_labels if correct_class_only else None)
                
                # Compute scores based on criterion
                if criterion == 'sparseness':
                    scores = sparseness(relevance_map)
                    sign = 1  # Higher is better
                elif criterion == 'pixel_flipping':
                    scores = pixel_flipping(relevance_map, model, subset_images, subset_labels, num_steps=3)
                    sign = 1  # Higher is better
                elif criterion == 'average_sensitivity':
                    scores, _ = average_sensitivity(relevance_map, model, subset_images, subset_labels, 
                                                num_steps=3, baseline=baseline, attr_model=lrp)
                    sign = -1  # Lower is better
                elif criterion == 'faithfulness':
                    scores = faithfulness_correlation(relevance_map, model, subset_images, subset_labels, 
                                                  num_steps=3, subset_size=5)
                    sign = 1  # Higher is better
                
                # Calculate average score
                avg_score = sum(scores) / len(scores) * sign
                print(f"  Average score: {avg_score:.4f}")
                
                # Update best configuration if this one is better
                if avg_score > best_score:
                    best_score = avg_score
                    best_rule_config = rule_config
                    print(f"  New best rule config found (score: {avg_score:.4f})")
            
            except Exception as e:
                print(f"  Error testing rule config: {e}")
                continue
        
        # If no valid configuration was found, fall back to basic config
        if best_rule_config is None:
            print("No valid rule configuration found, falling back to defaults")
            best_rule_config = {}
            for layer_type in layer_types:
                if layer_type in ['Dropout', 'ReLU', 'MaxPool2d', 'BatchNorm2d']:
                    best_rule_config[layer_type] = 'zero'
                else:
                    best_rule_config[layer_type] = 'epsilon'
        
        # Store the best rule configuration for use in the fitness function
        selected_rule_config = best_rule_config
        print(f"Selected rule configuration: {selected_rule_config}")
    
    
    # Do a single forward pass to collect MaxPool information
    # This is important for UniformValueLRP and UniformLayerLRP to track layer-specific indices
    if optimization_type in ['UniformValue', 'UniformLayer']:
        if optimization_type == 'UniformValue':
            # Use default rule type for initial forward pass
            temp_rule_types = [rule_type] * num_layers
            temp_lrp = UniformValueLRP(model, rule_types=temp_rule_types, rule_values=default_rule_values, 
                                    correct_class_only=correct_class_only)
        else:  # UniformLayer
            # Use the selected rule configuration for initial forward pass
            layer_values = {layer_type: default_rule_values[rule] for layer_type, rule in selected_rule_config.items()}
            temp_lrp = UniformLayerLRP(model, layer_rule_mapping=selected_rule_config, 
                                    layer_value_mapping=layer_values,
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
                
            elif optimization_type == 'UniformLayer':
                # Handle UniformLayerLRP optimization
                # In this case, params contains values for each layer type
                
                # Map parameters to layer values
                layer_values = {}
                param_idx = 0
                for layer_type in selected_rule_config.keys():
                    rule = selected_rule_config[layer_type]
                    
                    # Get the appropriate value based on the rule type
                    if rule == 'alpha_beta':
                        # Alpha must be >= 1.0
                        value = max(1.0, abs(params[param_idx]) + 1.0)
                    elif rule == 'epsilon':
                        # Epsilon should be in a reasonable range
                        value = min(1.0, abs(params[param_idx]))
                    elif rule == 'gamma':
                        # Gamma should be positive and reasonable
                        value = max(0.1, min(3.0, abs(params[param_idx])))
                    else:  # Zero rule
                        value = 0.0
                    
                    layer_values[layer_type] = value
                    param_idx += 1
                
                # Create LRP with the selected rule types and optimized values
                lrp = UniformLayerLRP(model, layer_rule_mapping=selected_rule_config, 
                                   layer_value_mapping=layer_values,
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
    
    # For UniformLayerLRP, return the fitness function and the selected rule configuration
    if optimization_type == 'UniformLayer':
        return fitness_function, selected_rule_config
    else:
        return fitness_function
