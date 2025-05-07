"""
Script for testing the LRP model. Runs a given LRP model on the full 
dataset in input/ and computes the quantus metrics for the samples
that are correctly classified by the model. Outputs the mean and standard deviation
of the metrics.
"""
import argparse
import time
import pathlib
from captum.attr import LayerAttribution, LayerGradCam, Lime, IntegratedGradients, LRP, GuidedGradCam
from skimage.segmentation import slic

import torch
from torchvision.models import vgg16, VGG16_Weights

from datasets import get_mnist_data, get_imagenet_data
from models import load_model
from lrp import UniformRuleLRP, UniformLayerLRP
from evaluation import sparseness, pixel_flipping, average_sensitivity, faithfulness_correlation
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import os
import time
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

def compute_scores(model, x_batch, y_batch, a_batch, num_steps, baseline, criterion, 
    attr_model=None, method='lrp', feature_masks=None):
    # x_batch = x_batch.detach().cpu().numpy()
    # y_batch = y_batch.detach().cpu().numpy()
    if criterion == 'sparseness':
        # Calculate sparseness for this image
        batch_scores = sparseness(a_batch)
        # For sparseness, higher is better  
    elif criterion == 'pixel_flipping':
        # Calculate pixel flipping score for this image
        batch_scores = pixel_flipping(
            a_batch,  
            model,
            x_batch,
            y_batch,
            num_steps=num_steps
        )
        # For pixel_flipping, higher score is better (indicates larger drop in prediction)
    elif criterion == 'average_sensitivity':
        batch_scores, _ = average_sensitivity(
            a_batch,
            model,
            x_batch,
            y_batch,
            num_steps=num_steps,
            baseline=baseline,
            attr_model=attr_model,
            method=method,
            feature_masks=feature_masks
        )
        # print(batch_scores)
        # For average_sensitivity, higher AUC is better
    elif criterion == 'faithfulness':
        kwargs = {
            'subset_size': 10, # check this
        }
        batch_scores = faithfulness_correlation(
            a_batch,
            model,
            x_batch,
            y_batch,
            num_steps=num_steps,
            **kwargs
        )
        # print(batch_scores)
        # higher is better
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    # Use a weighted combination of mean and minimum for robust optimization
    mean_score = sum(batch_scores) / len(batch_scores)
    
    return batch_scores, mean_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="/home/neuro/XAIProject/data/ImageNet",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=8, type=int
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        help="Device.",
        choices=["gpu", "cpu"],
        default="gpu",
        type=str,
    )

    parser.add_argument(
        "-m",
        "--mode",
        dest="mode",
        help="Mode",
        default="uniform_rule",
        choices=["uniform_rule", "uniform_value", "layer_type", "composite_layer"],
        type=str,
    )

    parser.add_argument(
        "-r",
        "--rule_name",
        dest="rule_name",
        help="rule name",
        default="epsilon",
        type=str,
    )

    parser.add_argument(
        "-x",
        "--xai-method",
        dest="xai_method",
        help="XAI method",
        default="IntegratedGradients",
        type=str,
        choices=["lrp", "LayerGradCam", "Lime", "IntegratedGradients", "LRP"]
    )
    parser.add_argument(
        "-c",
        "--criterion",
        dest="criterion",
        help="Criterion",
        default="sparseness",
        type=str,
        choices=["sparseness", "pixel_flipping", "average_sensitivity", "faithfulness"]
    )

    parser.add_argument(
        "-p",
        "--lrp-params",
        dest="lrp_params",
        help="LRP parameters",
        default="saved_params/opt_alpha_beta_faithfulness.json",
        type=str
    )

    config = parser.parse_args()

    pathlib.Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("Mode: ", config.mode)
    print("Rule name: ", config.rule_name)
    print("Batch size: ", config.batch_size)
    print("Device: ", config.device)
    print("Input dir: ", config.input_dir)
    print("Output dir: ", config.output_dir)
    print("XAI method: ", config.xai_method)
    print("Criterion: ", config.criterion)
    # set up parameters
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")
    conv_layer = None
    if "ImageNet" in config.input_dir:
        data_loader = get_imagenet_data(config)
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        conv_layer = model.features[-3]
    else:
        data_loader = get_mnist_data(batch_size=config.batch_size)
        model = load_model('models/mnist_cnn.pth')
        conv_layer = model.features[-4]
    model.to(device)
    model.eval()

    time_start = time.time()
    # set up LRP model
    # Load parameters from JSON files
    lrp_model = None

    if config.xai_method == "lrp":
        with open(config.lrp_params, 'r') as f:
            lrp_params = torch.tensor(json.load(f))

        if config.mode == "uniform_rule":
            lrp_model = UniformRuleLRP(model, rule_type=config.rule_name, rule_params=lrp_params)
        # elif config.mode == "uniform_value":
        #     lrp_model = LRPModel.from_uniform_value(model=model, per_layer_rules=config.per_layer_rules)
        # elif config.mode == "layer_type":
        #     lrp_model = LRPModel.from_layer_type(model=model, layer_type_config=config.layer_type_config)
        elif config.mode == "composite_layer":
            lrp_model = UniformLayerLRP(
                model, 
                layer_rule_mapping={'Conv2d': 'zero', 'Linear': 'alpha_beta', 'MaxPool2d': 'zero', 'ReLU': 'zero', 'Dropout': 'zero'},
                layer_value_mapping={'Conv2d': 4.8342, 'Linear': 1.0, 'MaxPool2d': 0.0, 'ReLU': 0.0, 'Dropout': 0.0},
                correct_class_only=True
            )
        else:
            raise ValueError(f"Invalid mode: {config.mode}")

    # set up XAI methods
    lime = Lime(model)
    ig = IntegratedGradients(model)
    gradcam = GuidedGradCam(model, conv_layer)
    lrp = LRP(model)

    scores = np.array([])
    # Store detailed score information
    score_details = []
    
    for batch_idx, (x, y) in enumerate(data_loader):
        # Print GPU memory usage
        # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
        # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9} GB")
            
        x = x.to(device)
        y = y.to(device) 

        # remove mislabeled images
        with torch.no_grad():  # Add no_grad here to prevent memory leaks during inference
            y_pred = model(x)
            y_pred_classes = torch.argmax(y_pred, dim=1)  # Get the predicted classes
        
        # Create a mask of correct predictions
        correct_mask = (y_pred_classes == y)
        
        # Keep only the correct predictions
        x = x[correct_mask]
        y = y[correct_mask]
        # Store indices of correct predictions
        correct_indices = torch.where(correct_mask)[0].cpu().numpy()
        
        # Check if we have any samples left
        if len(x) == 0:
            print("Warning: All predictions were incorrect in this batch")
            continue
        try:  # Add try-except to handle potential OOM errors gracefully
            feature_masks = None
            attr_model = None
            if config.xai_method == "lrp":
                a_batch = lrp_model.compute_relevance(x, y)
                attr_model = lrp_model
            elif config.xai_method == "LayerGradCam":
                a_batch = gradcam.attribute(
                    x,
                    target=y
                )
                # Use a more sophisticated interpolation method
                # a_batch = LayerAttribution.interpolate(
                #     gradcam_attr, 
                #     x.shape[2:],
                #     interpolate_mode='bicubic'  # Use bicubic interpolation instead of default
                # )
                attr_model = gradcam
            elif config.xai_method == "Lime":
                # Create feature masks for each image in the batch
                feature_masks = []
                for img in x:
                    # Convert single image to numpy format (C, H, W) -> (H, W, C)
                    x_numpy = img.cpu().permute(1, 2, 0).numpy()
                    feature_mask = slic(x_numpy, n_segments=100, compactness=10, sigma=0.5)
                    feature_masks.append(feature_mask)
                
                # Stack all feature masks and move to device
                feature_masks = torch.tensor(np.stack(feature_masks)).to(device)
                a_batch = lime.attribute(
                    x,
                    target=y,
                    feature_mask=feature_masks.unsqueeze(1),
                    n_samples=40,
                    perturbations_per_eval=16,
                    show_progress=False
                )   
                attr_model = lime
            elif config.xai_method == "IntegratedGradients":
                a_batch, _ = ig.attribute(
                    x,
                    baselines=0,
                    target=y,
                    method='gausslegendre',
                    return_convergence_delta=True
                )
                attr_model = ig
            elif config.xai_method == "LRP":
                a_batch = lrp.attribute(x, y)
                attr_model = lrp
            else:
                raise ValueError(f"Invalid XAI method: {config.xai_method}")

            if x.shape[0] != a_batch.shape[0]:
                a_batch = a_batch.unsqueeze(0)
            
            # Normalize attributions if using LIME before metric calculation
            if config.xai_method == "Lime":
                # Normalize LIME attributions to have similar scale as other methods
                # First ensure no NaN or Inf values
                a_batch = torch.nan_to_num(a_batch, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Apply min-max normalization per attribution map
                for i in range(a_batch.shape[0]):
                    attr_min = a_batch[i].min()
                    attr_max = a_batch[i].max()
                    if attr_max > attr_min:  # Avoid division by zero
                        a_batch[i] = (a_batch[i] - attr_min) / (attr_max - attr_min)
                        # Scale to typical range for other methods
                        a_batch[i] = a_batch[i] * 2.0 - 1.0  # Scale to [-1, 1] range
            
            # Get individual scores instead of just the mean
            individual_scores, batch_mean_score = compute_scores(
                model, x, y, a_batch, num_steps=5, baseline='zero', criterion=config.criterion,
                attr_model=attr_model, method=config.xai_method, feature_masks=feature_masks
            )
            
            # Convert to numpy if it's a tensor
            if isinstance(individual_scores, torch.Tensor):
                individual_scores = individual_scores.detach().cpu().numpy()
            
            # Store each sample's information
            for i, score in enumerate(individual_scores):
                if not np.isnan(score):
                    sample_idx = correct_indices[i]
                    label = y[i].item()
                    # Store score with metadata
                    score_details.append({
                        'score': float(score),
                        'batch_idx': batch_idx,
                        'sample_idx': int(sample_idx),
                        'label': int(label),
                        'pred_conf': float(torch.softmax(y_pred[correct_mask][i], dim=0).max().item())
                    })
            
            if isinstance(batch_mean_score, torch.Tensor):
                batch_mean_score = batch_mean_score.detach().cpu().numpy().item()
            if batch_mean_score is None:
                batch_mean_score = 0
            scores = np.append(scores, batch_mean_score)
            
            # Explicitly delete tensors to free memory
            del a_batch, x, y, y_pred, y_pred_classes, individual_scores, batch_mean_score
                
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA OOM in batch {i}, skipping and clearing memory...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                break
            else:
                raise e
                
    # Add checks before calculating statistics
    if scores.size == 0:
        print("Warning: No valid scores were collected")
        mean_score = float('nan')
        std_score = float('nan')
    else:
        # Filter out any NaN values and values greater than 5 before calculating statistics
        valid_scores = scores[~np.isnan(scores)]
        valid_scores = valid_scores[valid_scores < 5]
        if valid_scores.size == 0:
            print("Warning: All scores were NaN or greater than 10")
            mean_score = float('nan')
            std_score = float('nan')
        else:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
    
    print(f"Number of valid scores: {scores.size}")
    print(f"Mean score: {mean_score}")
    print(f"Standard deviation: {std_score}")

    # Process and display the top and bottom scores
    if score_details:
        # Sort by score (ascending)
        sorted_scores = sorted(score_details, key=lambda x: x['score'])
        
        # Get bottom 10 (lowest) scores
        bottom_10 = sorted_scores[:10] if len(sorted_scores) >= 10 else sorted_scores
        
        # Get top 10 (highest) scores
        top_10 = sorted_scores[-20:][::-1] if len(sorted_scores) >= 20 else sorted_scores[::-1]
        
        # Display the results
        print("\n=== Top 10 Highest Scores ===")
        print(f"{'Score':<10} {'Batch':<8} {'Sample':<8} {'Label':<8} {'Pred Conf':<10}")
        print("-" * 50)
        for item in top_10:
            print(f"{item['score']:<10.4f} {item['batch_idx']:<8} {item['sample_idx']:<8} {item['label']:<8} {item['pred_conf']:<10.4f}")
        
        print("\n=== Top 10 Lowest Scores ===")
        print(f"{'Score':<10} {'Batch':<8} {'Sample':<8} {'Label':<8} {'Pred Conf':<10}")
        print("-" * 50)
        for item in bottom_10:
            print(f"{item['score']:<10.4f} {item['batch_idx']:<8} {item['sample_idx']:<8} {item['label']:<8} {item['pred_conf']:<10.4f}")
        
        # Optionally, save the detailed results to a JSON file in the output directory
        result_file = os.path.join(config.output_dir, f"{config.xai_method}_{config.criterion}_detailed_results.json")
        with open(result_file, 'w') as f:
            json.dump({
                'top_scores': top_10,
                'bottom_scores': bottom_10,
                'mean_score': float(mean_score),
                'std_score': float(std_score),
                'config': {
                    'xai_method': config.xai_method,
                    'criterion': config.criterion,
                    'mode': config.mode,
                    'rule_name': config.rule_name
                }
            }, f, indent=2)
        print(f"\nDetailed results saved to {result_file}")

    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
