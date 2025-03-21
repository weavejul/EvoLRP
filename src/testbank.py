"""
Script for testing the LRP model. Runs a given LRP model on the full 
dataset in input/ and computes the quantus metrics for the samples
that are correctly classified by the model. Outputs the mean and standard deviation
of the metrics.
"""
import argparse
import time
import pathlib
from captum.attr import LayerAttribution, LayerGradCam, Lime, IntegratedGradients
from skimage.segmentation import slic

import torch
from torchvision.models import vgg16, VGG16_Weights

from datasets import get_mnist_data, get_imagenet_data
from models import load_model
from lrp import UniformRuleLRP
from evaluation import sparseness, pixel_flipping, average_sensitivity, faithfulness_correlation
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

def compute_scores(model, x_batch, y_batch, a_batch, num_steps, baseline, criterion):
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
            baseline=baseline
        )
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
        # higher is better
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    # Use a weighted combination of mean and minimum for robust optimization
    mean_score = sum(batch_scores) / len(batch_scores)
    
    return mean_score

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
        choices=["lrp", "LayerGradCam", "Lime", "IntegratedGradients"]
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
        default="optimized_alphas_in.json",
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
        # elif config.mode == "composite_layer":
        #     lrp_model = LRPModel.from_composite_layer(model=model, composite_config=config.composite_layer_config)
        else:
            raise ValueError(f"Invalid mode: {config.mode}")

    # set up XAI methods
    lime = Lime(model)
    ig = IntegratedGradients(model)
    gradcam = LayerGradCam(model, conv_layer)

    scores = np.array([])
    for i, (x, y) in enumerate(data_loader):
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
        # Check if we have any samples left
        if len(x) == 0:
            print("Warning: All predictions were incorrect in this batch")
            continue
        try:  # Add try-except to handle potential OOM errors gracefully
            if config.xai_method == "lrp":
                a_batch = lrp_model.compute_relevance(x, y)
            elif config.xai_method == "LayerGradCam":
                gradcam_attr = gradcam.attribute(
                    x,
                    target=y
                )
                # Use a more sophisticated interpolation method
                a_batch = LayerAttribution.interpolate(
                    gradcam_attr, 
                    x.shape[2:],
                    interpolate_mode='bicubic'  # Use bicubic interpolation instead of default
                )
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
            elif config.xai_method == "IntegratedGradients":
                a_batch, _ = ig.attribute(
                    x,
                    baselines=0,
                    target=y,
                    method='gausslegendre',
                    return_convergence_delta=True
                )
            else:
                raise ValueError(f"Invalid XAI method: {config.xai_method}")

            if x.shape[0] != a_batch.shape[0]:
                a_batch = a_batch.unsqueeze(0)
            batch_scores = compute_scores(model, x, y, a_batch, num_steps=5, baseline='zero', criterion=config.criterion)
            if type(batch_scores) == torch.Tensor:
                batch_scores = batch_scores.detach().cpu().numpy().item()
            if batch_scores is None:
                batch_scores = 0
            scores = np.append(scores, batch_scores)
            
            # Explicitly delete tensors to free memory
            del a_batch, x, y, y_pred, y_pred_classes, batch_scores
                
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
        # Filter out any NaN values before calculating statistics
        valid_scores = scores[~np.isnan(scores)]
        if valid_scores.size == 0:
            print("Warning: All scores were NaN")
            mean_score = float('nan')
            std_score = float('nan')
        else:
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
    
    print(f"Number of valid scores: {scores.size}")
    print(f"Mean score: {mean_score}")
    print(f"Standard deviation: {std_score}")
