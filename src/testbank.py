"""
Script for testing the LRP model. Runs a given LRP model on the full 
dataset in input/ and computes the quantus metrics for the samples
that are correctly classified by the model. Outputs the mean and standard deviation
of the metrics.
"""

import argparse
import time
import pathlib
import quantus
from captum.attr import LayerAttribution, LayerGradCam, Lime
from skimage.segmentation import slic

import torch
from torchvision.models import vgg16, VGG16_Weights

from src.optimize import Optimizer
from src.lrp import LRPModel
from src.dataloader import get_data_loader
import numpy as np

def compute_scores(model, x_batch, y_batch, a_batch, device, explain_func, explain_func_kwargs):
    x_batch = x_batch.detach().cpu().numpy()
    y_batch = y_batch.detach().cpu().numpy()
    
    metric = quantus.AvgSensitivity(
        nr_samples=10,
        lower_bound=0.2,
        norm_numerator=quantus.norm_func.fro_norm,
        norm_denominator=quantus.norm_func.fro_norm,
        perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
    )   

    # metric = quantus.EffectiveComplexity(
    #     eps=1e-5,   
    # )

    # quantus.FaithfulnessCorrelation(
    #     nr_runs=100,  
    #     subset_size=224,  
    #     perturb_baseline="black",
    #     perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    #     similarity_func=quantus.similarity_func.correlation_pearson,  
    #     abs=False,  
    #     return_aggregate=False,
    # )
    scores = metric(    
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        device=device,
        explain_func=explain_func,
        explain_func_kwargs=explain_func_kwargs,
    )

    return scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        dest="input_dir",
        help="Input directory.",
        default="./input/",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Output directory.",
        default="./output/",
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", help="Batch size.", default=1, type=int
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
        default="lrp",
        type=str,
        choices=["lrp", "LayerGradCam", "Lime", "IntegratedGradients"]
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

    # set up parameters
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Using: {device}\n")

    data_loader = get_data_loader(config)

    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    model.eval()

    # set up LRP model
    if config.mode == "uniform_rule":
        lrp_model = LRPModel.from_uniform_rule(model=model, rule_name=config.rule_name, per_layer_values=[0] * 40)
    elif config.mode == "uniform_value":
        lrp_model = LRPModel.from_uniform_value(model=model, per_layer_rules=config.per_layer_rules)
    elif config.mode == "layer_type":
        lrp_model = LRPModel.from_layer_type(model=model, layer_type_config=config.layer_type_config)
    elif config.mode == "composite_layer":
        lrp_model = LRPModel.from_composite_layer(model=model, composite_config=config.composite_layer_config)
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    lime = Lime(model)

    scores = np.array([])
    for i, (x, y) in enumerate(data_loader):
        x = x.to(device)
        y = y.to(device) 

        # remove mislabeled images
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
        # might have to fix types and shapes before calling optimizer
        def lrp_func(model, targets, **kwargs) -> np.ndarray:
            x = kwargs["inputs"]
            x = torch.tensor(x).to(device)
            a_batch = lrp_model(x)

            if a_batch.shape[0] != x.shape[0]:
                a_batch = a_batch.unsqueeze(0)

            return a_batch

        if config.xai_method == "lrp":
            a_batch = lrp_model.forward(x)
            a_batch = a_batch.detach().cpu().numpy()
            explain_func = lrp_func
            explain_func_kwargs = None
        else:
            explain_func = quantus.explain
            # Convert x to torch tensor and move to correct device
            # x = torch.tensor(x, device=device)

            if config.xai_method == "LayerGradCam":
                # layer_gradcam = LayerGradCam(model, model.features[-3])
                # attributions_lgc = layer_gradcam.attribute(x, target=y)
                # Add target layer for LayerGradCam
                explain_func_kwargs = {
                    "method": config.xai_method,
                    "gc_layer": model.features[-3],
                    "interpolate": (224, 224) # TODO: is this right - can also add method
                }
                a_batch = quantus.explain(model, x, y, kwargs=explain_func_kwargs)
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

                def lime_func(model, targets, **kwargs) -> np.ndarray:
                    x = kwargs["inputs"]
                    x = torch.tensor(x).to(device)
                    targets = torch.tensor(targets).to(device)
                    # Ensure feature mask matches batch dimension
                    # print(feature_masks.shape)
                    # curr_feature_mask = feature_masks.unsqueeze(1).repeat(1, x.shape[0], 1, 1)
                    curr_feature_mask = feature_masks.unsqueeze(1)
                    a_batch = lime.attribute(
                        x, 
                        target=targets,
                        feature_mask=curr_feature_mask,
                        n_samples=40,
                        perturbations_per_eval=16,
                        show_progress=True
                    )   

                    if a_batch.shape[0] != x.shape[0]:
                        a_batch = a_batch.unsqueeze(0)
                    a_batch = a_batch.detach().cpu().numpy()

                    return a_batch

                explain_func_kwargs = None
                explain_func = lime_func
                # print(feature_masks.shape)

                a_batch = lime.attribute(
                    x,
                    target=y,
                    feature_mask=feature_masks.unsqueeze(1),
                    n_samples=40,
                    perturbations_per_eval=16,
                    show_progress=True
                )   
                a_batch = a_batch.detach().cpu().numpy()
            else:
                explain_func_kwargs = {"method": config.xai_method}
                a_batch = quantus.explain(model, x, y, kwargs=explain_func_kwargs)

        if x.shape[0] != a_batch.shape[0]:
            a_batch = a_batch.unsqueeze(0)

        batch_scores = compute_scores(model, x, y, a_batch, device, explain_func, explain_func_kwargs)
        scores = np.concatenate((scores, np.array(batch_scores)))

    print(np.mean(scores))
    print(np.std(scores))