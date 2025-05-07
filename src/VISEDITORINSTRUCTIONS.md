# LRP Visual Editor Instructions

This document explains how to use the `lrp_visual_editor.py` script to interactively visualize Layer-wise Relevance Propagation (LRP) heatmaps for different models, datasets, LRP configurations, and target classes.

## Purpose

The LRP Visual Editor provides an interactive interface to:
- View input images alongside their corresponding LRP relevance heatmaps.
- Explore how LRP assigns relevance to different input features for a model's prediction.
- Dynamically change the target class for which LRP is calculated and observe the changes in the heatmap.
- Switch between different images within a batch.
- Load and visualize LRP configurations optimized by the main EvoLRP script (`main.py`).
- Save visualizations for specific images and classes.

## Basic Usage

Run the script from the root directory of the `XAIProject` project:

```bash
python EvoLRP/src/lrp_visual_editor.py [arguments]
```

## Command-Line Arguments

The script accepts several arguments to configure its behavior:

**Dataset and Model:**

*   `--dataset {mnist,imagenet}`: Specifies the dataset to use (default: `mnist`).
*   `--data_dir <path>`: Path to the dataset directory (required for ImageNet if not in the default location `../../data/ImageNet`).
*   `--model_path <path>`: Path to a custom pre-trained model file. If not provided, uses a default MNIST CNN or a pre-trained VGG16 for ImageNet.
*   `--batch_size <int>`: Number of images to load for interactive selection (default: 8).
*   `--subset_size <int>`: For ImageNet, limits the number of images loaded from the dataset (default: 100).
*   `--image_size <int>`: For ImageNet, the size images are resized to (default: 224).

**LRP Configuration:**

*   `--params_file <path>`: **Crucial argument for using optimized parameters.** Path to a JSON file containing LRP parameters saved by `main.py` (e.g., from the `./results` directory).
*   `--lrp_type {UniformRule,UniformValue,UniformLayer}`: Specifies the LRP configuration type if *not* using `--params_file`. If `--params_file` is provided, the type is usually inferred from the file content (`optimization_type` key), overriding this argument. (default: `UniformRule`).
*   `--rule {alpha_beta,epsilon,gamma}`: Default LRP rule type to use if *not* using `--params_file`. (default: `alpha_beta`).
*   `--init_value <float>`: Default initial value for rule parameters if *not* using `--params_file`. (default: 2.0).

**Hardware:**

*   `--device {cpu,cuda,mps}`: Specifies the computation device. If None, tries `cuda`, then `mps`, then `cpu`.
*   `--gpu_id <int>`: GPU ID to use if `--device cuda` is selected (default: 0).

## Interactive Interface

Once launched, a Matplotlib window appears with two main panels and interactive controls:

1.  **Left Panel (Input Image):** Displays the currently selected input image. The title shows the image index within the loaded batch.
2.  **Right Panel (LRP Heatmap):** Displays the LRP relevance heatmap calculated for the currently selected image and target class.
    *   The title indicates the target class index, its name (if available), and the model's predicted probability for that class.
    *   The colormap (`bwr` - blue/white/red) indicates negative (blue), near-zero (white), and positive (red) relevance.
    *   A colorbar shows the scale of the relevance values.
3.  **'Class Rank' Slider:**
    *   Allows you to select the target class for LRP calculation.
    *   The slider value represents the *rank* of the class based on the model's prediction probability (0 = most probable class, 1 = second most probable, etc.).
    *   **Changing this slider recalculates the LRP heatmap for the new target class.** This is key to seeing how relevance differs for various potential predictions.
4.  **'Image Index' Slider:**
    *   Allows you to switch between the different images loaded in the batch (controlled by `--batch_size`).
    *   Changing this updates both the input image panel and recalculates the LRP heatmap for the new image (using the currently selected class rank).
5.  **'Save' Button:**
    *   Saves the current view (both panels) as a PNG image to the `lrp_editor_output` directory.
    *   The filename indicates the image index and the class index (e.g., `img0_class281.png`).

## Using Optimized Parameters

The primary use case after running `main.py` is to visualize the LRP heatmaps generated using the optimized parameters.

1.  **Run `main.py`:** Execute the main optimization script (e.g., `python EvoLRP/src/main.py --save_params --criterion sparseness ...`) which will save a JSON file in the `--output_dir` (default: `./results`). The filename will reflect the configuration (e.g., `results/UR_alphas_sparseness_mnist.json`).
2.  **Run `lrp_visual_editor.py` with `--params_file`:** Use the saved JSON file to load the optimized parameters.

    ```bash
    # Example for MNIST with UniformRule parameters optimized for sparseness
    python EvoLRP/src/lrp_visual_editor.py --dataset mnist --params_file results/UR_alphas_sparseness_mnist.json

    # Example for ImageNet with UniformLayer parameters optimized for faithfulness
    python EvoLRP/src/lrp_visual_editor.py --dataset imagenet --data_dir /path/to/imagenet --params_file results/UL_layer_rules_faithfulness_imagenet.json --batch_size 4
    ```

The editor will automatically detect the LRP type (`UniformRule`, `UniformValue`, `UniformLayer`) from the JSON file and configure the LRP computation accordingly. You can then use the 'Class Rank' slider to see the relevance maps for different classes using these optimized parameters.

## Output

*   An initial view (`visual_editor_initial_view.png`) is automatically saved to `lrp_editor_output` when the script starts.
*   Clicking the 'Save' button saves the current visualization to `lrp_editor_output` with a descriptive filename. 