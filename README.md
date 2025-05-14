# EVO-LRP: Evolutionary Optimization of LRP for Interpretable Model Explanations

EVO-LRP is a framework that combines Layer-wise Relevance Propagation (LRP) with evolutionary optimization to find optimal LRP hyperparameters. The system uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to optimize LRP parameters for better interpretability of deep neural network decisions.

## Overview

This implementation allows you to:
- Run LRP on pre-trained models (VGG16 or custom MNIST CNN)
- Optimize LRP hyperparameters using CMA-ES
- Support different optimization modes and LRP rules
- Process both ImageNet and MNIST datasets

## Requirements

## Usage

### Basic Usage
Run main.py which will run the optimization algorithm and compute an optimal set of hyperparaters for a LRP configuration

### Command Line Arguments

#### Dataset and Model Configuration
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--dataset` | Dataset to use (choices: imagenet, mnist) | 'imagenet' | str |
| `--data_dir` | Path to dataset directory | None | str |
| `--model_path` | Path to custom model | None | str |
| `--batch_size` | Batch size for processing images | 8 | int |
| `--subset_size` | Subset size for ImageNet | 100 | int |
| `--image_size` | Image size for ImageNet preprocessing | 224 | int |

#### Optimization Configuration
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--optimization_type` | Type of optimization to perform | 'UniformRule' | str |
| `--correct_class_only` | Calculate relevance only for correct class | False | flag |
| `--rule` | LRP rule type | 'alpha_beta' | str |
| `--init_value` | Initial value for all rule parameters | 0.25 | float |

#### LRP Rule Base Values
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--alpha_beta_base` | Base value for alpha-beta rule | 1.0 | float |
| `--epsilon_base` | Base value for epsilon rule | 0.25 | float |
| `--gamma_base` | Base value for gamma rule | 0.25 | float |

#### Optimization Parameters
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--criterion` | Fitness criterion | 'sparseness' | str |
| `--generations` | Number of generations for optimization | 30 | int |
| `--population_size` | Population size for CMA-ES optimizer | 100 | int |
| `--init_sigma` | Initial sigma for CMA-ES optimizer | 0.3 | float |
| `--patience` | Early stopping patience (generations) | 10 | int |
| `--mean_weight` | Weight for mean score in fitness function | 0.8 | float |
| `--min_weight` | Weight for minimum score in fitness function | 0.2 | float |
| `--multi_opt` | Use multi-objective optimization | False | flag |
| `--multi_opt_criterion` | Multi-objective criteria | "sparseness,faithfulness" | str |

#### Evaluation Parameters
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--num_steps` | Number of steps for pixel flipping and sensitivity analysis | 5 | int |
| `--baseline` | Baseline for sensitivity analysis | 'zero' | str |

#### Output and Visualization
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--output_dir` | Directory to save results | './results' | str |
| `--save_params` | Save optimized parameters to JSON file | False | flag |
| `--no_plot` | Disable visualization plots | False | flag |
| `--save_plots` | Save visualization plots to timestamped folder | False | flag |
| `--verbose` | Enable verbose output | False | flag |

#### Hardware Configuration
| Argument | Description | Default | Type |
|----------|-------------|---------|------|
| `--device` | Device to use (cuda/cpu) | None | str |
| `--gpu_id` | GPU device ID to use | 0 | int |

### Available Choices

#### Datasets
- imagenet
- mnist

#### Optimization Types
- UniformRule
- UniformValue
- CompositeLayer

#### LRP Rules
- alpha_beta
- epsilon
- gamma
- zplus

#### Fitness Criteria
- sparseness
- faithfulness
- sensitivity
- pixel flipping

#### Baselines
- zero
- mean
- random

### Example Commands

1. Basic run with default settings:
```bash
python src/main.py
```

2. Run with specific dataset and optimization type:
```bash
python src/main.py --dataset mnist --optimization_type UniformValue --rule epsilon
```

3. Run with custom optimization parameters:
```bash
python src/main.py --generations 50 --population_size 150 --init_sigma 0.5
```

4. Run with multi-objective optimization:
```bash
python src/main.py --multi_opt --multi_opt_criterion sparseness,faithfulness
```

5. Run with visualization and parameter saving:
```bash
python src/main.py --save_plots --save_params --verbose
```

