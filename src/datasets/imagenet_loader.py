import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

def get_imagenet_data(config):
    """
    Load ImageNet dataset with proper preprocessing.
    
    Args:
        config: Configuration object with:
            - input_dir: Directory containing ImageNet data
            - batch_size: Batch size for the dataloader
            - resize: Image resize dimension (default 224 for standard models)
            - subset_size: Optional number of samples to use (for quick testing)
    
    Returns:
        DataLoader for ImageNet images
    """
    # Standard ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create transformation pipeline
    resize_dim = config.resize
    transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(resize_dim),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Ensure input directory exists
    if not os.path.exists(config.input_dir):
        raise FileNotFoundError(f"ImageNet directory not found at {config.input_dir}. Please ensure the directory exists and is correctly named.")
    
    # Check if we're using validation or training data
    # Default to validation folder for simpler evaluation
    data_dir = os.path.join(config.input_dir, 'val')
    if not os.path.exists(data_dir):
        # Try the training directory if val doesn't exist
        data_dir = os.path.join(config.input_dir, 'train')
        if not os.path.exists(data_dir):
            # If neither exists, use the provided directory directly
            data_dir = config.input_dir
    
    # Create dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Optionally use a subset for quicker testing
    if hasattr(config, 'subset_size') and config.subset_size > 0:
        indices = random.sample(range(len(dataset)), min(config.subset_size, len(dataset)))
        dataset = Subset(dataset, indices)
    
    # Create and return dataloader
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
