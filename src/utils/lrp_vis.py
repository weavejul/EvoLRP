import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from models.cnn import CNN
from ..lrp.uniform_rule_lrp import UniformRuleLRP
from ..utils.visualization import plot_heatmap

def load_mnist(batch_size=1):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return testloader

def visualize_lrp_for_images(model, num_images=5):
    """Run LRP on multiple MNIST images and visualize results."""
    # Load MNIST test data
    testloader = load_mnist(batch_size=1)
    
    # Initialize LRP with uniform rule
    rule_params = torch.ones(8) * 0.5  # Example parameters for 8 layers
    lrp = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=rule_params)
    
    # Process images
    for i, (images, labels) in enumerate(testloader):
        if i >= num_images:
            break
            
        # Get model prediction
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Compute LRP relevance
        relevance = lrp.compute_relevance(images)
        
        # Create subplot for original image and heatmap
        plt.figure(figsize=(10, 4))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(images[0].squeeze().numpy(), cmap='gray')
        plt.title(f'Original Image\nTrue: {labels[0]}, Pred: {predicted[0]}')
        plt.axis('off')
        
        # Plot LRP heatmap
        plt.subplot(1, 2, 2)
        plot_heatmap(relevance)
        plt.title('LRP Heatmap')
        
        plt.tight_layout()
        plt.show()

def main():
    # Load the trained model
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()
    
    # Visualize LRP for 5 different images
    visualize_lrp_for_images(model, num_images=5)

if __name__ == '__main__':
    main()
