import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_heatmap(relevance_map, show=True, colorbar=True, center_zero=True):
    """
    Plots an LRP heatmap with bidirectional colors centered at zero.
    
    Args:
        relevance_map: The LRP relevance map
        show: Whether to call plt.show() (set to False when using in subplots)
        colorbar: Whether to show a colorbar
        center_zero: Whether to center the colormap at zero
    """
    # Convert to numpy array
    heatmap = relevance_map.squeeze().cpu().numpy()
    
    # Handle multi-channel relevance maps (like for ImageNet)
    if len(heatmap.shape) == 3 and heatmap.shape[0] == 3:
        # This is an RGB relevance map with shape (3, H, W)
        # Either combine channels (option 1) or use one channel (option 2)
        
        # Option 1: Average across channels to get a single heatmap
        heatmap = np.mean(heatmap, axis=0)
        
        # Option 2 (alternative): Just use the first channel
        # heatmap = heatmap[0]  # Use R channel only
    
    if center_zero:
        # Use a diverging colormap centered at zero
        cmap = 'bwr'  # blue-white-red: negative values are blue, positive are red
        
        # Find the absolute maximum value to make the colormap symmetric
        vmax = np.max(np.abs(heatmap))
        vmin = -vmax
        
        # Plot with symmetric color scaling
        im = plt.imshow(heatmap, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
    else:
        # Original behavior for when only positive values matter
        im = plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    
    if colorbar:
        plt.colorbar(im)
    
    if show:
        plt.show()