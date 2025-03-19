import torch
import os
from datasets import get_mnist_data
from models import train_mnist_model, save_model

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get training and test data
    train_loader = get_mnist_data(batch_size=64, train=True)
    test_loader = get_mnist_data(batch_size=64, train=False)
    
    # Train model
    model = train_mnist_model(train_loader, epochs=2, device=device)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    save_model(model, 'models/mnist_cnn.pth')
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
