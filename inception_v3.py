import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights

def get_radimagenet_inception_v3(num_classes=165):
    """
    Creates an InceptionV3 model modified for RadImageNet.
    
    Modifications:
    1. Random weight initialization
    2. Global average pooling
    3. Dropout (0.5)
    4. Softmax output layer
    5. 165 output classes
    
    Args:
        num_classes (int): Number of output classes (default: 165 for RadImageNet)
    
    Returns:
        torch.nn.Module: Modified InceptionV3 model
    """
    # Load the model with random weights
    model = models.inception_v3(weights=None, aux_logits=False)
    
    # Modify the classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes),
        nn.Softmax(dim=1)
    )
    
    # Verify the architecture
    verify_architecture(model, num_classes)
    
    return model

def verify_architecture(model, expected_classes):
    """
    Verifies that the model architecture matches the requirements.
    
    Args:
        model (torch.nn.Module): The model to verify
        expected_classes (int): Expected number of output classes
    
    Raises:
        ValueError: If the model architecture doesn't match requirements
    """
    # Check output layer
    if not isinstance(model.fc[-1], nn.Softmax):
        raise ValueError("Final layer must be Softmax")
    
    # Check number of output features
    if model.fc[1].out_features != expected_classes:
        raise ValueError(f"Model must have {expected_classes} output classes, "
                        f"got {model.fc[1].out_features}")
    
    # Check dropout layer
    if not isinstance(model.fc[0], nn.Dropout) or model.fc[0].p != 0.5:
        raise ValueError("Must have dropout layer with p=0.5 before final layer")

if __name__ == "__main__":
    # Create and test the model
    model = get_radimagenet_inception_v3()
    
    # Print model summary
    print(model)
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be 1.0): {output.sum().item()}") 