import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_segmentation_model(
    architecture='unet', 
    encoder='efficientnet-b0', 
    weights='imagenet', 
    in_channels=3, 
    classes=1, 
    **kwargs):
    """
    Create and return a segmentation model from SMP.
    
    Args:
        architecture (str): Architecture type (unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer)
        encoder (str): Encoder name (resnet34, efficientnet-b0, etc.)
        weights (str): Pretrained weights (imagenet, ...)
        in_channels (int): Number of input channels (3 for RGB)
        classes (int): Number of output channels (1 for binary segmentation)
        
    Returns:
        model: The initialized segmentation model
    """
    
    # Create model dictionary
    MODELS = {
        'unet': smp.Unet,
        'unet++': smp.UnetPlusPlus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3+': smp.DeepLabV3Plus,
        'linknet': smp.Linknet,
        'manet': smp.MAnet,
        'pan': smp.PAN, 
        'upernet': smp.UPerNet,
        'segformer': smp.Segformer,
        'dpt': smp.DPT
    }
    
    if architecture.lower() in MODELS:
        model = MODELS[architecture.lower()](
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    return model


class SegmentationModel(nn.Module):
    """Wrapper around SMP models with additional functionality if needed."""
    
    def __init__(self, architecture='unet', encoder='efficientnet-b0', weights='imagenet', in_channels=3, classes=1, **kwargs):
        super().__init__()
        self.model = get_segmentation_model(
            architecture=architecture,
            encoder=encoder,
            weights=weights,
            in_channels=in_channels,
            classes=classes,
            **kwargs
        )
    
    def forward(self, x):
        return self.model(x)


# Example usage to test the model
if __name__ == "__main__":
    # Create a model
    model = SegmentationModel(architecture='unet', encoder='efficientnet-b0', weights='imagenet', in_channels=3, classes=1)
    
    # Create a random input tensor
    x = torch.randn(2, 3, 512, 512)  # batch_size, channels, height, width
    
    # Forward pass
    y = model(x)
    
    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")