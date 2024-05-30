import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

from ternausnet.models import UNet16


class ModelManager:
    """
    Download model files and prepare segmentation models.

    """

    def __init__(self, model_name, pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained

    def create_model(self):
        if self.model_name == 'ternausnet':
            return self._create_tern_model()
        return self._create_smp_model()

    def _create_tern_model(self):
        return TernausnetWrapper(pretrained=self.pretrained)

    def _create_smp_model(self):
        return smp.Unet(
            encoder_name=self.model_name,
            encoder_weights="imagenet" if self.pretrained else "",
            in_channels=3,
            classes=1,
        )


class TernausnetWrapper(nn.Module):
    def __init__(self, output_dim=1, hidden_dim=64, pretrained=True):
        super().__init__()
        self.encoder = UNet16(num_classes=hidden_dim, pretrained=pretrained)
        self.hidden_layer = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.output_layer = nn.Conv2d(hidden_dim, output_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.nn.ReLU()(x)
        x = self.hidden_layer(x)
        x = torch.nn.ReLU()(x)
        return self.output_layer(x)
