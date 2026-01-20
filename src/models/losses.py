import torch
import torch.nn as nn
import torchvision.models as models

class MultiLayerPerceptualLoss(nn.Module):
    """
    Multi-layer perceptual loss that combines features from multiple VGG layers.

    Uses features from relu1_2, relu2_2, relu3_4, relu4_4, and relu5_4
    """
    def __init__(self):
        super().__init__()

        # VGG19 layer indices for different relu layers
        # relu1_2: 4, relu2_2: 9, relu3_4: 18, relu4_4: 27, relu5_4: 36
        self.layer_indices = [4, 9, 18, 27, 36]
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]  # Weight higher layers more

        vgg = models.vgg19(pretrained=True).features.eval()

        # Create feature extractors for each layer
        self.feature_extractors = nn.ModuleList()
        for layer_idx in self.layer_indices:
            feature_extractor = vgg[:layer_idx]
            for param in feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractors.append(feature_extractor)

        # VGG preprocessing normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, x):
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        return x

    def forward(self, generated, target):
        generated = self.preprocess(generated)
        target = self.preprocess(target)

        total_loss = 0.0
        for extractor, weight in zip(self.feature_extractors, self.weights):
            gen_features = extractor(generated)
            target_features = extractor(target)
            total_loss += weight * nn.functional.l1_loss(gen_features, target_features)

        return total_loss
