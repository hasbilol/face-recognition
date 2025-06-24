import torch.nn as nn
import torchvision.models as models

class ResNet50_static(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet50_static, self).__init__()
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
