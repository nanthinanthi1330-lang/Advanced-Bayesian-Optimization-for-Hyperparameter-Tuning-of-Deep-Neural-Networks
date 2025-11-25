
import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, base_filters=32, dropout=0.0):
        super().__init__()
        f = base_filters
        self.features = nn.Sequential(
            nn.Conv2d(3, f, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, f*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(f*2, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
