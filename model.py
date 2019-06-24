import torch.nn as nn
from torchvision.models.resnet import resnet18


class Model(nn.Module):

    def __init__(self, num_class):
        super(Model, self).__init__()

        # backbone
        basic_model, layers = resnet18(pretrained=True), []
        for name, module in basic_model.named_children():
            if isinstance(module, nn.Linear):
                continue
            layers.append(module)
        self.features = nn.Sequential(*layers)

        # classifier
        self.fc = nn.Linear(512, num_class)
        # self.fc = CapsuleLinear(out_capsules=num_class, in_length=64, out_length=32)

    def forward(self, x):
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        out = self.fc(feature)
        return out
